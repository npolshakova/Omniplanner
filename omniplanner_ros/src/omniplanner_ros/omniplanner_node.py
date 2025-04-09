import threading
import time
import uuid

import numpy as np
import rclpy
import tf2_ros
import tf_transformations
from hydra_ros import DsgSubscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_executor_interface.action_descriptions import ActionSequence, Follow
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from ros_system_monitor_msgs.msg import NodeInfoMsg
from visualization_msgs.msg import MarkerArray

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner_msgs.msg import GotoPointsGoalMsg


# TODO: this needs to move somewhere and become generic
def temp_compile_plan(plan, plan_id, robot_name, frame_id):
    actions = []
    for p in plan.plan:
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[0], p.goal[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[1], p.goal[1]])
        p_interp = np.vstack([xs, ys])
        actions.append(Follow(frame=frame_id, path2d=p_interp.T))

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


def get_robot_pose(
    tf_buffer, target_frame: str = "map", source_frame: str = "base_link"
) -> np.ndarray:
    """
    Looks up the transform from target_frame to source_frame and returns [x, y, z, yaw].

    """
    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(
            target_frame, source_frame, now, timeout=rclpy.Duration(seconds=1.0)
        )
        transform = tf_buffer.lookup_transform(target_frame, source_frame, now)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to Euler angles
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        return np.ndarray([translation.x, translation.y, translation.z, yaw])

    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise


# NOTE: What's the best way to deal with multiple robots / robot discovery?
# Probably tie into the general robot discovery mechanism we were thinking
# about for multi-robot SLAM. Listen for messages broadcast from each robot on
# shared "bus" topic that identifies the robot id and its transform names. In
# this node, we can store a mapping from "robot type" to supported planners (I
# guess, each plugin specifies what robots it can run on). For each robot that
# appears, we create a publisher that can publish compiled plans.  If there are
# multiple robots, goal messages may need to specify which robot they are for.
# Some, such as NaturalLanguage, don't need to specify which robot they are for
# because the robot allocation is explicitly part of the planning process.


class OmniPlannerRos(Node):
    def __init__(self):
        super().__init__("omniplanner_ros")
        self.get_logger().info("Setting up omniplanner")

        self.last_dsg_time = 0
        self.current_planner = None
        self.plan_time_start = None

        self.last_dsg_time_lock = threading.Lock()
        self.current_planner_lock = threading.Lock()
        self.plan_time_start_lock = threading.Lock()

        self.goto_plan_sub = self.create_subscription(
            GotoPointsGoalMsg,
            "~/goto_points_goal",
            self.goto_points_callback,
            1,
        )

        self.dsg_lock = threading.Lock()
        DsgSubscriber(self, "~/dsg_in", self.dsg_callback)

        # TODO: need to generalize this cross robots.
        # When we discover a new robot, we should create a new
        # publisher based on the information that the robot provides.
        # Then we can look up the relevant publisher in the {name: publishers} map
        self.compiled_plan_pub = self.create_publisher(
            ActionSequenceMsg, "~/compiled_plan_out", 1
        )

        self.compiled_plan_viz_pub = self.create_publisher(
            MarkerArray, "~/compiled_plan_viz_out", 1
        )

        # self.pddl_plan_sub = self.create_subscription(
        #    PddlGoalMsg, "~/pddl_goal", self.pddl_goal_callback, 1
        # )

        # self.nlp_plan_sub = self.create_subscription(
        #    NlpGoalMsg, "~/nlp_goal", self.nlp_goal_callback, 1
        # )

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )

    def dsg_callback(self, header, dsg):
        self.get_logger().warning("Setting DSG!")

        with self.last_dsg_time_lock and self.dsg_lock:
            self.dsg_last = dsg
            self.last_dsg_time = time.time()


    def hb_callback(self):
        with (
            self.last_dsg_time_lock
            and self.current_planner_lock
            and self.plan_time_start_lock
        ):
            dsg_age = time.time() - self.last_dsg_time
            have_recent_dsg = dsg_age < 15

            notes = ""
            if not have_recent_dsg:
                notes += f"No recent dsg ({dsg_age} (s) old)"

            if self.current_planner is None:
                if have_recent_dsg:
                    notes += "Ready to plan!"
            else:
                elapsed_planning_time = time.time() - self.plan_time_start
                notes += f"Running {self.current_planner} ({elapsed_planning_time} s)"

        status = NodeInfoMsg.NOMINAL
        if not have_recent_dsg:
            status = NodeInfoMsg.WARNING

        msg = NodeInfoMsg()
        msg.nickname = "omniplanner"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = status
        msg.notes = notes
        self.heartbeat_pub.publish(msg)

    # def nlp_goal_callback(self, msg):
    #    pass

    def get_spot_pose(self):
        return get_robot_pose(
            self.tf_buffer, target_frame="vision", source_frame="base_link"
        )

    def goto_points_callback(self, msg):
        """TODO: in reality, this callback (and the subscription) should be
        loaded from an external plugin
        """

        # The plugin should provide a function that is (msg, dsg) --> compiled
        # plan. Publish/visualizing the plan is handled by Omnimapper (although
        # it requires that the compiled plan implements the visualizer
        # interface).

        # We make the restriction that ALL world state is captured by either 1)
        # the scene graph (or scene graph replacement), or 2) the initial robot
        # poses

        with self.current_planner_lock and self.plan_time_start_lock:
            self.current_planner = "GotoPointsPlanner"
            self.plan_time_start = time.time()

        robot_poses = {"spot", self.get_spot_pose}
        goal = GotoPointsGoal(
            goal_points=msg.point_names_to_visit, robot_id=msg.robot_id
        )
        req = PlanRequest(
            domain=GotoPointsDomain(),
            goal=goal,
            robot_states=robot_poses,
        )

        with self.dsg_lock:
            plan = full_planning_pipeline(req, self.dsg)
        spot_path_frame = "vision"  # TODO: parameter
        compiled_plan = temp_compile_plan(
            plan, str(uuid.uuid4()), "spot", spot_path_frame
        )

        # 1. Publish compiled plan <-- probably also should happen from omniplanner, not plugin
        self.compiled_plan_pub(to_msg(compiled_plan))
        # 2. Publish compiled plan viz <-- should happen from omniplanner, not inside plugin
        self.compiled_plan_viz_pub(to_viz_msg(compiled_plan))

        with self.current_planner_lock and self.plan_time_start_lock:
            self.current_planner = None
            self.plan_time_start = None


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OmniPlannerRos()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
