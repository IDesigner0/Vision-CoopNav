#!/usr/bin/env python3
import rospy
import actionlib
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String # 用于同步状态到 UI
from topo_manager import TopoMap
from local_detector import LocalDetector

class ActiveSearcher:
    def __init__(self, target_class="tool_box"):
        self.target_class = target_class
        self.topo = TopoMap()
        self.detector = LocalDetector(target_class=target_class)
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.status_pub = rospy.Publisher('/search_status', String, queue_size=1)
        self.tf_listener = tf.TransformListener()
        self.move_base_client.wait_for_server()

    def set_status(self, msg):
        rospy.loginfo(msg)
        self.status_pub.publish(msg)

    def run(self):
        self.set_status("SEARCH MODE START")
        candidates = [nid for nid, info in self.topo.nodes.items() if info['tag'] not in ['start', 'end']]
        
        # 贪心排序逻辑
        search_queue = candidates # 简便起见直接使用，可按上版加入排序

        for node_id in search_queue:
            if rospy.is_shutdown(): break
            self.set_status(f"GOTO: {node_id}")
            if self.navigate_to_node(node_id):
                self.set_status(f"SCANNING @ {node_id}")
                if self.perform_rotation_scan():
                    self.set_status("TARGET FOUND!")
                    self.approach_and_stop()
                    return True
        return False

    def navigate_to_node(self, node_id):
        info = self.topo.nodes[node_id]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x, goal.target_pose.pose.position.y = info['x'], info['y']
        q = tf.transformations.quaternion_from_euler(0, 0, info['yaw'])
        goal.target_pose.pose.orientation.x, goal.target_pose.pose.orientation.y, \
        goal.target_pose.pose.orientation.z, goal.target_pose.pose.orientation.w = q
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result(rospy.Duration(40))
        return self.move_base_client.get_state() == 3

    def perform_rotation_scan(self):
        vel = Twist()
        vel.angular.z = 0.4
        start_t = rospy.get_time()
        while rospy.get_time() - start_t < 16.0:
            if self.detector.is_target_detected():
                self.cmd_vel_pub.publish(Twist())
                return True
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.1)
        self.cmd_vel_pub.publish(Twist())
        return False

    def approach_and_stop(self):
        vel = Twist()
        vel.linear.x = 0.15
        for _ in range(15):
            self.cmd_vel_pub.publish(vel)
            rospy.sleep(0.1)
        self.cmd_vel_pub.publish(Twist())
