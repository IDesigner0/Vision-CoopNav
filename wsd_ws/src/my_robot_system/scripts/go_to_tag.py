#!/usr/bin/env python3
import rospy
import tf
import math
import actionlib
from topo_manager import TopoMap
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class SemanticNavigator:
    def __init__(self):
        self.topo = TopoMap()
        self.tf_listener = tf.TransformListener()
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        rospy.loginfo("⌛ Waiting for move_base server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("✅ Semantic Navigator Online.")

    def get_robot_node(self):
        """自动定位：寻找距离机器人当前坐标最近的拓扑节点"""
        try:
            # 查找 base_link 在 map 下的坐标
            (trans, _) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            rx, ry = trans[0], trans[1]
            
            best_node = None
            min_dist = 999.0
            for nid, data in self.topo.nodes.items():
                d = math.hypot(rx - data['x'], ry - data['y'])
                if d < min_dist:
                    min_dist = d
                    best_node = nid
            return best_node
        except:
            rospy.logwarn("TF lookup failed, using default node_05")
            return "node_05"

    def go_to_tag(self, target_tag):
        # 1. 查找目标节点 ID
        goal_node = self.topo.find_node_by_tag(target_tag)
        if not goal_node:
            rospy.logerr(f"Target tag '{target_tag}' not in map!")
            return False

        # 2. 自动寻找起始节点
        start_node = self.get_robot_node()
        rospy.loginfo(f"��️ Planning: {start_node} -> {goal_node}")

        # 3. Dijkstra 规划路径
        path = self.topo.dijkstra(start_node, goal_node)
        if not path:
            rospy.logerr("No path available!")
            return False

        # 4. 依次导航
        for node_id in path[1:]:
            rospy.loginfo(f"➡️ Navigating to: {node_id}")
            if not self.execute_move(node_id):
                rospy.logerr(f"Failed to reach {node_id}")
                return False
        
        rospy.loginfo("�� Destination Reached!")
        return True

    def execute_move(self, node_id):
        node_data = self.topo.nodes[node_id]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        goal.target_pose.pose.position.x = node_data['x']
        goal.target_pose.pose.position.y = node_data['y']
        
        # 转换 Yaw 为四元数
        q = tf.transformations.quaternion_from_euler(0, 0, node_data['yaw'])
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        return self.move_base_client.get_state() == 3

if __name__ == '__main__':
    rospy.init_node('semantic_navigator')
    # 从参数服务器读取目标，例如：rosrun my_robot_system go_to_tag.py _target:=potted_plant
    target = rospy.get_param('~target', 'tool_box')
    
    nav = SemanticNavigator()
    nav.go_to_tag(target)
