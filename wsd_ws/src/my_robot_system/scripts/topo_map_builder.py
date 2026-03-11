#!/usr/bin/env python3
"""
拓扑语义地图构建器（半自动模式）
功能：
  - 在 RViz 中点击 2D Nav Goal 或 Publish Point
  - 自动记录点击位置为拓扑节点
  - 提示用户输入语义标签（如 potted_plant, green_box_zone）
  - 自动生成连通边（距离 < threshold）
  - 保存为 topo_map.yaml
"""

import rospy
import yaml
import math
from geometry_msgs.msg import PoseStamped, PointStamped
import os

class TopoMapBuilder:
    def __init__(self):
        rospy.init_node('topo_map_builder', anonymous=True)
        
        # 参数
        self.map_file = rospy.get_param('~output_file', 'topo_map.yaml')
        self.connectivity_threshold = rospy.get_param('~connectivity_threshold', 3.0)  # 米
        
        # 存储节点
        self.nodes = []  # 每个元素: {'id': str, 'x': float, 'y': float, 'yaw': float, 'tags': [str]}
        
        # 订阅 RViz 的两种常用点击方式
        self.nav_goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.point_sub = rospy.Subscriber('/clicked_point', PointStamped, self.point_callback)
        
        rospy.loginfo("🟢 拓扑地图构建器启动！")
        rospy.loginfo("👉 在 RViz 中右键 -> '2D Nav Goal' 或 'Publish Point' 来添加节点")
        rospy.loginfo("📝 每次点击后，请在终端输入语义标签（多个用空格分隔，如: potted_plant near_window）")
        rospy.loginfo("🛑 按 Ctrl+C 保存地图并退出")

    def goal_callback(self, msg):
        """处理 2D Nav Goal（带朝向）"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        # 从四元数计算 yaw（简化版）
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.add_node(x, y, yaw)

    def point_callback(self, msg):
        """处理 Publish Point（无朝向，默认 yaw=0）"""
        x = msg.point.x
        y = msg.point.y
        self.add_node(x, y, 0.0)

    def add_node(self, x, y, yaw):
        node_id = f"node_{len(self.nodes):02d}"
        rospy.loginfo(f"📍 捕获新节点: ({x:.2f}, {y:.2f}) @ {yaw:.2f} rad → ID: {node_id}")
        
        # 获取用户输入的语义标签
        try:
            tags_input = input("请输入语义标签（空格分隔，直接回车跳过）: ").strip()
            tags = tags_input.split() if tags_input else []
        except KeyboardInterrupt:
            rospy.logwarn("用户中断输入")
            tags = []
        
        self.nodes.append({
            'id': node_id,
            'x': round(x, 3),
            'y': round(y, 3),
            'yaw': round(yaw, 3),
            'tags': tags
        })
        rospy.loginfo(f"✅ 节点 {node_id} 已添加，标签: {tags}")

    def generate_edges(self):
        """基于距离阈值自动生成边"""
        edges = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = self.nodes[i]['x'], self.nodes[i]['y']
                xj, yj = self.nodes[j]['x'], self.nodes[j]['y']
                dist = math.hypot(xi - xj, yi - yj)
                if dist <= self.connectivity_threshold:
                    edges.append({
                        'from': self.nodes[i]['id'],
                        'to': self.nodes[j]['id'],
                        'cost': round(dist, 2)
                    })
                    # 双向边（无向图）
                    edges.append({
                        'from': self.nodes[j]['id'],
                        'to': self.nodes[i]['id'],
                        'cost': round(dist, 2)
                    })
        return edges

    def save_to_yaml(self):
        if not self.nodes:
            rospy.logwarn("⚠️ 未添加任何节点，跳过保存")
            return

        edges = self.generate_edges()
        
        map_data = {
            'nodes': [
                {
                    'id': node['id'],
                    'x': node['x'],
                    'y': node['y'],
                    'yaw': node['yaw'],
                    'tags': node['tags']
                }
                for node in self.nodes
            ],
            'edges': edges
        }

        # 确保目录存在
        output_path = os.path.expanduser(self.map_file)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(map_data, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        rospy.loginfo(f"💾 拓扑语义地图已保存至: {output_path}")
        rospy.loginfo(f"📊 共 {len(self.nodes)} 个节点, {len(edges)} 条边")

    def run(self):
        rospy.loginfo("⏳ 等待用户点击...")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.save_to_yaml()

if __name__ == '__main__':
    builder = TopoMapBuilder()
    builder.run()
