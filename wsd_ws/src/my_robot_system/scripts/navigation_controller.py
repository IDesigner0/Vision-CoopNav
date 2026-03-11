#!/usr/bin/env python3
import rospy
import actionlib
import math
from geometry_msgs.msg import Pose2D, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus

class NavigationController:
    def __init__(self):
        self.current_target = None
        self.detection_status = "NOT_FOUND"
        
        # 🔥 新增：目标发布控制
        self.navigation_active = False
        self.last_goal_time = rospy.Time(0)
        self.goal_cooldown = rospy.Duration(10.0)  # 10秒冷却时间
        self.last_goal_position = None
        self.position_threshold = 0.5  # 目标移动至少0.5米才重新发送
        
        rospy.loginfo("=== 导航控制器启动（带目标过滤）===")
        
        # 订阅黄色娃娃位置
        rospy.Subscriber("/yellow_doll_position", Pose2D, self.position_callback)
        rospy.Subscriber("/detection_status", String, self.status_callback)
        
        # 🔥 新增：监控导航结果
        rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.result_callback)
        
        # 发布到move_base_simple/goal (简单接口)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        
        # 同时保留action client (完整接口)
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("等待move_base服务器...")
        
        if self.move_base_client.wait_for_server(rospy.Duration(5.0)):
            rospy.loginfo("move_base Action服务器连接成功")
        else:
            rospy.logwarn("move_base Action服务器连接失败，使用简单接口")
        
        rospy.loginfo("导航控制器初始化完成")
        
    def position_callback(self, msg):
        """接收到黄色娃娃位置"""
        rospy.loginfo("=== 收到目标位置 ===")
        rospy.loginfo("位置: x=%.2f, y=%.2f", msg.x, msg.y)
        
        self.current_target = msg
        
        # 🔥 修改：添加智能目标发布逻辑
        if self.should_send_goal(msg):
            # 先验证目标点是否可达
            if self.is_goal_reachable(msg):
                self.send_goal_to_move_base(msg)
            else:
                rospy.logwarn("❌ 目标点可能不可达，跳过导航")
        
    def should_send_goal(self, target_pose):
        """智能判断是否应该发送目标"""
        current_time = rospy.Time.now()
        
        # 检查是否正在导航
        if self.navigation_active:
            rospy.loginfo_throttle(5.0, "🚗 导航进行中，跳过新目标")
            return False
            
        # 检查冷却时间
        if (current_time - self.last_goal_time) < self.goal_cooldown:
            rospy.loginfo_throttle(5.0, "⏳ 目标发布冷却中...")
            return False
            
        # 检查是否与上次目标相同
        if self.last_goal_position:
            dx = target_pose.x - self.last_goal_position[0]
            dy = target_pose.y - self.last_goal_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < self.position_threshold:
                rospy.loginfo_throttle(5.0, "🎯 目标位置变化不大，跳过发布")
                return False
        
        return True
        
    def result_callback(self, msg):
        """处理导航结果"""
        status = msg.status.status
        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("🎉 导航成功到达目标!")
            self.navigation_active = False
            self.last_goal_position = None  # 重置目标位置
        elif status == GoalStatus.ABORTED:
            rospy.logwarn("❌ 导航失败: 目标不可达")
            self.navigation_active = False
        elif status == GoalStatus.REJECTED:
            rospy.logwarn("⚠️ 目标被规划器拒绝")
            self.navigation_active = False
        
    def status_callback(self, msg):
        """更新检测状态"""
        rospy.loginfo("检测状态: %s", msg.data)
        self.detection_status = msg.data
        
    def send_goal_to_move_base(self, target_pose):
        """发送目标到move_base"""
        rospy.loginfo("发送导航目标到move_base...")
        
        # 更新状态
        self.last_goal_time = rospy.Time.now()
        self.last_goal_position = (target_pose.x, target_pose.y)
        self.navigation_active = True
        
        # 方法1：使用简单接口（你手动测试成功的）
        self.send_simple_goal(target_pose)
        
        # 方法2：使用Action接口（注释掉，避免重复发送）
        # self.send_action_goal(target_pose)
        
    def is_goal_reachable(self, target_pose):
        """检查目标点是否可达"""
        try:
            # 获取机器人当前位置
            amcl_pose = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=2.0)
            robot_x = amcl_pose.pose.pose.position.x
            robot_y = amcl_pose.pose.pose.position.y
            rospy.loginfo("✅ 机器人定位正常")
            rospy.loginfo(f"机器人当前位置: ({robot_x:.2f}, {robot_y:.2f})")
            
            # 计算到目标的距离
            distance = ((target_pose.x - robot_x)**2 + (target_pose.y - robot_y)**2)**0.5
            rospy.loginfo(f"目标距离: {distance:.2f} 米")
            
            # 检查目标点是否在地图范围内
            try:
                map_info = rospy.wait_for_message("/map_metadata", rospy.AnyMsg, timeout=1.0)
                rospy.loginfo("✅ 地图数据可用")
            except:
                rospy.logwarn("⚠️ 无法获取地图元数据")
            
            # 这里可以添加更复杂的目标点验证逻辑
            rospy.loginfo(f"目标点坐标: ({target_pose.x:.2f}, {target_pose.y:.2f})")
            
            # 简单验证：距离不能为0，且应该在合理范围内
            if distance < 0.1:
                rospy.logwarn("❌ 目标点与当前位置太近")
                return False
            elif distance > 10.0:
                rospy.logwarn("⚠️ 目标点距离较远，可能无法到达")
                
            return True
            
        except Exception as e:
            rospy.logwarn(f"目标点验证失败: {e}")
            return False

    def send_simple_goal(self, target_pose):
        """使用/move_base_simple/goal接口"""
        try:
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            
            goal.pose.position.x = target_pose.x
            goal.pose.position.y = target_pose.y
            goal.pose.position.z = 0.0
            goal.pose.orientation.x = 0.0
            goal.pose.orientation.y = 0.0
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 1.0
            
            self.goal_pub.publish(goal)
            rospy.loginfo("✅ 通过简单接口发送目标: (%.2f, %.2f)", target_pose.x, target_pose.y)
            
        except Exception as e:
            rospy.logerr("发送简单目标失败: %s", str(e))
            
    def send_action_goal(self, target_pose):
        """使用Action接口"""
        try:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            
            goal.target_pose.pose.position.x = target_pose.x
            goal.target_pose.pose.position.y = target_pose.y
            goal.target_pose.pose.orientation.w = 1.0
            
            self.move_base_client.send_goal(goal)
            rospy.loginfo("✅ 通过Action接口发送目标")
            
        except Exception as e:
            rospy.logerr("发送Action目标失败: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('navigation_controller')
    try:
        controller = NavigationController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("导航控制器关闭")
