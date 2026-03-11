#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalStatus

class YellowDollDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/overhead_cam/image_raw", Image, self.image_callback)
        self.position_pub = rospy.Publisher("/yellow_doll_position", Pose2D, queue_size=10)
        self.status_pub = rospy.Publisher("/detection_status", String, queue_size=10)

        # 目标发布
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

        # 导航状态监控
        self.result_sub = rospy.Subscriber("/move_base/result", MoveBaseActionResult, self.result_callback)

        # 机器人位置订阅（用于计算安全距离）
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.robot_pose_callback)

        # 黄色颜色范围
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        # 目标发布控制
        self.navigation_active = False
        self.last_goal_time = rospy.Time(0)
        self.goal_cooldown = rospy.Duration(10.0)
        self.last_goal_position = None
        self.position_threshold = 0.5

        # 🔥 新增：机器人当前位置
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_pose_received = False

        # 🔥 新增：安全距离参数
        self.safe_distance = 0.8  # 距离玩偶多少米停下

        rospy.loginfo("黄色玩偶检测器启动（带安全距离导航）")

    def robot_pose_callback(self, msg):
        """更新机器人当前位置"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_pose_received = True

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 🔥 修复：适应不同OpenCV版本的返回值
            contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 处理不同版本的返回值
            if len(contours_result) == 3:
                # OpenCV 3.x: image, contours, hierarchy
                _, contours, _ = contours_result
            elif len(contours_result) == 2:
                # OpenCV 4.x: contours, hierarchy
                contours, _ = contours_result
            else:
                contours = contours_result[0] if contours_result else []

            detection_made = False

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    world_x, world_y = self.pixel_to_world(cx, cy)

                    position = Pose2D()
                    position.x = world_x
                    position.y = world_y
                    position.theta = 0

                    self.position_pub.publish(position)
                    self.status_pub.publish("FOUND")
                    detection_made = True

                # 智能目标发布
                    self.smart_goal_publish(world_x, world_y)

                # 可视化
                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.putText(cv_image, f"Yellow Doll ({world_x:.2f}, {world_y:.2f})", 
                               (cx-80, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if not detection_made:
                self.status_pub.publish("NOT_FOUND")

            cv2.imshow("Yellow Doll Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Detection error: %s", str(e))

    def smart_goal_publish(self, world_x, world_y):
        """智能决定是否发送目标点"""
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
            dx = world_x - self.last_goal_position[0]
            dy = world_y - self.last_goal_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < self.position_threshold:
                rospy.loginfo_throttle(5.0, "🎯 目标位置变化不大，跳过发布")
                return False

        # 检查机器人位置是否就绪
        if not self.robot_pose_received:
            rospy.logwarn("❌ 机器人位置未知，无法计算安全距离")
            return False

        # 发送目标
        self.send_navigation_goal(world_x, world_y)
        return True

    def send_navigation_goal(self, target_x, target_y):
        """发送导航目标（带安全距离）"""
        try:
            # 🔥 计算安全距离的目标点
            # 计算机器人到目标的向量
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            distance_to_target = math.sqrt(dx*dx + dy*dy)

            if distance_to_target < self.safe_distance:
                rospy.logwarn("❌ 目标距离太近，无法设置安全距离")
                return

            # 计算单位向量
            if distance_to_target > 0:
                unit_dx = dx / distance_to_target
                unit_dy = dy / distance_to_target
            else:
                unit_dx = 0
                unit_dy = 0

            # 计算安全位置（从目标位置后退safe_distance）
            safe_x = target_x - self.safe_distance * unit_dx
            safe_y = target_y - self.safe_distance * unit_dy

            # 计算朝向角度（面向目标）
            target_angle = math.atan2(dy, dx)

            goal = PoseStamped()
            goal.header = Header()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "map"

            goal.pose.position.x = safe_x
            goal.pose.position.y = safe_y
            goal.pose.position.z = 0.0

            # 设置朝向（面向玩偶）
            goal.pose.orientation.z = math.sin(target_angle / 2.0)
            goal.pose.orientation.w = math.cos(target_angle / 2.0)

            self.goal_pub.publish(goal)
            self.last_goal_time = rospy.Time.now()
            self.last_goal_position = (target_x, target_y)  # 保存原始目标位置
            self.navigation_active = True

            rospy.loginfo(f"🎯 发送安全导航目标")
            rospy.loginfo(f"📍 安全位置: ({safe_x:.2f}, {safe_y:.2f})")
            rospy.loginfo(f"🎯 玩偶位置: ({target_x:.2f}, {target_y:.2f})")
            rospy.loginfo(f"📏 安全距离: {self.safe_distance}米")

        except Exception as e:
            rospy.logerr(f"发送目标失败: {e}")

    def result_callback(self, msg):
        """处理导航结果"""
        status = msg.status.status
        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("🎉 导航成功到达安全位置!")
            self.navigation_active = False
        elif status == GoalStatus.ABORTED:
            rospy.logwarn("❌ 导航失败: 目标不可达")
            self.navigation_active = False
        elif status == GoalStatus.REJECTED:
            rospy.logwarn("⚠️ 目标被规划器拒绝")
            self.navigation_active = False

    def pixel_to_world(self, pixel_x, pixel_y):
        try:
            H = np.load('/home/wheeltec/wsd_ws/src/my_robot_system/scripts/homography.npy')
            src = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src[None, :, :], H)

        # 调试信息（可选）
            rospy.logdebug(f"透视变换结果形状: {dst.shape}, 内容: {dst}")

        # 安全解包 - 多种方法
            if dst.shape == (1, 1, 2):
                world_x, world_y = dst[0, 0, 0], dst[0, 0, 1]
            elif dst.shape == (1, 1, 3):  # 如果有齐次坐标
                world_x, world_y = dst[0, 0, 0], dst[0, 0, 1]
            else:
            # 通用方法：取第一个点的前两个坐标
                world_x, world_y = float(dst.flat[0]), float(dst.flat[1])

            rospy.logdebug(f"转换后坐标: ({world_x:.2f}, {world_y:.2f})")
            return world_x, world_y

        except Exception as e:
            rospy.logerr(f"坐标转换错误: {e}")
            return 0.0, 0.0

if __name__ == '__main__':
    rospy.init_node('yellow_doll_detector')
    detector = YellowDollDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("正在关闭检测器...")
    finally:
        cv2.destroyAllWindows()
