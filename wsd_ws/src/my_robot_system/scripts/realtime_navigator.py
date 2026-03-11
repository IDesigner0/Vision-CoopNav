#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseActionResult
import signal
import sys

class RealtimeNavigator:
    def __init__(self):
        rospy.init_node('realtime_navigator', anonymous=True)

        # 参数配置
        self.rtsp_url = rospy.get_param('~rtsp_url', 
            'rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1')
        self.homography_path = rospy.get_param('~homography_path',
            '/home/idesigner/wsd_ws/src/my_robot_system/scripts/homography_real2.npy')
        self.safe_distance = rospy.get_param('~safe_distance', 0.8)
        self.process_interval = rospy.get_param('~process_interval', 2.0)
        self.relocation_threshold = rospy.get_param('~relocation_threshold', 0.2)  # 移动多少米算“新目标”

        # 状态变量
        self.last_target_world = None      # 上次成功发布的【玩偶真实位置】(world_x, world_y)
        self.goal_published = False        # 是否已发布导航目标
        self.goal_reached = False          # 是否已到达（来自 move_base result）

        # 加载单应矩阵 H
        try:
            self.H = np.load(self.homography_path)
            assert self.H.shape == (3, 3), f"Expected (3,3), got {self.H.shape}"
            rospy.loginfo("✅ 单应矩阵加载成功")
        except Exception as e:
            rospy.logfatal(f"❌ 无法加载单应矩阵: {e}")
            rospy.signal_shutdown("Homography load failed")

        # 初始化摄像头
        self.cap = None
        self.connect_camera()

        # TF2 和 ROS 通信
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.move_base_result_callback)

        # 控制变量
        self.last_process_time = 0.0
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown_hook)

    def move_base_result_callback(self, msg):
        """监听导航结果"""
        if msg.status.status == 3:  # SUCCEEDED
            rospy.loginfo("🎉 目标已成功到达！")
            self.goal_reached = True
            # 注意：不重置 last_target_world！它用于判断后续是否移动
        else:
            rospy.logwarn(f"⚠️ 导航未成功，状态码: {msg.status.status}")
            # 可选：失败时不清除状态，避免震荡

    def connect_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            rospy.logerr("❌ 无法打开 RTSP 视频流！")
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        rospy.loginfo("📹 RTSP 视频流已连接")
        return True

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            return trans.transform.translation.x, trans.transform.translation.y
        except Exception as e:
            rospy.logwarn_throttle(5, f"TF2 lookup failed: {e}")
            return None, None

    def detect_yellow_target(self, frame, visualize=False):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        debug_frame = frame.copy() if visualize else None

        if not contours:
            return None, None, debug_frame

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, None, debug_frame

        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        if visualize:
            cv2.drawContours(debug_frame, [largest], -1, (0, 255, 0), 2)
            cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(debug_frame, "Target", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        src = np.array([[cx, cy]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src[None, :, :], self.H)
        if dst.size >= 2:
            world_x, world_y = float(dst.flat[0]), float(dst.flat[1])
            return world_x, world_y, debug_frame
        else:
            rospy.logwarn("透视变换结果无效")
            return None, None, debug_frame

    def publish_goal(self, world_x, world_y):
        robot_x, robot_y = self.get_robot_pose()
        if robot_x is None:
            rospy.logwarn("使用默认机器人位置 (0, 0)")
            robot_x, robot_y = 0.0, 0.0

        dx = world_x - robot_x
        dy = world_y - robot_y
        dist = math.hypot(dx, dy)

        if dist < self.safe_distance:
            goal_x, goal_y = world_x, world_y
            rospy.logwarn("⚠️ 目标太近，直接导航至玩偶位置")
        else:
            goal_x = world_x - self.safe_distance * dx / dist
            goal_y = world_y - self.safe_distance * dy / dist

        angle = math.atan2(dy, dx)
        quat = Quaternion()
        quat.z = math.sin(angle / 2.0)
        quat.w = math.cos(angle / 2.0)

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation = quat

        self.goal_pub.publish(goal)
        self.goal_published = True
        self.goal_reached = False
        self.last_target_world = (world_x, world_y)  # 记录本次玩偶的真实位置
        rospy.loginfo(f"🚀 已发布导航目标: ({goal_x:.2f}, {goal_y:.2f}) | 玩偶位置: ({world_x:.2f}, {world_y:.2f})")

    def is_significantly_new_target(self, new_x, new_y):
        """判断新检测到的目标是否与 last_target_world 距离超过阈值"""
        if self.last_target_world is None:
            return True  # 第一次检测，肯定是新的
        old_x, old_y = self.last_target_world
        distance = math.hypot(new_x - old_x, new_y - old_y)
        rospy.logdebug(f"🎯 与上次目标距离: {distance:.2f} m (阈值: {self.relocation_threshold})")
        return distance >= self.relocation_threshold

    def run(self):
        rospy.loginfo("▶️ 实时导航节点启动，按 Ctrl+C 退出")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("⚠️ 视频流读取失败，尝试重连...")
                rospy.sleep(1.0)
                self.connect_camera()
                continue

            current_time = rospy.get_time()
            should_process = (current_time - self.last_process_time >= self.process_interval)

            # === 关键：始终允许检测和可视化，不因 goal_reached 跳过！===
            if should_process:
                self.last_process_time = current_time
                world_x, world_y, debug_frame = self.detect_yellow_target(frame, visualize=True)
                display_frame = debug_frame if debug_frame is not None else frame.copy()

                if world_x is not None:
                    rospy.loginfo(f"🎯 检测到玩偶: ({world_x:.2f}, {world_y:.2f})")

                    if self.is_significantly_new_target(world_x, world_y):
                        self.publish_goal(world_x, world_y)
                    else:
                        rospy.loginfo("📍 目标位置变化小，忽略重复检测")
                else:
                    rospy.loginfo_throttle(5, "🔍 未检测到黄色目标")

                # ====== 在左下角显示状态 ======
                height, width = display_frame.shape[:2]
                if self.goal_reached:
                    status_text = "STATUS: REACHED"
                    color = (0, 255, 0)  # green
                elif self.goal_published:
                    status_text = "STATUS: NAVIGATING"
                    color = (0, 165, 255)  # orange (BGR)
                else:
                    status_text = "STATUS: SEARCHING"
                    color = (255, 255, 255)  # white

                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = 10
                text_y = height - 20
                cv2.rectangle(display_frame, (text_x, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, status_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.imshow("Yellow Target Detection", display_frame)
                cv2.waitKey(1)
            else:
                # 非处理帧：显示原始画面
                cv2.imshow("Yellow Target Detection", frame)
                cv2.waitKey(1)

            rate.sleep()

        cv2.destroyAllWindows()
        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        rospy.loginfo("🛑 实时导航节点已关闭")

    def shutdown_hook(self, signum, frame):
        self.running = False
        rospy.signal_shutdown("User requested shutdown")


if __name__ == '__main__':
    try:
        navigator = RealtimeNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal(f"节点崩溃: {e}")
        raise
