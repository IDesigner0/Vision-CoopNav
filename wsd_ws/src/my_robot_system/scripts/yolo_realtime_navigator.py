#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import tf2_ros
import signal
import os

from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Ultralytics YOLO
from ultralytics import YOLO

class RealtimeNavigator:
    def __init__(self):
        rospy.init_node('realtime_navigator', anonymous=True)

        # === 1. 路径与显示配置 ===
        dataset_path = '/home/wheeltec/wsd_ws/src/my_robot_system/dataset'
        self.display_scale = 0.9  # 缩放比例：0.7表示缩小30%，方便看全画面
        
        self.rtsp_url = rospy.get_param('~rtsp_url',
            'rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1')
        self.homography_path = rospy.get_param('~homography_path',
            '/home/wheeltec/wsd_ws/src/my_robot_system/scripts/homography_real2.npy')
        
        self.global_model_path = os.path.join(dataset_path, 'tool_robot.pt')
        self.local_model_path = os.path.join(dataset_path, 'tool_box.pt')

        # === 2. 导航与稳定参数 ===
        self.safe_distance = 0.5
        self.confidence_threshold = 0.6
        self.stability_duration = 15.0  
        
        self.last_seen_time = 0.0      
        self.is_stabilizing = False    
        self.current_state_text = "Waiting..." # 状态：Waiting..., Navigating..., GOAL Reached!

        # === 3. 类别与配色 ===
        self.target_classes_global = ["tool_box", "robot"]
        self.navigation_target_classes = ["tool_box"]
        self.class_colors = {"tool_box": (0, 255, 255), "robot": (255, 0, 255)}

        # === 4. 状态变量 ===
        self.visited_targets = []
        self.goal_published = False
        self.goal_reached = False 

        # 加载硬件与模型
        try:
            self.H = np.load(self.homography_path)
            self.yolo_global = YOLO(self.global_model_path) if os.path.exists(self.global_model_path) else None
            self.yolo_local = YOLO(self.local_model_path) if os.path.exists(self.local_model_path) else None
            rospy.loginfo("✅ 系统初始化完成")
        except Exception as e:
            rospy.logfatal(f"❌ 初始化失败: {e}")
            rospy.signal_shutdown("Init failed")

        self.cap = None
        self.connect_camera_low_latency()
        self.bridge = CvBridge()
        self.local_image = None
        self.local_image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.local_image_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.move_base_result_callback)

        self.running = True
        signal.signal(signal.SIGINT, self.shutdown_hook)

    def local_image_callback(self, msg):
        try: self.local_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def connect_camera_low_latency(self):
        gst_str = (f"rtspsrc location={self.rtsp_url} latency=0 ! rtph264depay ! h264parse ! "
                   "avdec_h264 ! videoconvert ! appsink")
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_latest_frame(self):
        ret, frame = False, None
        if self.cap:
            for _ in range(2): ret, frame = self.cap.read()
        return ret, frame

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
            return trans.transform.translation.x, trans.transform.translation.y
        except: return None, None

    def detect_and_annotate_global(self, frame):
        if self.yolo_global is None: return frame.copy(), []
        results = self.yolo_global(frame, verbose=False, conf=self.confidence_threshold)
        ann_frame = frame.copy()
        valid_detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = self.yolo_global.names[cls_id]
                if class_name in self.target_classes_global:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    px, py = int((x1 + x2) / 2), int(y2)
                    dst = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H)
                    world_x, world_y = float(dst.flat[0]), float(dst.flat[1])
                    valid_detections.append((world_x, world_y, class_name))
                    
                    # 绘制检测框和置信度 (用于论文展示)
                    color = self.class_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(ann_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(ann_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return ann_frame, valid_detections

    def annotate_local_image(self, frame):
        if self.yolo_local is None or frame is None: return frame
        results = self.yolo_local(frame, verbose=False, conf=self.confidence_threshold)
        ann_frame = frame.copy()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = self.yolo_local.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(ann_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(ann_frame, f"local {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return ann_frame

    def publish_goal(self, world_x, world_y):
        rx, ry = self.get_robot_pose()
        rx, ry = (rx, ry) if rx is not None else (0.0, 0.0)
        dx, dy = world_x - rx, world_y - ry
        dist = math.hypot(dx, dy)
        gx, gy = (world_x - self.safe_distance*(dx/dist), world_y - self.safe_distance*(dy/dist)) if dist > self.safe_distance else (world_x, world_y)
        angle = math.atan2(dy, dx)
        
        goal = PoseStamped()
        goal.header.stamp, goal.header.frame_id = rospy.Time.now(), "map"
        goal.pose.position.x, goal.pose.position.y = gx, gy
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(angle/2), math.cos(angle/2)
        
        self.goal_pub.publish(goal)
        self.goal_published = True
        self.current_state_text = "Navigating..."
        rospy.loginfo("�� Goal sent to MoveBase")

    def move_base_result_callback(self, msg):
        if msg.status.status == 3:
            self.goal_reached = True
            self.goal_published = False
            self.current_state_text = "GOAL Reached!"
            rospy.loginfo("�� Reached Goal")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.running:
            ret, global_frame = self.get_latest_frame()
            if not ret:
                self.connect_camera_low_latency()
                continue

            # 1. 全局检测逻辑
            ann_global, targets = self.detect_and_annotate_global(global_frame)
            
            # 2. 静默稳定计时与状态切换
            if not self.goal_reached and not self.goal_published:
                current_boxes = [t for t in targets if t[2] in self.navigation_target_classes]
                if current_boxes:
                    now = rospy.get_time()
                    if not self.is_stabilizing:
                        self.last_seen_time = now
                        self.is_stabilizing = True
                        self.current_state_text = "Waiting..."
                    else:
                        if now - self.last_seen_time >= self.stability_duration:
                            self.publish_goal(current_boxes[0][0], current_boxes[0][1])
                            self.is_stabilizing = False
                else:
                    self.is_stabilizing = False
                    self.current_state_text = "Waiting..."

            # 3. 局部检测处理
            local_raw = self.local_image if self.local_image is not None else np.zeros((480, 640, 3), np.uint8)
            ann_local = self.annotate_local_image(local_raw)

            # 4. 拼接图像
            w = max(ann_global.shape[1], ann_local.shape[1])
            h1 = int(ann_global.shape[0] * w / ann_global.shape[1])
            h2 = int(ann_local.shape[0] * w / ann_local.shape[1])
            i1 = cv2.resize(ann_global, (w, h1))
            i2 = cv2.resize(ann_local, (w, h2))
            combined = np.vstack((i1, i2))

            # 5. 左下角状态文本叠加 (符合论文排版)
            # 背景条增强对比度
            cv2.rectangle(combined, (5, combined.shape[0]-45), (450, combined.shape[0]-5), (0,0,0), -1)
            # 根据状态改变文字颜色
            status_color = (0, 255, 0) if "Reached" in self.current_state_text else (0, 165, 255)
            cv2.putText(combined, f"Status: {self.current_state_text}", 
                        (15, combined.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

            # 6. 窗口缩放显示
            disp_w = int(combined.shape[1] * self.display_scale)
            disp_h = int(combined.shape[0] * self.display_scale)
            display_img = cv2.resize(combined, (disp_w, disp_h))

            cv2.imshow("System Monitor (Scaling)", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            rate.sleep()

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    def shutdown_hook(self, signum, frame):
        self.running = False
        rospy.signal_shutdown("User exit")

if __name__ == '__main__':
    try:
        navigator = RealtimeNavigator()
        navigator.run()
    except Exception as e: print(f"Error: {e}")
