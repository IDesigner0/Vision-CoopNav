#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import tf2_ros
import signal
import os
import threading

from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class RealtimeNavigator:
    def __init__(self):
        rospy.init_node('realtime_navigator', anonymous=True)

        # === 1. 路径与显示配置 (保持原有) ===
        dataset_path = '/home/wheeltec/wsd_ws/src/my_robot_system/dataset'
        self.display_scale = 0.9  
        
        self.rtsp_url = rospy.get_param('~rtsp_url', 'rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1')
        self.homography_path = rospy.get_param('~homography_path', '/home/wheeltec/wsd_ws/src/my_robot_system/scripts/homography_real2.npy')
        
        self.global_model_path = os.path.join(dataset_path, 'tool_robot.pt')
        self.local_model_path = os.path.join(dataset_path, 'tool_box.pt')

        # === 2. 状态变量 (新增搜索支持) ===
        self.current_state_text = "Waiting..." 
        self.global_detected_flag = False
        self.search_triggered = False
        self.confidence_threshold = 0.6
        self.navigation_target_classes = ["tool_box"]
        self.class_colors = {"tool_box": (0, 255, 255), "robot": (255, 0, 255)}

        # 加载硬件与模型
        try:
            self.H = np.load(self.homography_path)
            self.yolo_global = YOLO(self.global_model_path)
            self.yolo_local = YOLO(self.local_model_path)
            rospy.loginfo("✅ 视觉与模型初始化完成")
        except Exception as e:
            rospy.logerr(f"❌ 初始化失败: {e}")

        # === 3. 相机与通讯 ===
        self.cap = None
        self.connect_camera_low_latency()
        self.bridge = CvBridge()
        self.local_image = None
        
        # 订阅局部相机与搜索状态
        rospy.Subscriber('/camera/color/image_raw', Image, self.local_image_callback)
        rospy.Subscriber('/search_status', String, self.status_cb)
        
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown_hook)

    def local_image_callback(self, msg):
        try: self.local_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def status_cb(self, msg):
        """接收来自 active_searcher 的状态更新"""
        self.current_state_text = msg.data

    def connect_camera_low_latency(self):
        """保持原有的 GStreamer 连接方式防止卡顿"""
        gst_str = (f"rtspsrc location={self.rtsp_url} latency=0 ! rtph264depay ! h264parse ! "
                   "avdec_h264 ! videoconvert ! appsink")
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

    def get_latest_frame(self):
        ret, frame = False, None
        if self.cap:
            for _ in range(2): ret, frame = self.cap.read()
        return ret, frame

    def annotate(self, frame, model, is_global=True):
        """原汁原味的标注样式"""
        if frame is None: return np.zeros((480, 640, 3), np.uint8)
        results = model(frame, verbose=False, conf=self.confidence_threshold)
        ann_frame = frame.copy()
        found = False
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = model.names[cls_id]
                
                if class_name in ["tool_box", "robot"]:
                    found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    color = self.class_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(ann_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(ann_frame, f"{class_name} {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if is_global: self.global_detected_flag = found
        return ann_frame

    def run(self):
        # 启动后台监控：10秒没看到全局目标则启动搜索脚本
        threading.Thread(target=self.logic_timer, daemon=True).start()
        
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and self.running:
            ret, global_frame = self.get_latest_frame()
            if not ret:
                self.connect_camera_low_latency()
                continue

            # 1. 场端与机器人视角处理 (维持原有拼接逻辑)
            ann_global = self.annotate(global_frame, self.yolo_global, is_global=True)
            local_raw = self.local_image if self.local_image is not None else np.zeros((480, 640, 3), np.uint8)
            ann_local = self.annotate(local_raw, self.yolo_local, is_global=False)

            # 2. 图像拼接 (与之前完全一致)
            w = max(ann_global.shape[1], ann_local.shape[1])
            h1 = int(ann_global.shape[0] * w / ann_global.shape[1])
            h2 = int(ann_local.shape[0] * w / ann_local.shape[1])
            i1 = cv2.resize(ann_global, (w, h1))
            i2 = cv2.resize(ann_local, (w, h2))
            combined = np.vstack((i1, i2))

            # 3. 状态文本叠加 (原有样式：左下角黑条 + 彩色字)
            cv2.rectangle(combined, (5, combined.shape[0]-45), (450, combined.shape[0]-5), (0,0,0), -1)
            status_color = (0, 165, 255) # 搜索时用橙色
            if "Target Found" in self.current_state_text or "Reached" in self.current_state_text:
                status_color = (0, 255, 0)
            
            cv2.putText(combined, f"Status: {self.current_state_text}", 
                        (15, combined.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

            # 4. 窗口显示
            disp_w, disp_h = int(combined.shape[1] * self.display_scale), int(combined.shape[0] * self.display_scale)
            cv2.imshow("System Monitor (Scaling)", cv2.resize(combined, (disp_w, disp_h)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            rate.sleep()

        cv2.destroyAllWindows()

    def logic_timer(self):
        """盲区检测计时器"""
        rospy.sleep(10.0)
        if not self.global_detected_flag and not self.search_triggered:
            self.search_triggered = True
            from active_searcher import ActiveSearcher
            searcher = ActiveSearcher(target_class="tool_box")
            searcher.run()

    def shutdown_hook(self, signum, frame):
        self.running = False
        rospy.signal_shutdown("User exit")

if __name__ == '__main__':
    nav = RealtimeNavigator()
    nav.run()
