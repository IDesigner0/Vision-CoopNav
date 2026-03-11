#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import threading
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from ultralytics import YOLO

class ThesisMonitor:
    def __init__(self):
        rospy.init_node('thesis_monitor_node', anonymous=True)
        self.bridge = CvBridge()

        # === 1. 配置参数 ===
        # 请确保 RTSP 地址与你之前代码一致
        self.rtsp_url = "rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1"
        self.global_model_path = "/home/wheeltec/wsd_ws/src/my_robot_system/dataset/tool_robot.pt"
        self.local_model_path = "/home/wheeltec/wsd_ws/src/my_robot_system/dataset/tool_box.pt"
        self.target_class = "tool_box"

        # === 2. 加载模型 ===
        self.yolo_global = YOLO(self.global_model_path)
        self.yolo_local = YOLO(self.local_model_path)

        # === 3. 图像与状态变量 ===
        self.global_frame = None
        self.local_frame = None
        self.current_status = "GLOBAL MONITORING"
        self.global_detected = False
        self.search_triggered = False

        # === 4. 订阅与连接 ===
        # 订阅局部相机
        rospy.Subscriber('/camera/color/image_raw', Image, self.local_cb)
        # 订阅搜索状态（由 active_searcher.py 发布）
        rospy.Subscriber('/search_status', String, self.status_cb)
        # 订阅全局状态判定
        rospy.Subscriber('/global/target_detected', Bool, self.global_status_cb)

        # 启动场端 RTSP 抓取线程
        self.cap = cv2.VideoCapture(self.rtsp_url)
        threading.Thread(target=self.grab_rtsp_thread, daemon=True).start()

        # 启动主逻辑监控线程
        threading.Thread(target=self.logic_watchdog, daemon=True).start()

        rospy.loginfo("✅ 论文监视器已就绪，正在拼接画面...")

    def grab_rtsp_thread(self):
        """独立线程抓取场端画面，防止阻塞"""
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if ret:
                self.global_frame = frame
            else:
                self.cap.open(self.rtsp_url) # 自动重连
            rospy.sleep(0.01)

    def local_cb(self, msg):
        try:
            self.local_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def status_cb(self, msg):
        self.current_status = msg.data

    def global_status_cb(self, msg):
        self.global_detected = msg.data

    def annotate_frame(self, frame, model, label_prefix=""):
        """对画面进行 YOLO 标注"""
        if frame is None: return None
        results = model(frame, verbose=False, conf=0.5)
        ann_frame = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                name = r.names[cls]
                conf = float(box.conf[0])
                # 只有是我们关心的类别才画框
                if name == self.target_class or name == "robot":
                    cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ann_frame, f"{label_prefix}{name} {conf:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return ann_frame

    def logic_watchdog(self):
        """逻辑触发：10秒没看到就启动 active_searcher"""
        rospy.sleep(10.0)
        if not self.global_detected and not self.search_triggered:
            rospy.logwarn("�� Global Lost! Starting Active Searcher...")
            self.search_triggered = True
            # 这里的导入放在这，确保搜索器只在需要时初始化
            try:
                from active_searcher import ActiveSearcher
                searcher = ActiveSearcher(target_class=self.target_class)
                searcher.run()
            except Exception as e:
                rospy.logerr(f"Failed to start searcher: {e}")

    def run_display(self):
        rate = rospy.Rate(15) # 15fps 足够论文截图
        while not rospy.is_shutdown():
            # 1. 标注场端图像
            ann_g = self.annotate_frame(self.global_frame, self.yolo_global, "Global: ")
            # 2. 标注局部图像
            ann_l = self.annotate_frame(self.local_frame, self.yolo_local, "Local: ")

            # 如果没有画面则显示黑屏占位
            if ann_g is None: ann_g = np.zeros((480, 640, 3), np.uint8)
            if ann_l is None: ann_l = np.zeros((480, 640, 3), np.uint8)

            # 3. 拼接
            w = 800 # 统一宽度
            h_g = int(ann_g.shape[0] * w / ann_g.shape[1])
            h_l = int(ann_l.shape[0] * w / ann_l.shape[1])
            
            img_g = cv2.resize(ann_g, (w, h_g))
            img_l = cv2.resize(ann_l, (w, h_l))
            combined = np.vstack((img_g, img_l))

            # 4. 绘制符合论文审美的状态条
            cv2.rectangle(combined, (0, combined.shape[0]-50), (combined.shape[1], combined.shape[0]), (0,0,0), -1)
            # 状态颜色：盲区搜索用橙色，正常用绿色
            color = (0, 165, 255) if self.search_triggered else (0, 255, 0)
            cv2.putText(combined, f"SYSTEM STATUS: {self.current_status}", 
                        (20, combined.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 5. 显示
            cv2.imshow("Semantic Search Framework Monitor", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

if __name__ == '__main__':
    monitor = ThesisMonitor()
    monitor.run_display()
