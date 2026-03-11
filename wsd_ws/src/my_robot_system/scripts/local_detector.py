#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class LocalDetector:
    def __init__(self, target_class="tool_box"):
        self.target_class = target_class
        self.bridge = CvBridge()
        self.model = YOLO("/home/wheeltec/wsd_ws/src/my_robot_system/dataset/tool_box.pt")
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.detected = False

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_img, verbose=False, conf=0.5)
            self.detected = any(self.model.names[int(box.cls)] == self.target_class for r in results for box in r.boxes)
        except:
            pass

    def is_target_detected(self):
        return self.detected
