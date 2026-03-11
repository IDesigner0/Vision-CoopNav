#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

rospy.init_node('tp6c_camera')
pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
bridge = CvBridge()

# 使用子码流，降低带宽
rtsp_url = "rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1"

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    rospy.logerr("无法打开 TP6C 视频流！")
    exit(1)

rate = rospy.Rate(5)  # 5 FPS
while not rospy.is_shutdown():
    ret, frame = cap.read()
    if ret:
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
    rate.sleep()

cap.release()
