#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraCalibrator:
    def __init__(self):
        self.bridge = CvBridge()
        self.image = None
        self.clicked_points = []
        self.world_points = np.array([
            [1.40688,-1.13688, 0], # calib_4
            [1.45571,0.927283, 0],   # calib_2
            [3.43507,-1.09092, 0],  # calib_1  
            [3.49364,0.889806, 0]  # calib_3
        ], dtype=np.float32)
        
        rospy.Subscriber("/overhead_cam/image_raw", Image, self.image_callback)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        
    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x, y])
            print(f"Clicked pixel: ({x}, {y})")
            if len(self.clicked_points) == 4:
                self.compute_homography()
                
    def compute_homography(self):
        pixel_points = np.array(self.clicked_points, dtype=np.float32)
        H, _ = cv2.findHomography(pixel_points, self.world_points)
        np.save('homography.npy', H)
        print("Homography matrix saved!")
        print("Now you can use it in yellow_doll_detector.py")
        rospy.signal_shutdown("Calibration complete")
        
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.image is not None:
                img = self.image.copy()
                for i, pt in enumerate(self.clicked_points):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    cv2.putText(img, f"P{i+1}", (int(pt[0])+10, int(pt[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Calibration", img)
                cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('camera_calibrator')
    calibrator = CameraCalibrator()
    calibrator.run()
