#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler

# 全局变量声明
tf_buffer = None
tf_listener = None
goal_pub = None

def pixel_to_world(pixel_x, pixel_y, H):
    src = np.array([[pixel_x, pixel_y]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src[None, :, :], H)
    if dst.size >= 2:
        return float(dst.flat[0]), float(dst.flat[1])
    else:
        raise ValueError("透视变换结果无效")

def get_robot_pose():
    global tf_buffer
    try:
        trans = tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(4.0))
        return trans.transform.translation.x, trans.transform.translation.y
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn("TF2 lookup failed: %s" % str(e))
        return None, None

def main():
    global tf_buffer, tf_listener, goal_pub

    rospy.init_node('single_frame_navigator', anonymous=True)

    # 初始化全局变量
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    image_path = rospy.get_param('~image_path', '/home/wheeltec/wsd_ws/src/my_robot_system/maps/calib_image.jpg')
    homography_path = rospy.get_param('~homography_path', '/home/wheeltec/wsd_ws/src/my_robot_system/scripts/homography_real.npy')
    safe_distance = rospy.get_param('~safe_distance', 0.8)

    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        rospy.logerr(f"❌ 无法加载图像: {image_path}")
        return

    # 加载单应矩阵
    try:
        H = np.load(homography_path)
        if H.shape != (3, 3):
            rospy.logerr(f"单应矩阵形状错误: {H.shape}, 应为 (3,3)")
            return
    except Exception as e:
        rospy.logerr(f"加载单应矩阵失败: {e}")
        return

    # 黄色检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if not contours:
        rospy.loginfo("🔍 未检测到黄色玩偶")
        return

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        rospy.logwarn("轮廓面积为零")
        return

    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
    try:
        world_x, world_y = pixel_to_world(cx, cy, H)
    except Exception as e:
        rospy.logerr(f"坐标转换失败: {e}")
        return

    rospy.loginfo(f"🎯 玩偶位置: ({world_x:.2f}, {world_y:.2f})")

    # 在图像上画出检测框和中心点
    cv2.drawContours(img, [largest], -1, (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
    cv2.putText(img, f"({world_x:.2f}, {world_y:.2f})", (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 保存带有标注的图像
    output_image_path = "/home/wheeltec/wsd_ws/src/my_robot_system/maps/detected.jpg"
    cv2.imwrite(output_image_path, img)
    rospy.loginfo(f"已保存检测结果到 {output_image_path}")

    # 获取机器人位置
    robot_x, robot_y = get_robot_pose()
    if robot_x is None:
        rospy.logwarn("使用默认机器人位置 (0, 0)")
        robot_x, robot_y = 0.0, 0.0
    else:
        rospy.loginfo(f"🤖 机器人位置: ({robot_x:.2f}, {robot_y:.2f})")

    # 计算安全目标
    dx = world_x - robot_x
    dy = world_y - robot_y
    dist = math.hypot(dx, dy)
    if dist < safe_distance:
        goal_x, goal_y = world_x, world_y
        rospy.logwarn("⚠️ 目标太近，直接导航至玩偶位置")
    else:
        goal_x = world_x - safe_distance * dx / dist
        goal_y = world_y - safe_distance * dy / dist

    angle = math.atan2(dy, dx)
    quat_list = quaternion_from_euler(0, 0, angle)
    quat = Quaternion(*quat_list)

    # 发布目标
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"
    goal.pose.position.x = goal_x
    goal.pose.position.y = goal_y
    goal.pose.orientation = quat

    goal_pub.publish(goal)
    rospy.loginfo(f"🚀 已发布导航目标: ({goal_x:.2f}, {goal_y:.2f})")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        rospy.logfatal(f"节点崩溃: {e}")
        raise