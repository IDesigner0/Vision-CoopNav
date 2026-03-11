# Vision-CoopNav

A Vision-Based Cooperative Navigation framework for mobile robots in unstructured warehouse environments. The system integrates fixed global cameras with robot-onboard perception via a Finite State Machine (FSM) to address perception discontinuity and occlusion challenges.

## 📦 Installation & Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/IDesigner0/Vision-CoopNav.git
   ```

2. **Build the ROS workspace**
   ```bash
   cd wsd_ws
   catkin_make
   source devel/setup.bash
   ```

3. **Run the cooperative navigator**
   ```bash
   roslaunch my_robot_system yolo_realtime_navigator.launch
   ```

---

## 🔗 Related Resources

* **jie_ware driver package**:
  * [GitHub](https://github.com/6-robot/jie_ware)
  * [Gitee](https://gitee.com/s-robot/jie_ware)
* **Tutorial Video**:
  * [Bilibili Video](https://www.bilibili.com/video/BV1kwzqYyEe7/)

---
