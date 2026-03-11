#!/usr/bin/env python3
"""
采集 YOLO 训练数据帧 - 使用 Intel RealSense D435i（仅 RGB）
按 空格键 保存当前帧
按 ESC 或 'q' 退出
"""

import cv2
import os
import time
import pyrealsense2 as rs
import numpy as np

# ================== 配置区（按需修改）==================
OUTPUT_DIR = "/home/wheeltec/wsd_ws/src/my_robot_system/dataset2/images"  # 图像保存目录
PREFIX = "tool_box"                                          # 文件名前缀
# =====================================================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 图像将保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("📷 正在初始化 RealSense D435i...")
    print("⌨️  操作说明:\n    - 按 [空格] 保存当前帧\n    - 按 [ESC] 或 'q' 退出")

    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用 RGB 流（YOLO 只需彩色图）
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print("❌ 无法启动 RealSense 相机！请确认设备已连接并被识别。")
        print("错误信息:", e)
        return

    saved_count = 0

    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("⚠️ 未获取到彩色帧")
                continue

            # 转为 numpy 数组（BGR 格式，OpenCV 可直接显示）
            frame = np.asanyarray(color_frame.get_data())

            # 显示画面 + 状态
            display = frame.copy()
            cv2.putText(display, f"Saved: {saved_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to capture, ESC/q to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("YOLO Dataset Capture - RealSense D435i", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键保存
                filename = f"{PREFIX}_{int(time.time() * 1000)}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, frame)
                saved_count += 1
                print(f"✅ 已保存: {filename}")

            elif key in [27, ord('q'), ord('Q')]:  # ESC 或 q 退出
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n🎉 采集完成！共保存 {saved_count} 张图像。")


if __name__ == "__main__":
    main()
