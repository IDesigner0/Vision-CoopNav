#!/usr/bin/env python3
"""
采集 YOLO 训练数据帧 - 固定配置版
按 空格键 保存当前帧
按 ESC 或 'q' 退出
"""

import cv2
import os
import time

# ================== 配置区（按需修改）==================
RTSP_URL = "rtsp://admin:w15032887069@192.168.43.15:554/cam/realmonitor?channel=1&subtype=1"
OUTPUT_DIR = "/home/wheeltec/wsd_ws/src/my_robot_system/dataset/images"      # 图像保存目录
PREFIX = "greenbox_pottedplant"                # 文件名前缀（如 teddy_bear_12345.jpg）
# =====================================================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 图像将保存至: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📷 RTSP 源: {RTSP_URL}")
    print(f"⌨️  操作说明:\n    - 按 [空格] 保存当前帧\n    - 按 [ESC] 或 'q' 退出")

    # 打开视频流
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("❌ 无法打开 RTSP 视频流！请检查网络或账号密码。")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降低延迟

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，可能连接中断")
            break

        # 显示画面 + 状态
        display = frame.copy()
        cv2.putText(display, f"Saved: {saved_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE to capture, ESC/q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("YOLO Dataset Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键保存
            filename = f"{PREFIX}_{int(time.time() * 1000)}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"✅ 已保存: {filename}")

        elif key in [27, ord('q'), ord('Q')]:  # ESC 或 q 退出
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 采集完成！共保存 {saved_count} 张图像。")


if __name__ == "__main__":
    main()
