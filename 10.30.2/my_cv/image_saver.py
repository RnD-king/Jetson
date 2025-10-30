#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob
import threading

class ImageSaverROIKey(Node):
    def __init__(self):
        super().__init__('image_save')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.frame = None
        self.image_count = 0
        self.save_dir = os.path.abspath('saved_images')
        os.makedirs(self.save_dir, exist_ok=True)

        existing_imgs = sorted(glob.glob(os.path.join(self.save_dir, 'img_*.jpg')))
        if existing_imgs:
            last_img = existing_imgs[-1]
            last_num = int(os.path.basename(last_img).split('_')[1].split('.')[0])
            self.image_count = last_num + 1

        self.roi_x = 160
        self.roi_y = 0
        self.roi_w = 320
        self.roi_h = 480

        # 쓰레드로 OpenCV GUI 관리
        self.running = True
        self.gui_thread = threading.Thread(target=self.cv_loop)
        self.gui_thread.start()

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cv_loop(self):
        print("Press SPACE to save image, ESC to quit.")
        cv2.namedWindow("ROI Viewer", cv2.WINDOW_NORMAL)
        while self.running:
            if self.frame is not None:
                roi = self.frame[
                    self.roi_y:self.roi_y + self.roi_h,
                    self.roi_x:self.roi_x + self.roi_w
                ]
                cv2.imshow("ROI Viewer", roi)
                key = cv2.waitKey(10)

                if key == 27:  # ESC
                    self.running = False
                elif key == 32:  # SPACE
                    filename = os.path.join(self.save_dir, f"img_{self.image_count:04}.jpg")
                    cv2.imwrite(filename, roi)
                    print(f"Saved: {filename}")
                    self.image_count += 1
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = ImageSaverROIKey()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()

