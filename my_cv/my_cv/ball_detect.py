#!/usr/bin/env python3
# 쿠다 가속 넣고
# 영상 전처리 더 생각
# 욜로 버전도 한번 해보자

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from geometry_msgs.msg import PointStamped  # type: ignore
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화

class BasketballDetectorNode(Node):
    def __init__(self):
        super().__init__('basketball_detector')
        # 디버깅용
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = 'AVG: --- ms | FPS: --'

        # 변수
        self.lower_hsv = np.array([8, 60, 0])
        self.upper_hsv = np.array([60, 255, 255]) # 주황색 기준으로
        self.depth_thresh = 1500.0  # mm 기준
        self.depth_scale = 0.001    # m 변환용
        self.fx, self.fy = 615.0, 615.0
        self.cx_intr, self.cy_intr = 320.0, 240.0
        self.lost = 0
        # 이전 위치 저장
        self.last_cx_img = None
        self.last_cy_img = None
        self.last_radius = None
        self.last_z = None

        self.bridge = CvBridge()
        # 칼라, 깊이 영상 동기화
        color_sub = Subscriber(self, Image, '/camera/color/image_raw') # 칼라
        depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw') # 깊이
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.pub_ball = self.create_publisher(PointStamped, '/basketball/position', 10) # 나중에 다른 노드에 전송하기 위한 퍼블리셔

    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()
        last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력
        ball_color = (0, 255, 0)
        rect_color = (0, 255, 0)

        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        h, w = frame.shape[:2]

        # ROI
        roi_y_start = h * 1 // 3  # 위
        roi_y_end = h // 1        # 아래
        roi_x_start = w * 2 // 5  # 왼 
        roi_x_end = w * 3 // 5    # 오
        roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # 원점 지정
        origin_x = int((roi_x_start + roi_x_end) / 2)
        origin_y = h - 30

        # HSV 색 조절
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        raw_mask[roi_depth >= self.depth_thresh] = 0  # 주황색 범위에 드는 색만 추출한 마스크

        mask = raw_mask.copy()

        # 모르포지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 침식 - 팽창
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 팽창 - 침식

        # 컨투어
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어
        best_cnt = None
        best_area = 0
        best_ratio = 1
        x_roi = y_roi = radius = None
        for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
            area = cv2.contourArea(cnt) # 1. 면적 > 1000
            if area > 1000:
                (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                circle_area = circle_r * circle_r * 3.1416
                ratio = abs((area / circle_area) - 1)
                # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                if ratio < best_ratio and ratio < 0.3:
                    best_ratio = ratio
                    best_area = area
                    best_cnt = cnt
                    x_roi = x
                    y_roi = y
                    radius = circle_r

        # 검출 결과 처리: 이전 위치 유지 로직
        if best_cnt is not None:
            # 원 탐지를 했다면
            self.lost = 0
            cx_img = int(x_roi) + roi_x_start
            cy_img = int(y_roi) + roi_y_start  # 중심 좌표
            z = float(roi_depth[int(y_roi), int(x_roi)]) * self.depth_scale # 거리 (깊이)
            # 이전 위치 업데이트
            self.last_cx_img = cx_img
            self.last_cy_img = cy_img
            self.last_radius = radius
            self.last_z = z
        elif self.lost < 10 and self.last_cx_img is not None:
            # 최근 10프레임 내에는 이전 위치 유지
            self.lost += 1
            cx_img = self.last_cx_img
            cy_img = self.last_cy_img
            radius = self.last_radius
            z = self.last_z
            ball_color = (255, 0, 0)
        else:
            # Miss 상태
            self.lost += 1
            last_position_text = 'Miss'
            # 표시할 위치 없음
            cx_img = cy_img = radius = z = None
            rect_color = (0, 0, 255)

        # 위치 퍼블리시 및 화면 표시
        if cx_img is not None:
            # 카메라까지 거리 보정 값
            X = (cx_img - self.cx_intr) * z / self.fx
            Y = (cy_img - self.cy_intr) * z / self.fy
            msg = PointStamped() # 메세지 만들고
            msg.header = color_msg.header # 칼라 화면 프레임 정보
            msg.point.x, msg.point.y, msg.point.z = X, Y, z # 공 위치 좌표 3개
            self.pub_ball.publish(msg) # 보내기

            # 원 그리기
            cv2.circle(frame, (cx_img, cy_img), int(radius), ball_color, 2)
            cv2.circle(frame, (cx_img, cy_img), 5, (0, 0, 255), -1)

            # 원점과의 거리 정보
            dx = cx_img - origin_x
            dy = cy_img - origin_y
            last_position_text = f'Dist: {z:.2f}m | Pos: {dx}, {-dy}'

        # ROI랑 속도 표시
        cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), rect_color, 1)
        cv2.circle(frame, (origin_x, origin_y), 5, (255, 255, 255), -1)
        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, last_position_text, (roi_x_start, roi_y_start - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 80, 215), 2)

        # 후처리 된 마스킹 영상
        final_mask = np.zeros_like(mask)
        if best_cnt is not None and best_area > 1000:
            cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        
        cv2.imshow('Raw Basketball Mask', raw_mask) # 기준 거리 이내, 주황색
        cv2.imshow('Mask', mask) # 기준 거리 이내, 주황색, 보정 들어간 마스크
        cv2.imshow('Final Mask', final_mask) # 최종적으로 공이라 판단한 마스크
        cv2.imshow('Basketball Detection', frame)
        cv2.waitKey(1)

        # 딜레이 측정
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            fps = self.frame_count / (now - self.last_report_time)
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now


def main():
    rclpy.init()
    node = BasketballDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
