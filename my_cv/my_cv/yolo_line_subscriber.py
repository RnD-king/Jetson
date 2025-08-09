#!/usr/bin/env python3
import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque, Counter
from my_cv_msgs.msg import LinePoint, LinePointsArray # type: ignore
from rcl_interfaces.msg import SetParametersResult

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('line_subscriber')
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback,             
            10)                             
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        self.bridge = CvBridge()

        # 파라미터 선언
        self.declare_parameter("limit_x", 10)
        self.declare_parameter("max_len", 30)
        self.declare_parameter("delta_s", 15) # 커브 판단할 때
        self.declare_parameter("vertical", 75)
        self.declare_parameter("horizonal", 15)

        # 파라미터 적용
        self.limit_x = self.get_parameter("limit_x").value
        self.max_len = self.get_parameter("max_len").value
        self.delta_s = self.get_parameter("delta_s").value
        self.vertical = self.get_parameter("vertical").value
        self.horizonal = self.get_parameter("horizonal").value
        
        self.candidates = [] 
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.out_text = "" # 화면 출력
        self.recent_curve = deque(maxlen=self.max_len) # maxlen 프레임 중에서 최빈값
        self.recent_tilt = deque(maxlen=self.max_len)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 시간 디버깅
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

    def parameter_callback(self, params):
        for param in params:
            if param.name == "limit_x":
                if param.value > 0 and param.value < 64:
                    self.limit_x = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "max_len":
                if param.value > 0:
                    self.max_len = param.value
                    self.recent_curve = deque(maxlen=self.max_len)
                    self.recent_tilt = deque(maxlen=self.max_len)
                else:
                    return SetParametersResult(successful=False)
            if param.name == "delta_s":
                if param.value > 0:
                    self.delta_s = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "vertical":
                if param.value > self.horizonal and param.value < 90:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizonal":
                if param.value > 0 and param.value < self.vertical:
                    self.horizonal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)


    def get_angle(self, c1, c2): # 단순 각도 계산
        dx = c2[0]-c1[0]
        dy = c2[1]-c1[1]
        angle = 180 / math.pi * math.atan2(dy, dx) + 90
        if angle > 89:  # 계산 과부하 방지
            angle = 89
        elif angle < -89:
            angle = -89
        return round(angle, 2)

    def line_callback(self, msg: LinePointsArray):  # 좌표 구독
        self.candidates = [(i.cx, i.cy, i.lost) for i in msg.points]
        for idx, (cx, cy, lost) in enumerate(self.candidates):
            self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}')

    def color_image_callback(self, msg): # 이미지 받아오기

        start_time = time.time()  # 시작 시간 기록

        # ROS 이미지 CV 이미지로 받아오기
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        screen_h, screen_w, _ = cv_image.shape

        # ROI 좌표 정의, 컷
        roi_y_start = screen_h * 1 // 5  # 위
        roi_y_end = screen_h // 1        # 아래
        roi_x_start = screen_w * 2 // 5  # 왼 
        roi_x_end = screen_w * 3 // 5    # 오

        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

#---------------------------------------------------------------1차 판단 시작  << 최소자승 직선 근사  << 얘 좀 이상함

        # ───────── 1. 위치 판단 ─────────
        sum_x = 0
        if self.candidates:
            for i in range(len(self.candidates)):
                sum_x += self.candidates[i][0]

            avg_x = float(sum_x / len(self.candidates))
            if avg_x > (roi_x_end + roi_x_start) / 2 + self.limit_x:
                self.out_text = "Out Left"
            elif avg_x < (roi_x_end + roi_x_start) / 2 - self.limit_x:
                self.out_text = "Out Right"
            else:
                self.out_text = "Straight"
        
        # ───────── 2. 방향 판단 ─────────
        if len(self.candidates) >= 3:
            # tip = 가장 위의 점 (yolo_cpp가 아래→위로 보내므로 [-1]이 위)
            tip_x, tip_y, tip_lost = self.candidates[-1]
            cv2.circle(roi, (tip_x - roi_x_start, tip_y - roi_y_start),
                       4, (0, 255, 255) if tip_lost > 0 else (0, 0, 255), -1)

            # tip 제외한 점들 시각화
            for acx, acy, ac_lost in self.candidates[:-1]:
                color = (0, 255, 255) if ac_lost > 0 else (255, 0, 0)
                cv2.circle(roi, (acx - roi_x_start, acy - roi_y_start),
                           3, color, 2 if ac_lost > 0 else -1)

            xs = np.array([c[0] for c in self.candidates[:-1]], dtype=np.float32)
            ys = np.array([c[1] for c in self.candidates[:-1]], dtype=np.float32)

            if float(xs.max() - xs.min()) < 2.0:
                # 거의 수직: 중앙 x로 세워서 선 그리기 + delta는 x기준 편차
                x_center = int(round(float(np.median(xs))))
                x1 = x2 = x_center
                y1, y2 = roi_y_end, roi_y_start
                line_angle = 90.0
                delta = float(x_center - tip_x)
                if delta >  self.delta_s: self.curve_text = "Turn Left"
                elif delta < -self.delta_s: self.curve_text = "Turn Right"
                else: self.curve_text = "Straight"
                self.tilt_text = "Straight"

            else:
                # fitLine은 (N,2) float32 점 배열을 요구
                pts = np.stack([xs, ys], axis=1)  # (N,2)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)

                # 각도/기울기
                m = vy / (vx + 1e-12)
                line_angle = math.degrees(math.atan(m))

                # tip의 '부호 있는 수직거리' → 좌/우 커브 판단 (현장 부호만 1번 확인)
                signed = (float(tip_x) - x0) * vy - (float(tip_y) - y0) * vx
                delta = signed / (math.hypot(vx, vy) + 1e-12)
                if delta >  self.delta_s: self.curve_text = "Turn Left"
                elif delta < -self.delta_s: self.curve_text = "Turn Right"
                else: self.curve_text = "Straight"

                # 선분 엔드포인트 계산 (ROI 경계와 교차)
                b = y0 - m * x0
                if abs(m) > 1:
                    y1, y2 = roi_y_end, roi_y_start
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:
                    x1, x2 = roi_x_end, roi_x_start
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)

                # 틸트 판단
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else:
                    self.tilt_text = "Straight"

            # 시각화: 선
            cv2.line(roi,
                     (int(x1 - roi_x_start), int(y1 - roi_y_start)),
                     (int(x2 - roi_x_start), int(y2 - roi_y_start)),
                     (255, 0, 0), 2)
        
        elif len(self.candidates) == 2:
            # 두 점으로 단순 선
            x1, y1, down_lost = self.candidates[0]
            x2, y2, up_lost   = self.candidates[1]
            cv2.circle(roi, (x1 - roi_x_start, y1 - roi_y_start),
                       3, (0,255,255) if down_lost > 0 else (0,0,255), 2 if down_lost > 0 else -1)
            cv2.circle(roi, (x2 - roi_x_start, y2 - roi_y_start),
                       3, (0,255,255) if up_lost > 0 else (255,0,0), 2 if up_lost > 0 else -1)

            if abs(x1 - x2) < 2:
                line_angle = 90.0
                self.tilt_text = "Straight"
                x = int(round((x1 + x2) * 0.5))
                x1 = x2 = x
                y1, y2 = roi_y_end, roi_y_start
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                line_angle = math.degrees(math.atan(m))
                if abs(m) > 1:
                    y1, y2 = roi_y_end, roi_y_start
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:
                    x1, x2 = roi_x_end, roi_x_start
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else:
                    self.tilt_text = "Straight"
            self.curve_text = "Straight"
            cv2.line(roi,
                     (int(x1 - roi_x_start), int(y1 - roi_y_start)),
                     (int(x2 - roi_x_start), int(y2 - roi_y_start)),
                     (255, 0, 0), 2)

        else:
            self.curve_text = "Miss"
            self.tilt_text  = "Miss"
            self.out_text   = "Miss"
            line_angle = 0.0


        self.recent_curve.append(self.curve_text)
        self.recent_tilt.append(self.tilt_text)
        stable_curve = Counter(self.recent_curve).most_common(1)[0][0] if self.recent_curve else "Miss"  # 최빈값에 맞게 커브 판단  >> 후에 이거로 상태함수 넣어도 됨 / fsm
        stable_tilt = Counter(self.recent_tilt).most_common(1)[0][0] if self.recent_tilt else "Miss"

        #-------------------------------------------------------------------------------------------------------------- 1차 판단 끝

        if stable_curve == "Straight" and stable_tilt == "Straight" and self.out_text == "Straight":
            res = 1 # 직진
        elif  stable_tilt == "Spin Right" or self.out_text == "Out Left":
            res = 2 # 우회전 해라
            angle = line_angle
        elif  stable_tilt == "Spin Left" or self.out_text == "Out Right":
            res = 3 # 좌회전 해라
            angle = line_angle
        elif stable_curve == "Turn Left" or stable_curve == "Turn Right":
            res = 4

        # 여기에 퍼블리시

        #--------------------------------------------------------------------------------------------------------------- 2차 판단 끝

        # 판단 텍스트 출력
        cv2.putText(cv_image, f"Rotate: {line_angle:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Tilt: {stable_tilt}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Curve: {stable_curve}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (roi_x_start - 1, roi_y_start - 1), (roi_x_end + 1, roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        cv2.line(cv_image, (int(screen_w / 2) - self.limit_x, roi_y_start), (int(screen_w / 2) - self.limit_x, roi_y_end), (255, 22, 255), 1)
        cv2.line(cv_image, (int(screen_w / 2) + self.limit_x, roi_y_start), (int(screen_w / 2) + self.limit_x, roi_y_end), (255, 22, 255), 1)
        
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정

        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed

        now = time.time()

        # 1초에 한 번 평균 계산
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (now - self.last_report_time)
            
            # 텍스트 준비
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            
            # 타이머 리셋
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        # 평균 처리 시간 텍스트 영상에 출력
        if hasattr(self, "last_avg_text"):
            cv2.putText(cv_image, self.last_avg_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        #---------------------------------------------------------------------------------------------------- 결과

        cv2.imshow("Subscriber", cv_image)  # 결과
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
