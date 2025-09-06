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
from robot_msgs.msg import LinePoint, LinePointsArray, LineResult, MotionEnd # type: ignore
from rcl_interfaces.msg import SetParametersResult

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('line_subscriber')
        
        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 0 // 5)  # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 5 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)

        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        
        self.collecting = False     # 수집 중 여부
        self.sixth_frame = False

        self.status_list = [] # 누적 값 저장
        self.angle_list = []

        self.bridge = CvBridge()
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback, 10)      
        # self.motion_end_sub = self.create_subscription(  # 모션 끝나면 받아올 T/F
        #     MotionEnd,                   
        #     '/motion_end',                   
        #     self.motion_callback, 10)                         
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        # 파라미터 선언
        self.declare_parameter("delta_zandi_min", 100) # 핑크 라인 두께
        self.declare_parameter("collecting_frames", 9) # 몇 프레임 동안 보고 판단할 거냐  << yolo는 15프레임, 정지 5프레임, 탐지는 10프레임만
        self.declare_parameter("delta_tip_min", 15) # 빨간 점이 얼마나 벗어나야 커브일까
        self.declare_parameter("vertical", 15) # 직진 판단 각도
        self.declare_parameter("horizontal", 75) # 수평 판단 각도  <<<  아직 안 만듬

        # 파라미터 적용
        self.delta_zandi_min = self.get_parameter("delta_zandi_min").value
        self.collecting_frames = self.get_parameter("collecting_frames").value
        self.delta_tip_min = self.get_parameter("delta_tip_min").value
        self.vertical = self.get_parameter("vertical").value
        self.horizontal = self.get_parameter("horizontal").value
        
        self.candidates = [] 
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.out_text = "" # 화면 출력

        self.delta_out = 40
        self.out_now = False
        self.out_angle = 0

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 퍼블리셔
        self.line_result_pub = self.create_publisher(LineResult, '/line_result', 10)

    def parameter_callback(self, params):
        for param in params:
            if param.name == "delta_zandi_min":
                if param.value > 0 and param.value < 320:
                    self.delta_zandi_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "collecting_frames":
                if param.value > 0:
                    self.collecting_frames = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "delta_tip_min":
                if param.value > 0:
                    self.delta_tip_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "vertical":
                if 0 < param.value and param.value < self.horizontal:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizontal":
                if param.value < 90 and param.value > self.vertical:
                    self.horizontal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def line_callback(self, msg: LinePointsArray):  # 좌표 구독
        self.candidates = [(i.cx, i.cy, i.lost) for i in msg.points]
        for idx, (cx, cy, lost) in enumerate(self.candidates):
           self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}') # 잘 받아오고 있나 확인용
        if not self.collecting:
            self.collecting = True  # 탐지 시작해라
            self.frames_left = self.collecting_frames  # 프레임 초기화
            
            self.out_text = ""
            self.curve_text = ""
            self.tilt_text = ""

            self.status_list.clear()
            self.angle_list.clear()

            self.line_start_time = time.time()
            
            self.get_logger().info(f'[Line] Start collecting {self.collecting_frames} frames')

    def color_image_callback(self, msg): # 이미지 받아오기
        
        line_angle = 0.0 # 직선 각도 (맨 위의 점 제외)
        angle = 0.0 # 잔디가 회전해야할 각도
        status = 0
        start_time = time.time()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        roi = cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        if self.collecting: # 모션이 끝났다고 신호 받으면 시작
            
            self.get_logger().info(f"{9 - self.frames_left + 1}")
            # 1. 위치 판단
            if len(self.candidates) >= 3:  #========================================================================================= I. 세 점 이상 탐지
                # 정렬)
                tip_x, tip_y, tip_lost = self.candidates[-1] # 맨위
                cv2.circle(roi, (tip_x - self.roi_x_start, tip_y - self.roi_y_start), 4, (0,255,255) if tip_lost > 0 else (0,0,255), -1)

                for acx, acy, ac_lost in self.candidates[:-1]: # 나머지 밑에
                    color = (0, 255, 255) if ac_lost > 0 else (255, 0, 0)
                    cv2.circle(roi, (acx - self.roi_x_start, acy - self.roi_y_start), 3, color, 2 if ac_lost > 0 else -1)

                line_x = np.array([c[0] for c in self.candidates[:-1]],dtype = np.float32)
                line_y = np.array([c[1] for c in self.candidates[:-1]],dtype = np.float32)
                avg_line_x = int(round(np.mean(line_x)))  # 1. out

                if float(line_x.max() - line_x.min()) < 2.0:  # 일직선인 경우
                    x1 = x2 = avg_line_x
                    y1, y2 = self.roi_y_end, self.roi_y_start # x1, x2, y1, y2는 모두 직선 시각화
                    line_angle = 0  # 2. tilt
                    delta_tip = float(avg_line_x - tip_x)  # 3. curve
                    delta_zandi = float(avg_line_x - self.zandi_x)

                else: # 일반적인 경우
                    pts = np.stack([line_x, line_y], axis = 1)  # (N,2)
                    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
                    # 각도/기울기
                    m = vy / vx
                    line_angle = math.degrees(math.atan2(vx, -vy))  #(vx, vy) 순서 주의
                    if line_angle >= 90:
                        line_angle -= 180 # 범위(-90, 90)
                    # tip 좌표로부터 커브 판단
                    signed_tip = (tip_x - x0) * vy - (tip_y - y0) * vx
                    signed_zandi = (self.zandi_x - x0) * vy - (self.zandi_y - y0) * vx 
                    delta_tip = signed_tip / (math.hypot(vx, vy) + 1e-5)  # 3. curve
                    delta_zandi = signed_zandi / (math.hypot(vx, vy) + 1e-5) # 점이 왼쪽에 있으면 양수 (둘 다)
                    delta_zandi = delta_zandi * abs(line_angle) / line_angle
                    #self.get_logger().info(f'{line_angle}')

                    b = y0 - m * x0
                    if abs(m) > 1:
                        y1, y2 = self.roi_y_end, self.roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:
                        x1, x2 = self.roi_x_end, self.roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)
                #-------------------------------------------------------------------------------------------------------#
                if self.out_now == True: 
                    if abs(delta_zandi) <= self.delta_out:
                        if abs(line_angle) < self.vertical:  # 분기 3. tilt  =  Straight
                            angle = 0
                            self.tilt_text = "Straight"
                            status = 1 # straight
                        elif self.horizontal > line_angle > self.vertical:
                            angle = line_angle
                            status = 3 # 우회전
                            self.tilt_text = "Spin Right"
                        elif -self.horizontal < line_angle < -self.vertical:
                            angle = line_angle
                            status = 2 # 좌회전
                            self.tilt_text = "Spin Left"
                    else: 
                        angle = 0
                        status = 12
                        self.out_text = "Still Out"
                        
                elif self.out_now == False:
                    if abs(delta_tip) > self.delta_tip_min:  # 분기 1-1. turn  =  RL 
                        angle = math.degrees(math.atan2(tip_x - self.zandi_x, -(tip_y - self.zandi_y)))  # 커브길이면 끝점만 보고 따라가게
                        # 범위를 (-90, 90]로 정규화 (수직에서 좌/우로 기운 각도)
                        if angle >= 90:
                            angle -= 180
            
                        if abs(angle) <= self.vertical:
                            status = 1 # straight
                        elif angle > self.vertical: 
                            status = 3 # 우회전 
                        elif angle < -self.vertical: 
                            status = 2 # 좌회전 << 실제 회전값

                        if delta_tip * line_angle > 0:
                            self.curve_text = "Turn Left"  # <<<< 트랙 구조상 좌회전
                        else:
                            self.curve_text = "Turn Right"

                        if abs(delta_zandi) <= self.delta_zandi_min:  # 분기 2-1. out  =  IN
                            self.out_text = "In"

                            if abs(line_angle) < self.vertical:  # 분기 3-1. tilt = Straight
                                self.tilt_text = "Straight"
                            elif self.horizontal > line_angle > self.vertical:  # 분기 3-2. tilt = Left
                                self.tilt_text = "Spin Right"
                            elif -self.horizontal < line_angle < -self.vertical:  # 분기 3-3. tilt = Right
                                self.tilt_text = "Spin Left"
                        else:  # 분기 2-2. out  =  OUT
                            if delta_zandi > 0:
                                self.out_text = "Out Left"
                            else:
                                self.out_text = "Out Right"
                            
                            if abs(line_angle) < self.vertical: 
                                self.tilt_text = "Straight"
                            elif self.horizontal > line_angle > self.vertical:
                                self.tilt_text = "Spin Right"
                            elif -self.horizontal < line_angle < -self.vertical:
                                self.tilt_text = "Spin Left"

                    else:  # 분기 1-2. turn  =  Straight
                        self.curve_text = "Straight"
                        if abs(delta_zandi) < self.delta_zandi_min:  # 분기 2-1. out  =  IN
                            self.out_text = "In"
                            if abs(line_angle) < self.vertical:  # 분기 3. tilt  =  Straight
                                angle = 0
                                self.tilt_text = "Straight"
                                status = 1 # straight
                            elif self.horizontal > line_angle > self.vertical:
                                angle = line_angle
                                status = 3 # 우회전
                                self.tilt_text = "Spin Right"
                            elif -self.horizontal < line_angle < -self.vertical:
                                angle = line_angle
                                status = 2 # 좌회전
                                self.tilt_text = "Spin Left"
                            # else: << 수평선
                        else:  # 분기 2-2. out  =  OUT
                            if delta_zandi > 0:
                                self.out_text = "Out Left"
                            else:
                                self.out_text = "Out Right"
                            
                            r = float(np.clip(delta_zandi / 320.0, -1.0, 1.0))
                            angle = math.degrees(math.asin(r)) + line_angle # 
                            self.get_logger().info(f"{angle},{line_angle},{delta_zandi}")
                            if abs(line_angle) < self.vertical:  # 분기 3. tilt  =  Straight
                                self.tilt_text = "Straight"
                            elif self.horizontal > line_angle > self.vertical:
                                self.tilt_text = "Spin Right"
                            elif -self.horizontal < line_angle < -self.vertical:
                                self.tilt_text = "Spin Left"

                            if abs(angle) < self.vertical:
                                status = 12 # straight
                            elif self.horizontal > angle > self.vertical:
                                status = 14 # 우회전
                            elif -self.horizontal < angle < -self.vertical:
                                status = 13 # 좌회전

                # 시각화
                cv2.line(roi, (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)), (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)), (255, 0, 0), 2)

            elif len(self.candidates) == 2:  #=============================================================================================== II. 두 점 탐지
                # 정렬
                self.curve_text = "Straight" # 커브길 판단 X
                down_x, down_y, down_lost = self.candidates[0] # 아래
                up_x, up_y, up_lost = self.candidates[1] # 위
                cv2.circle(roi, (down_x - self.roi_x_start, down_y - self.roi_y_start), 3, (0, 255, 255) if down_lost > 0 else (255, 0, 0), 2 if down_lost > 0 else -1)
                cv2.circle(roi, (up_x - self.roi_x_start, up_y - self.roi_y_start), 3, (0, 255, 255) if up_lost > 0 else (0, 0, 255), 2 if up_lost > 0 else -1)

                avg_x = int(round((down_x + up_x) / 2)) # 1. out

                if abs(down_x - up_x) < 2:  # 일직선인 경우
                    line_angle = 0 # 2. tilt
                    x1 = x2 = avg_x
                    y1, y2 = self.roi_y_end, self.roi_y_start
                    delta_zandi = abs(float(avg_x - self.zandi_x))
                else:  # 일반적인 경우
                    m = (up_y - down_y) / (up_x - down_x)
                    b = down_y - m * down_x
                    line_angle = math.degrees(math.atan2(up_x - down_x, up_y - down_y))  # <- 핵심 변경! (vx, vy) 순서 주의
                    # 범위를 (-90, 90]로 정규화 (수직에서 좌/우로 기운 각도)
                    if line_angle > 90:
                        line_angle -= 180
                    signed_zandi = (self.zandi_x - down_x) * (up_y - down_y) - (self.zandi_y - down_y) * (up_x - down_x) 
                    delta_zandi = abs(signed_zandi / (math.hypot(up_x - down_x, up_y - down_y) + 1e-5)) # 점이 왼쪽에 있으면 양수 (둘 다)
                    if abs(m) > 1:
                        y1, y2 = self.roi_y_end, self.roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:
                        x1, x2 = self.roi_x_end, self.roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)
                #----------------------------------------------------------------------------------------------------------#
                if delta_zandi < self.delta_zandi_min:  # 분기 1-1. out  =  In
                    self.out_text = "In"
                    if abs(line_angle) <= self.vertical:  # 분기 2-1. tilt  =  Straight
                        angle = 0
                        self.tilt_text = "Straight"
                        status = 1 # straight
                    elif self.horizontal > line_angle > self.vertical:  # 분기 2-2. tilt  =  Left
                        angle = line_angle
                        status = 3 # 우회전
                        self.tilt_text = "Spin Right"
                    elif -self.horizontal < line_angle < -self.vertical:  # 분기 2-3. tilt  =  Right
                        angle = line_angle
                        status = 2 # 좌회전
                        self.tilt_text = "Spin Left"
                    # else: << 수평선

                else:  # 분기 1-2. out  =  RL ######################################################## 나중에 고치기
                    if avg_x-self.zandi_x > 0:
                        self.out_text = "Out Left"
                    else:
                        self.out_text = "Out Right"
                    r = float(np.clip(delta_zandi / 320.0, -1.0, 1.0))
                    angle = math.degrees(math.asin(r)) # 

                    if abs(line_angle) <= self.vertical:  # 분기 3. tilt  =  Straight
                        self.tilt_text = "Straight"
                    elif self.horizontal > line_angle > self.vertical:
                        self.tilt_text = "Spin Right"
                    elif -self.horizontal < line_angle < -self.vertical:
                        self.tilt_text = "Spin Left"
                    angle += line_angle
                    
                    if abs(angle) <= self.vertical:
                        status = 12 # straight
                    elif self.horizontal > angle > self.vertical:
                        status = 14 # 우회전
                    elif -self.horizontal < angle < -self.vertical:
                        status = 13 # 좌회전

                cv2.line(roi, (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)), (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)), (255, 0, 0), 2)

            else:  #================================================================================================================== 3. 점 1개 or 탐지 실패
                self.curve_text = "Miss"
                self.tilt_text = "Miss"
                self.out_text = "Miss"
                angle = 0
                line_angle = 0
                delta_zandi = 0
                status = 99 # retry
            #=======================================================================================================================================================

            angle = int(round(angle))
            self.frames_left -= 1
            self.status_list.append(status) # 결과 저장
            self.angle_list.append(angle)

            if self.frames_left <= 0: # 10 프레임 지났으면
                cnt=Counter(self.status_list)
                res=max(cnt.items(), key=lambda kv:kv[1])[0]  # 가장 많이 나온 대표 status 구하고
                angles=[a for s,a in zip(self.status_list,self.angle_list) if s==res]
                mean_angle=int(round((np.mean(angles))))  # 그 대표 status에서의 각도들의 평균

                if self.out_now == True: # out 상태였는데 발 밑에 점 발견 했으면 원래대로
                    if res != 12:
                        self.out_now = False
                else:
                    if res in(12, 13, 14):
                        self.out_now = True

                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                if res != 99:
                    self.get_logger().info(f"[Line] Done: res= {res}, angle_avg= {mean_angle}, line_angle= {line_angle}, delta= {delta_zandi}, "
                                           f"frames= {len(self.status_list)}, wall= {process_time*1000:.1f} ms")
                else:
                    self.get_logger().info(f"[Line] Window done: res= {res}, angle_avg= Miss, frames= {len(self.status_list)}, "
                                           f"wall= {process_time*1000:.1f} ms")

                # 퍼블리시
                msg_out = LineResult()
                msg_out.res = res
                msg_out.angle = abs(mean_angle)
                self.line_result_pub.publish(msg_out)

                self.collecting = False # 초기화 ## #yolo_cpp랑 토픽 순서때매 여기 한번 더 되는드쇼``
                self.frames_left = 0
                
            
            #------------------------------------------------------------------------------------------------------------  출력

            cv2.putText(cv_image, f"Rotate: {angle}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Tilt: {self.tilt_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Curve: {self.curve_text}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        else: # 모션 중,,,
            cv2.putText(cv_image, "In move", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            

        cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (self.roi_x_start - 1, self.roi_y_start - 1), (self.roi_x_end + 1, self.roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        # cv2.line(cv_image, (self.zandi_x - self.delta_zandi_min, self.roi_y_start), (self.zandi_x - self.delta_zandi_min, self.roi_y_end), (255, 22, 255), 1)
        # cv2.line(cv_image, (self.zandi_x + self.delta_zandi_min, self.roi_y_start), (self.zandi_x + self.delta_zandi_min, self.roi_y_end), (255, 22, 255), 1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), self.delta_zandi_min, (255, 22, 255), 1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), self.delta_out, (255, 22, 22), 1)
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정

        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed

        if time.time() - self.last_report_time >= 1.0:  # 1초마다 평균 계산
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (time.time() - self.last_report_time)
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            
            self.frame_count = 0  # 초기화
            self.total_time = 0.0
            self.last_report_time = time.time()

        cv2.putText(cv_image, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Line", cv_image)  # 이미지
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()