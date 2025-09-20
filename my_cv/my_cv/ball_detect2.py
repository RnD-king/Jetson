#!/usr/bin/env python3
import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import Counter
from robot_msgs.msg import BallResult, HoopResult, MotionEnd # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화용
from rcl_interfaces.msg import SetParametersResult

camera_width = 640
camera_height = 480

roi_x_start = int(camera_width * 0 // 5)
roi_x_end = int(camera_width * 5 // 5)
roi_y_start = int(camera_height * 1 // 12)
roi_y_end = int(camera_height * 11 // 12)

class LineListenerNode(Node): #################################################################### 판단 프레임 수 바꿀 때 yolo_cpp 도 고려해라~~~
    def __init__(self):
        super().__init__('line_subscriber')

        # zandi
        self.zandi_x = int((roi_x_start + roi_x_end) / 2)
        self.zandi_y = int(camera_height - 100)

        # pick
        self.pick_x = self.zandi_x + 15
        self.pick_y = self.zandi_y - 70 
        self.pick_rad = 20

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력

        # 추적
        self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = None
        self.last_agv_cy_ball = 0 # 화면 전환용
        self.ball_lost = 0 # B

        # 변수
        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype = np.uint8)

        self.lower_hsv_ball = np.array([8, 60, 60])
        self.upper_hsv_ball = np.array([60, 255, 255]) # 주황색 기준으로
        
        self.depth_max_ball = 1500.0  # mm 기준 마스크 거리
        self.depth_min = 50.0 
        self.depth_scale = 0.001    # m 변환용 >> B
        
        self.collecting = False     # 수집 중 여부
        self.armed = False               # motion_end 방어~!

        self.pick_attempt = 0
        self.backboard_score_text = "Hoop Now"
        self.picked = False
        self.ball_saw_once = False
        self.ball_never_seen = True
        self.cam2_miss_count = 0

        self.window_id = 0
        self.frame_idx = 0       
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)

        self.cam1_ball_count = 0
        
        self.ball_valid_list = []
        self.ball_cx_list = []
        self.ball_cy_list = []
        self.ball_dis_list = [] # B

        self.bridge = CvBridge()   
                              
        self.cam1_color_sub = Subscriber(self, Image, '/camera/color/image_raw') # CAM1
        self.cam1_depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') 
        self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1)
        self.cam1_sync.registerCallback(self.cam1_image_callback)
        
        self.motion_end_sub = self.create_subscription(  # 모션엔드
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10) 
        
        # 퍼블리셔
        self.ball_result_pub = self.create_publisher(BallResult, '/ball_result', 10)
        
        # 파라미터 선언 B
        self.declare_parameter("cam_mode", 1)

        self.declare_parameter("orange_h_low", 8) 
        self.declare_parameter("orange_h_high", 60)  
        self.declare_parameter("orange_s_low", 80)  
        self.declare_parameter("orange_s_high", 255)  
        self.declare_parameter("orange_v_low", 130)  
        self.declare_parameter("orange_v_high", 255)

        # 파라미터 적용 B
        self.cam_mode = self.get_parameter("cam_mode").value

        self.orange_h_low = self.get_parameter("orange_h_low").value
        self.orange_h_high = self.get_parameter("orange_h_high").value
        self.orange_s_low = self.get_parameter("orange_s_low").value
        self.orange_s_high = self.get_parameter("orange_s_high").value
        self.orange_v_low = self.get_parameter("orange_v_low").value
        self.orange_v_high = self.get_parameter("orange_v_high").value

        self.add_on_set_parameters_callback(self.param_callback)

        self.lower_hsv_ball = np.array([self.orange_h_low, self.orange_s_low, self.orange_v_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv_ball = np.array([self.orange_h_high, self.orange_s_high, self.orange_v_high], dtype=np.uint8)
        self.hsv = None 

        # 창
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.on_click)
        cv2.setMouseCallback('Ball Detection', self.on_click)

    def param_callback(self, params):
        for p in params:
            if p.name == "cam_mode": self.cam_mode = int(p.value)
            elif p.name == "orange_h_low": self.orange_h_low = int(p.value)
            elif p.name == "orange_h_high": self.orange_h_high = int(p.value)
            elif p.name == "orange_s_low": self.orange_s_low = int(p.value)
            elif p.name == "orange_s_high": self.orange_s_high = int(p.value)
            elif p.name == "orange_v_low": self.orange_v_low = int(p.value)
            elif p.name == "orange_v_high": self.orange_v_high = int(p.value)

        # HSV 배열 갱신
        self.lower_hsv_ball = np.array([self.orange_h_low, self.orange_s_low, self.orange_v_low], dtype=np.uint8)
        self.upper_hsv_ball = np.array([self.orange_h_high, self.orange_s_high, self.orange_v_high], dtype=np.uint8)

        return SetParametersResult(successful=True)
        
    def on_click(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if not (roi_x_start <= x <= roi_x_end and roi_y_start <= y <= roi_y_end):
            return

        rx = x - roi_x_start
        ry = y - roi_y_start
        k = 1  # (3,3)

        y0, y1 = max(0, ry - k), min(roi_y_end - roi_y_start, ry + k + 1)
        x0, x1 = max(0, rx - k), min(roi_x_end - roi_x_start, rx + k +1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1,3)

        H, S, V = np.mean(patch, axis=0).astype(int)
        self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={-(y - self.zandi_y)} | HSV=({H},{S},{V})")

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def cam1_image_callback(self, cam1_color_msg: Image, cam1_depth_msg: Image): # yolo_cpp에서 토픽 보내면 실핼
        
        start_time = time.time()
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(cam1_color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(cam1_depth_msg, desired_encoding='passthrough').astype(np.float32)

        roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # HSV 색 조절
        self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball, self.upper_hsv_ball) # 주황색 범위 색만

        # 모폴로지 연산
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

        bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if self.cam_mode == 1: ##################################################################################################3
            if not self.collecting:
                if self.armed:
                    self.collecting = True
                    self.armed = False
                    self.frames_left = self.collecting_frames
                    self.window_id += 1
                    
                    # 초기화
                    self.ball_valid_list.clear()
                    self.ball_cx_list.clear()
                    self.ball_cy_list.clear()
                    self.ball_dis_list.clear()
                    self.frame_idx = 0

                    self.last_cx_ball = self.last_cy_ball = self.last_radius = self.last_z_ball = None
                    self.ball_lost = 0
                    self.cam2_miss_count = 0

                    self.line_start_time = time.time()

                    self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')

            if self.collecting:
                self.frame_idx += 1
                self.get_logger().info(f"step {self.frame_idx}")

                raw_mask[roi_depth >= self.depth_max_ball] = 0  
                raw_mask[roi_depth <= self.depth_min] = 0 

                # 컨투어
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어

                best_cnt_ball = None
                best_ratio_ball = 0.5
                best_cx_ball = best_cy_ball = best_z_ball = 0

                for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                    area = cv2.contourArea(cnt) # 1. 면적 > 1000
                    if area > 200:
                        (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                        circle_area = circle_r * circle_r * math.pi
                        ratio = abs((area / circle_area) - 1)
                        # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                        if ratio < best_ratio_ball:
                            best_cnt_ball = cnt
                            best_ratio_ball = ratio
                            best_cx_ball = int(x + roi_x_start) 
                            best_cy_ball = int(y + roi_y_start)
                            best_radius = int(circle_r)

                # 검출 결과 처리: 이전 위치 유지 로직
                if best_cnt_ball is not None:
                    # 원 탐지를 했다면
                    self.ball_lost = 0
                    
                    x1 = max(best_cx_ball - roi_x_start - 1, 0)
                    x2 = min(best_cx_ball - roi_x_start + 2, roi_x_end - roi_x_start)
                    y1 = max(best_cy_ball - roi_y_start - 1, 0)
                    y2 = min(best_cy_ball - roi_y_start + 2, roi_y_end - roi_y_start)

                    roi_patch = roi_depth[y1:y2, x1:x2]
                    valid = np.isfinite(roi_patch) & (roi_patch > self.depth_min) & (roi_patch < self.depth_max_ball)
                    if np.count_nonzero(valid) >= 1:
                        best_z_ball = float(np.mean(roi_patch[valid])) * self.depth_scale
                        is_ball_valid = True
                        self.get_logger().info(f"Found! {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}, ratio = {best_ratio_ball}")

                        self.last_z_ball = best_z_ball
                    else:
                        best_z_ball = -1 
                        is_ball_valid = False
                        self.get_logger().info(f"Invalid depth! {best_cx_ball}, {best_cy_ball}")
                    
                    # 이전 위치 업데이트
                    self.last_cx_ball = best_cx_ball
                    self.last_cy_ball = best_cy_ball
                    self.last_radius = best_radius
                    
                    self.rect_color = (0, 255, 0)
                    self.draw_color = (255, 0, 0)

                    cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                    cv2.circle(frame, [best_cx_ball, best_cy_ball], 3, (0,0,255), -1)
                    cv2.circle(bgr, [best_cx_ball, best_cy_ball - roi_y_start], best_radius, (255,0,255), 2)
                    cv2.circle(bgr, [best_cx_ball, best_cy_ball - roi_y_start], 2, (255,0,255), -1)


                else:
                    if self.ball_lost < 3 and self.last_cx_ball is not None:
                        self.ball_lost += 1

                        best_cx_ball = self.last_cx_ball
                        best_cy_ball = self.last_cy_ball
                        best_z_ball = self.last_z_ball  
                        best_radius = self.last_radius

                        self.draw_color = (0, 255, 255)
                        self.rect_color = (255, 0, 0)

                        is_ball_valid = True
                        self.get_logger().info(f"Lost! {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}")

                        cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                        cv2.circle(frame, [best_cx_ball, best_cy_ball], 1, (0,0,255), -1)
                        cv2.circle(bgr, [best_cx_ball, best_cy_ball - roi_y_start], best_radius, (255,0,255), 2)
                        cv2.circle(bgr, [best_cx_ball, best_cy_ball - roi_y_start], 2, (255,0,255), -1)           

                    else:
                        self.ball_lost = 3

                        best_cx_ball = best_cy_ball = best_z_ball = None
                        self.last_cx_ball = self.last_cy_ball = self.last_radius = self.last_z_ball = None
                        
                        self.rect_color = (0, 0, 255)

                        is_ball_valid = False
                        self.get_logger().info(f"Miss!")

                self.frames_left -= 1
                self.ball_valid_list.append(is_ball_valid)
                self.ball_cx_list.append(best_cx_ball)
                self.ball_cy_list.append(best_cy_ball)
                self.ball_dis_list.append(best_z_ball)

                if self.frames_left <= 0:
                    result = Counter(self.ball_valid_list).most_common(1)[0][0]

                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                    if result == True:
                        cxs = [a for s, a in zip(self.ball_valid_list, self.ball_cx_list) if s == result and a is not None]
                        cys = [a for s, a in zip(self.ball_valid_list, self.ball_cy_list) if s == result and a is not None]
                        dists = [a for s, a in zip(self.ball_valid_list, self.ball_dis_list) if s == result and a is not None]
                        avg_cx = int(round(np.mean(cxs)))
                        avg_cy = int(round(np.mean(cys)))
                        avg_dis = np.mean(dists)

                        angle = int(round(math.degrees(math.atan2(avg_cx - self.zandi_x, -avg_cy + self.zandi_y))))
                        self.last_agv_cy_ball = avg_cy # 다음 프레임에 판단용

                        if angle > 90:
                            angle -= 180
                        elif angle <= -90:
                            angle += 180
                        
                        if angle >= 8:
                            res = 14 # 우회전
                        elif angle <= -8: # 좌회전
                            res = 13
                        else: # 직진
                            res = 12

                        self.get_logger().info(f"[Ball] Done: CAM1 found ball | {avg_cx}, {avg_cy}, dis: {avg_dis:.2f}, "
                                            f"res= {res}, angle= {angle} "
                                            f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms")
                        self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                        
                        # 퍼블리시
                        msg_out = BallResult()
                        msg_out.res = res
                        #msg_out.angle = abs(angle)
                        self.ball_result_pub.publish(msg_out)

                        self.cam1_ball_count += 1

                    else:
                        if self.last_agv_cy_ball >= 300 and self.cam1_ball_count >= 1: # 그 전까지 공을 보고 있었다면
                            res = 12
                            angle = 0
                            self.cam_mode = 2 # 2번 캠으로 ㄱㄱ
                            self.last_agv_cy_ball = 0

                            self.get_logger().info(f"[Ball] CAM1 Missed, CAM2 will find,,, | frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")
                            
                            self.last_position_text = "Recieving,,,"
                            self.cam1_ball_count = 0

                            self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = None

                            # 퍼블리시
                            msg_out = BallResult()
                            msg_out.res = res
                            #msg_out.angle = abs(angle)
                            self.ball_result_pub.publish(msg_out)

                        elif self.cam1_ball_count == 1:
                            self.get_logger().info(f"[Ball] Wrong thing | frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")

                            self.last_position_text = "Back to line,,,"
                            self.cam1_ball_count = 0
                        
                        else: 
                            self.get_logger().info(f"[Ball] No ball detected | frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")
                            
                            self.last_position_text = "No ball"
                            self.cam1_ball_count = 0

                    # 리셋
                    self.collecting = False
                    self.frames_left = 0
                    self.frame_idx = 0

                    self.last_position_text = ""
            else:
                self.last_position_text = "In move"

            cv2.line(frame, (roi_x_start, 280), (roi_x_end, 280), (250, 122, 122), 1)

        elif self.cam_mode == 2:
            if not self.collecting:
                if self.armed:
                    self.collecting = True
                    self.armed = False
                    self.frames_left = self.collecting_frames
                    self.window_id += 1

                    # 초기화
                    self.ball_valid_list.clear()
                    self.ball_cx_list.clear()
                    self.ball_cy_list.clear()
                    self.ball_dis_list.clear()
                    self.frame_idx = 0
                    self.ball_lost = 0

                    self.last_cx_ball = self.last_cy_ball = self.last_radius = None

                    self.cam1_ball_count = 0

                    self.line_start_time = time.time()
                    
                    self.draw_color = (0, 255, 0)
                    self.rect_color = (0, 255, 0)

                    self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')
            
            if self.collecting:
                
                angle = 0
                self.frame_idx += 1
                self.get_logger().info(f"{self.frame_idx}")
                
                # HSV 색 조절
                self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball, self.upper_hsv_ball) # 주황색 범위 색만
                
                # 모폴로지 연산
                mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어

                best_cnt_ball = None
                best_ratio_ball = 0.3
                best_cx_ball = best_cy_ball = 0

                for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                    area = cv2.contourArea(cnt) # 1. 면적 > 1000
                    if area > 1000:
                        (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                        circle_area = circle_r * circle_r * math.pi
                        ratio = abs((area / circle_area) - 1)
                        # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈1
                        if ratio < best_ratio_ball:
                            best_ratio_ball = ratio
                            best_cnt_ball = cnt
                            best_cx_ball = int(x + roi_x_start)
                            best_cy_ball = int(y + roi_y_start)
                            best_radius = int(circle_r)

                # 검출 결과 처리: 이전 위치 유지 로직
                if best_cnt_ball is not None:
                    self.ball_lost = 0
                    self.last_cx_ball = best_cx_ball
                    self.last_cy_ball = best_cy_ball
                    self.last_radius = best_radius

                    self.rect_color = (0, 255, 0)
                    self.draw_color = (255, 0, 0)
                    is_ball_valid = True

                    self.get_logger().info(f"Found! {best_cx_ball}, {best_cy_ball}")

                    cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)

                else:
                    if self.ball_lost < 3 and self.last_cx_ball is not None:
                        self.ball_lost += 1
                        best_cx_ball = self.last_cx_ball
                        best_cy_ball = self.last_cy_ball
                        best_radius = self.last_radius

                        self.rect_color = (255, 0, 0)
                        self.draw_color = (0, 255, 255)
                        is_ball_valid = True

                        self.get_logger().info(f"Lost! {best_cx_ball}, {best_cy_ball}")

                        cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                    else:
                        self.ball_lost = 3
                        best_cx_ball = best_cy_ball = None

                        self.last_cx_ball = None
                        self.last_cy_ball = None
                        self.last_radius = None

                        self.rect_color = (0, 0, 255)
                        is_ball_valid = False
                        self.get_logger().info(f"Miss!")
                
                self.frames_left -= 1
                self.ball_valid_list.append(is_ball_valid)
                self.ball_cx_list.append(best_cx_ball)
                self.ball_cy_list.append(best_cy_ball)

                if self.frames_left <= 0:
                    result = Counter(self.ball_valid_list).most_common(1)[0][0]

                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                    if result == True: # 공을 봤다면
                        if self.ball_never_seen: # 여긴 그냥 지금껏 공을 본 적 있는지 없는지 저장용
                            self.ball_saw_once = True
                            self.ball_never_seen = False

                        self.cam2_miss_count = 0

                        cxs = [a for s, a in zip(self.ball_valid_list, self.ball_cx_list) if s == result and a is not None]
                        cys = [a for s, a in zip(self.ball_valid_list, self.ball_cy_list) if s == result and a is not None]
                        avg_cx = int(round(np.mean(cxs)))
                        avg_cy = int(round(np.mean(cys)))

                        dx = avg_cx - self.pick_x
                        dy = avg_cy - self.pick_y

                        if self.picked: # 줍기 모션을 했다면 >> 나 방금 주웠는데 눈 앞에 공이 또 있네?
                            if self.pick_attempt >= 3: # 3번 넘게 실패
                                res = 22
                                self.get_logger().info(f"[Ball] Failed to pick,,, I'll give up | Pos : {dx}, {-dy} | "
                                f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms "
                                f"attempt= {self.pick_attempt + 1}")

                                self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                                self.pick_attempt = 0
                                self.backboard_score_text = "Hoop Now"

                                self.cam2_miss_count = 0
                                self.picked = False
                                self.ball_saw_once = False
                                self.ball_never_seen = True
                                self.cam_mode = 1
                                # 그러니까 이 뒤에 당분간 10번 정도는 공을 봐도 무시하게끔 (res = 99) 주는 플래그 넣고,
                                # 코드 전체에 두르자

                            else: # 시도는 3번만
                                res = 9
                                self.pick_attempt += 1
                                self.get_logger().info(f"[Ball] Pick one more time! Pos : {dx}, {-dy} | "
                                f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms "
                                f"attempt= {self.pick_attempt}")

                        else: # 아직 줍지 않았고, 공은 발견함 (평소 루프)

                            # if math.hypot(dx, dy) <= self.pick_rad: # 오케이 조준 완료
                            #     self.cam_mode = CAM1
                            #     self.cam1_mode = HOOP

                            #     self.get_logger().info(f"[Ball] Pick! Pos : {dx}, {-dy} | "
                            #                     f"frames= {len(self.ball_valid_list)}, "
                            #                     f"wall= {process_time*1000:.1f} ms")
                            #     self.get_logger().info(f"[Hoop] Findind hoop,,,")
                            #     res = 9 # pick 모션

                            #     self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                            #     self.backboard_score_text = "Hoop Now"

                            self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"

                            if abs(dx) <= 25 and abs(dy) <= 15: # 오케이 조준 완료  310 360     315 285
                                self.get_logger().info(f"[Ball] Pick! Pos : {dx}, {-dy} | "
                                                f"frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")
                                res = 9 # pick 모션
                                self.picked = True
                                self.pick_attempt += 1

                            # x합격, y 합격 나누고 둘다 불합격일떄만 크기 비교해서 하기
                            elif abs(dx) <= 25 and abs(dy) > 15:
                                if abs(dy) >= 60:
                                    if dy > 0:
                                        res = 4 #back_one

                                    elif dy < 0:
                                        res = 12 #forward_one
                                else:
                                    if dy > 0:
                                        res = 5 #back_half

                                    elif dy < 0:
                                        res = 6 #forward_half

                            elif abs(dx) > 25 and abs(dy) <= 15:
                                if abs(dx) >= 60: #만약 보폭이 다를떄 큰 보폭의 길이
                                    if dx > 0: 
                                        res = 8 #right_half
                                
                                    elif dx < 0:
                                        res = 7 #left_half
                                else:
                                    if dx > 0: 
                                        res = 8 #right_short
                            
                                    elif dx < 0:
                                        res = 7 #left_short
                            
                            elif abs(dx) > 25 and abs(dy) > 15:
                                if abs(dx) >= abs(dy):
                                    if abs(dx) >= 60: #만약 보폭이 다를떄 큰 보폭의 길이
                                        if dx > 0: 
                                            res = 8 #right_half
                                    
                                        elif dx < 0:
                                            res = 7 #left_half
                                    else:
                                        if dx > 0: 
                                            res = 8 #right_short
                                
                                        elif dx < 0:
                                            res = 7 #left_short

                                elif abs(dx) < abs(dy):
                                    if abs(dy) >= 60:
                                        if dy > 0:
                                            res = 4 #back_one

                                        elif dy < 0:
                                            res = 12 #forward_one
                                    else:
                                        if dy > 0:
                                            res = 5 #back_half

                                        elif dy < 0:
                                            res = 6 #forward_half

                            else: # 여기로 빠질 일 없음
                                self.get_logger().info(f"[Ball] CAM2 Found, Relative position: {dx}, {-dy} | "
                                                f"frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")
                                res = 99 

                    else: # 탐지 실패
                        if self.picked: # 방금 줍는 모션을 했는데 내 눈앞에 공이 없다 = 잘 주웠다 (아니면 흘렸거나)
                            self.get_logger().info(f"[Ball] I made it! | frames= {len(self.ball_valid_list)}, "
                                                f"wall= {process_time*1000:.1f} ms")
                            res = 10 # 재그립 모션

                            self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                            self.backboard_score_text = "Hoop Now"

                            self.cam2_miss_count = 0
                            self.pick_attempt = 0
                            self.picked = False
                            self.ball_saw_once = False
                            self.ball_never_seen = True

                            self.cam_mode = 1

                        else: # 공을 탐지도 못 했고, 줍지도 않았다
                            if self.ball_saw_once: # 근데 그 전에 공을 발견한 적이 있긴 함
                                self.cam2_miss_count += 1

                                if self.cam2_miss_count >= 5: # 한 번 봐놓고 공을 5번 연속이나 못 보면 뭐할까?  >>>  뭔가 찾는 모션
                                    self.get_logger().info(f"[Ball] I totally missed,,, | frames= {len(self.ball_valid_list)}, "
                                                        f"wall= {process_time*1000:.1f} ms")
                                    res = 99
                                    self.last_position_text = f""
                                
                                else: # 한번만 다시 봐봐
                                    self.get_logger().info(f"[Ball] Retry,,, | frames= {len(self.ball_valid_list)}, "
                                                        f"wall= {process_time*1000:.1f} ms, miss count= {self.cam2_miss_count}")
                                    res = 99
                                    self.last_position_text = "Miss"
                                    # 오른쪽 움직일 때 회전하느라공 뒤로 사라지는 거 생각하기
                            
                            else: # 공을 탐지도 못 했고, 줍지도 않았고 공을 본 적도 없음 = 방금 막 HOOP모드에서 바뀜 >> 직진만 하면서 찾아보자
                                self.get_logger().info(f"[Ball] Finding,,, | frames= {len(self.ball_valid_list)}, "
                                                        f"wall= {process_time*1000:.1f} ms")
                                res = 12
                                self.last_position_text = f""
            
                    msg_out = BallResult()
                    msg_out.res = res
                    msg_out.angle = angle
                    self.ball_result_pub.publish(msg_out)
                    self.get_logger().info(f"res = {res}")
                    # 리셋
                    self.collecting = False
                    self.frames_left = 0
                    self.frame_idx = 0

                    self.last_position_text = ""

            else:
                self.last_position_text = "In move"

            cv2.rectangle(frame, (roi_x_start+1, roi_y_start+1), (roi_x_end-1, roi_y_end-1), self.rect_color, 1)
            cv2.circle(frame, (self.pick_x, self.pick_y), self.pick_rad, (111,255,111), 2)


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

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, roi_y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
      
        cv2.imshow('Basketball Mask', bgr) 
        cv2.imshow('Detection', frame)
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
