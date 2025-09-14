#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rcl_interfaces.msg import SetParametersResult
from collections import deque, Counter
from robot_msgs.msg import HoopResult, MotionEnd

import rclpy
import cv2
import numpy as np
import time
import math

class HoopDetectorNode(Node):
    def __init__(self):
        super().__init__('hoop_detector')

        # 공용 변수
        self.image_width  = 640
        self.image_height = 480
    
        self.roi_x_start = self.image_width * 0 // 5
        self.roi_x_end   = self.image_width * 5 // 5
        self.roi_y_start = self.image_height * 3 // 12
        self.roi_y_end   = self.image_height * 9 // 12

        # 잔디
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = 'AVG: --- ms | FPS: --'
        self.last_position_text = 'Miss'
        self.score_text = 'Miss'

        # 추적
        self.lost = 0

        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
        self.last_cx = self.last_cy = None
        self.last_depth = None
        self.last_box = None
        self.last_band_mask = None

        self.last_yaw = None

        # 변수
        self.fx, self.fy = 607.0, 606.0
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)


        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        
        self.collecting = False     # 수집 중 여부

        self.armed = False               # motion_end 방어~!
        self.window_id = 0
        self.frame_idx = 0 

        self.hoop_count = 0

        self.valid_list = []
        self.cx_list = []
        self.dis_list = []
        self.yaw_list = []

        # 깊이
        self.depth_scale = 0.001  # mm -> m
        self.depth_min = 100.0
        self.depth_max = 2000.0

        self.bridge = CvBridge()

        self.motion_end_sub = self.create_subscription(  # 모션 끝나면 받아올 T/F
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10)  

        color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.hoop_result_pub = self.create_publisher(HoopResult, '/hoop_result', 10)

        # 파라미터 선언 
        self.declare_parameter('red_h1_low', 0) # 빨강
        self.declare_parameter('red_h1_high', 10)
        self.declare_parameter('red_h2_low', 160)
        self.declare_parameter('red_h2_high', 180)
        self.declare_parameter('red_s_min', 80)
        self.declare_parameter('red_v_min', 60)
        
        self.declare_parameter('white_s_max', 155) # 하양 70. 190
        self.declare_parameter('white_v_min', 90)

        self.declare_parameter('band_top_ratio', 0.15)   # 백보드 h x 0.15
        self.declare_parameter('band_side_ratio', 0.10)   # w x 0.10

        self.declare_parameter('red_ratio_min', 0.55)     # 백보드 영역
        self.declare_parameter('white_min_inner', 0.50)  
        self.declare_parameter('backboard_area', 1500)

        # 파라미터 적용
        self.red_h1_low = self.get_parameter('red_h1_low').value
        self.red_h1_high = self.get_parameter('red_h1_high').value
        self.red_h2_low = self.get_parameter('red_h2_low').value
        self.red_h2_high = self.get_parameter('red_h2_high').value
        self.red_s_min = self.get_parameter('red_s_min').value
        self.red_v_min = self.get_parameter('red_v_min').value

        self.white_s_max = self.get_parameter('white_s_max').value
        self.white_v_min = self.get_parameter('white_v_min').value

        self.band_top_ratio = self.get_parameter('band_top_ratio').value
        self.band_side_ratio = self.get_parameter('band_side_ratio').value

        self.red_ratio_min = self.get_parameter('red_ratio_min').value
        self.white_min_inner = self.get_parameter('white_min_inner').value
        self.backboard_area = self.get_parameter('backboard_area').value

        self.add_on_set_parameters_callback(self.param_callback)

        self.hsv = np.empty((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start, 3), dtype=np.uint8)

        # 클릭
        cv2.namedWindow('Hoop Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Hoop Detection', self.on_click)

    def param_callback(self, params): # 파라미터 변경 적용
        for param in params:
            if param.name == "red_h1_low":
                if param.value >= 0 and param.value <= self.red_h1_high:
                    self.red_h1_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h1_high":
                if param.value >= self.red_h1_low and param.value <= 179:
                    self.red_h1_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h2_low":
                if param.value >= 0 and param.value <= self.red_h2_high:
                    self.red_h2_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h2_high":
                if param.value >= self.red_h2_low and param.value <= 179:
                    self.red_h2_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_s_min":
                if param.value >= 0 and param.value <= 255:
                    self.red_s_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_v_min":
                if param.value >= 0 and param.value <= 255:
                    self.red_v_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_s_max":
                if param.value >= 0 and param.value <= 255:
                    self.white_s_max = param.value
                    self.get_logger().info(f"Changed")
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_v_min":
                if param.value >= 0 and param.value <= 255:
                    self.white_v_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "band_top_ratio":
                if param.value > 0 and param.value < 1:
                    self.band_top_ratio = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "band_side_ratio":
                if param.value > 0 and param.value < 1:
                    self.band_side_ratio = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_ratio_min":
                if param.value > 0 and param.value < 1:
                    self.red_ratio_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_min_inner":
                if param.value > 0 and param.value < 1:
                    self.white_min_inner = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "backboard_area":
                if param.value > 0 and param.value <= 5000:
                    self.backboard_area = param.value
                else:
                    return SetParametersResult(successful=False)
            
        return SetParametersResult(successful=True)
   
    def on_click(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if not (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            return

        rx = x - self.roi_x_start
        ry = y - self.roi_y_start
        k = 1  # (3,3)

        y0, y1 = max(0, ry-k), min(self.roi_y_end - self.roi_y_start, ry+k+1)
        x0, x1 = max(0, rx-k), min(self.roi_x_end - self.roi_x_start, rx+k+1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1,3)

        H,S,V = np.mean(patch, axis=0).astype(int)
        self.get_logger().info(f"[Pos] x={x-self.zandi_x}, y={-(y-self.zandi_y)} | HSV=({H},{S},{V})")

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)  # mm

        # ROI
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]  # mm

        if not self.collecting:
            if self.armed:
                self.collecting = True
                self.armed = False
                self.frames_left = self.collecting_frames
                self.window_id += 1
                
                # 초기화
                self.valid_list.clear()
                self.cx_list.clear()
                self.dis_list.clear()
                self.yaw_list.clear()
                self.frame_idx = 0

                self.lost = 0

                self.line_start_time = time.time()

                self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames')
        
        if self.collecting:
            self.frame_idx += 1
            self.get_logger().info(f"step {self.frame_idx}")
        
            t1 = time.time()
            
            self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

            t2 = time.time()

            # 빨강 마스킹
            red_mask1 = cv2.inRange(self.hsv, (self.red_h1_low, self.red_s_min, self.red_v_min), (self.red_h1_high, 255, 255))
            red_mask2 = cv2.inRange(self.hsv, (self.red_h2_low, self.red_s_min, self.red_v_min), (self.red_h2_high, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_mask[roi_depth >= self.depth_max] = 0  
            red_mask[roi_depth <= self.depth_min] = 0 

            white_mask = cv2.inRange(self.hsv, (0, 0, self.white_v_min), (180, self.white_s_max, 255))
            
            t3 = time.time()

            # 모폴로지
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  self.kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)

            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  self.kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, self.kernel)

            t4 = time.time()

            # 컨투어
            contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 출력용 카피

            best_score = best_top_score = best_left_score = best_right_score = 0.5
            best_cnt = best_cx = best_depth = best_yaw = None
            best_box = None
            best_band_mask = best_left_mask = best_right_mask = best_left_src = best_right_src = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.backboard_area: # 1. 일정 넓이 이상
                    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(W,H),angle >> (-90 ~ 0))
                    (cx, cy), (width, height), angle = rect
                    if width < height:
                        width, height = height, width
                        angle += 90.0

                    box = cv2.boxPoints(((cx, cy), (width, height), angle)).astype(np.float32)
                    
                    s= box.sum(axis=1)
                    diff = np.diff(box, axis=1).ravel()
                    tl = box[np.argmin(s)]
                    br = box[np.argmax(s)]
                    tr = box[np.argmin(diff)]
                    bl = box[np.argmax(diff)]
                    src = np.array([tl, tr, br, bl], dtype=np.float32)

                    # 밴드 두께
                    t = max(1, int(round(height * self.band_top_ratio)))
                    s = max(1, int(round(width * self.band_side_ratio)))
    
                    top_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)
                    left_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)
                    right_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)
                    inner_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)
                    band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)

                    dst_rect = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
                    M_inv = cv2.getPerspectiveTransform(dst_rect, src.astype(np.float32))

                    # 박스 평면의 밴드, 내부 
                    top_dst = np.array([[0,0],[width,0],[width,t],[0,t]], dtype=np.float32)
                    left_dst = np.array([[0,0],[s,0],[s,height],[0,height]], dtype=np.float32)
                    right_dst = np.array([[width-s,0],[width,0],[width,height],[width-s,height]], dtype=np.float32)
                    inner_dst = np.array([[s,t],[width-s,t],[width-s,height],[s,height]], dtype=np.float32)

                    top_src = np.round(cv2.perspectiveTransform(top_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                    left_src = np.round(cv2.perspectiveTransform(left_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                    right_src = np.round(cv2.perspectiveTransform(right_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                    inner_src = np.round(cv2.perspectiveTransform(inner_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)

                    cv2.fillPoly(top_mask, [top_src], 255, lineType=cv2.LINE_8)
                    cv2.fillPoly(left_mask, [left_src], 255, lineType=cv2.LINE_8)
                    cv2.fillPoly(right_mask, [right_src], 255, lineType=cv2.LINE_8)
                    cv2.fillPoly(inner_mask, [inner_src], 255, lineType=cv2.LINE_8)
                    cv2.fillPoly(band_mask, [top_src, left_src, right_src], 255, lineType=cv2.LINE_8)

                    area_top = cv2.countNonZero(top_mask)
                    area_left = cv2.countNonZero(left_mask)
                    area_right = cv2.countNonZero(right_mask)
                    if area_top < 5 or area_left < 5 or area_right < 5:
                        continue

                    # 빨강 비율
                    ratio_top = cv2.countNonZero(cv2.bitwise_and(red_mask, top_mask)) / float(area_top)
                    ratio_left = cv2.countNonZero(cv2.bitwise_and(red_mask, left_mask)) / float(area_left)
                    ratio_right = cv2.countNonZero(cv2.bitwise_and(red_mask, right_mask)) / float(area_right)
                    ratio_band = (ratio_top + ratio_left + ratio_right) / 3

                    if ratio_top >= self.red_ratio_min and ratio_left >= self.red_ratio_min and ratio_right >= self.red_ratio_min:
                        # 내부 하양 비율
                        area_inner = cv2.countNonZero(inner_mask)
                        if area_inner == 0:
                            continue
                        white_hits = cv2.countNonZero(cv2.bitwise_and(white_mask, inner_mask))
                        ratio_inner = white_hits / float(area_inner)
                        if ratio_inner <= self.white_min_inner:
                            continue

                        # 깊이 > 
                        inner_depth_vals = roi_depth[inner_mask.astype(bool)]
                        valid = np.isfinite(inner_depth_vals) & (inner_depth_vals > self.depth_min) & (inner_depth_vals < self.depth_max)
                        if np.count_nonzero(valid) <= 10:
                            continue
                        depth_mean = float(np.mean(inner_depth_vals[valid])) * self.depth_scale

                        # 스코어
                        if best_score < ratio_band:
                            best_score, best_top_score, best_left_score, best_right_score = ratio_band, ratio_top, ratio_left, ratio_right
                            best_cnt = cnt
                            best_cx, best_cy = int(round(cx + self.roi_x_start)), int(round(cy + self.roi_y_start))
                            best_depth = depth_mean
                            best_box = (box + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)
                            best_band_mask = band_mask
                            best_left_mask = left_mask
                            best_right_mask = right_mask
                            best_left_src = left_src
                            best_right_src = right_src
            t5 = time.time()
                    
            # 좌표 갱신
            if best_cnt is not None:   
                self.lost = 0
                self.rect_color = (0, 255, 0)

                self.last_score, self.last_top_score, self.last_left_score, self.last_right_score = best_score, best_top_score, best_left_score, best_right_score
                self.last_cx, self.last_cy = best_cx, best_cy
                self.last_depth = best_depth
                self.last_box = best_box
                self.last_band_mask = best_band_mask

                left_vals = roi_depth[best_left_mask.astype(bool)]
                right_vals = roi_depth[best_right_mask.astype(bool)]

                left_valid = np.isfinite(left_vals) & (left_vals > self.depth_min) & (left_vals < self.depth_max)
                right_valid = np.isfinite(right_vals) & (right_vals > self.depth_min) & (right_vals < self.depth_max)

                nL = int(np.count_nonzero(left_valid))
                nR = int(np.count_nonzero(right_valid))

                if nL >= 10 and nR >= 10:
                    depth_left = float(np.mean(left_vals[left_valid])) * self.depth_scale
                    depth_right = float(np.mean(right_vals[right_valid]))* self.depth_scale

                    x_left_px_roi = float(best_left_src[:, 0].mean())
                    x_right_px_roi = float(best_right_src[:, 0].mean())
                    x_left_px_full = x_left_px_roi + self.roi_x_start
                    x_right_px_full = x_right_px_roi + self.roi_x_start

                    x_left_m = (x_left_px_full  - self.cx_intr) * depth_left / self.fx
                    x_right_m = (x_right_px_full - self.cx_intr) * depth_right / self.fx

                    dx = x_right_m - x_left_m
                    dz = depth_left - depth_right  # 오른쪽이 더 가깝 +

                    if abs(dx) > 0.001:  
                        best_yaw = round(math.degrees(math.atan2(dz, dx)), 2)
                        valid_info = True   
                        self.last_yaw = best_yaw             

                        self.get_logger().info(f"[HOOP] ZL={depth_left:.3f}m, ZR={depth_right:.3f}m, yaw={best_yaw}, nL={nL}, nR={nR}")
                    else:
                        best_yaw = 99
                        valid_info = False
                        self.get_logger().info(f"[HOOP] Retry : not enough big box")
                else:
                    self.get_logger().info(f"[HOOP] Retry : not enough valid depth")
                    best_yaw = 99
                    valid_info = False

                cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (0, 0, 255), -1)

                self.last_position_text = f'Dist: {best_depth:.2f}m | Pos: {best_cx}, {-best_cy}, | Acc: {best_score:.2f}, | Ang: {best_yaw}'
                self.score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score:.2f} | Right: {best_right_score:.2f}, | Ang: {best_yaw}'

            elif self.lost < 3 and self.last_cx is not None:            
                self.lost += 1
                self.rect_color = (255, 0, 0)
                valid_info = True

                best_score, best_top_score, best_left_score, best_right_score = self.last_score, self.last_top_score, self.last_left_score, self.last_right_score
                best_cx, best_cy = self.last_cx, self.last_cy
                best_depth = self.last_depth 
                best_box = self.last_box
                best_band_mask = self.last_band_mask
                best_yaw = self.last_yaw

                cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (0, 0, 255), -1)

                self.last_position_text = f'Dist: {best_depth:.2f}m | Pos: {best_cx}, {-best_cy}, | Acc: {best_score:.2f}, | Ang: {best_yaw}'
                self.score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score} | Right: {best_right_score:.2f}, | Ang: {best_yaw}'

            else:
                self.lost = 3
                self.rect_color = (0, 0, 255)
                valid_info = False

                best_score = best_top_score = best_left_score = best_right_score = None
                best_cx = best_depth = best_box = None
                best_band_mask = best_yaw = None
                
                self.last_position_text = f'Miss'
                self.score_text = f"Miss"
                self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)

            self.frames_left -= 1
            self.valid_list.append(valid_info)
            self.cx_list.append(best_cx)
            self.dis_list.append(best_depth)
            self.yaw_list.append(best_yaw)

            if self.frames_left <= 0:
                result = Counter(self.valid_list).most_common(1)[0][0]

                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                if result == True: # 탐지가 더 많음
                    self.hoop_count += 1
                    cxs = [a for s, a in zip(self.valid_list, self.cx_list) if s == result]
                    dists = [a for s, a in zip(self.valid_list, self.dis_list) if s == result]
                    yaws = [a for s, a in zip(self.valid_list, self.yaw_list) if s == result]

                    avg_cx = int(round(np.mean(cxs)))
                    avg_dis = round(np.mean(dists),2)
                    avg_yaw = round(np.mean(yaws),2)

                    if 0.7 < avg_dis: # 거리가 0.7m 이상 > 위치 정렬

                        angle = int(round(math.degrees(math.atan((avg_cx - self.cx_intr) / self.fx))))

                        if angle > 90:
                            angle -= 180
                        
                        if angle >= 8:
                            res = 14
                        elif angle <= -8:
                            res = 13
                        else:
                            res = 12

                        self.get_logger().info(f"[Hoop] Done: Approaching | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, line angle= {angle}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.valid_list)}, wall= {process_time*1000:.1f} ms")
                    elif 0.5 <= avg_dis: # 거리가 0.7m 이내 > 미세 조정
                        if avg_yaw > 30:
                            angle = int(round(avg_yaw))
                            res = 82 # 오ㅗ른쪽으로 꺾고 왼쪽으로 이동
                        elif avg_yaw < -30:
                            angle = int(round(avg_yaw))
                            res = 83 # 왼쪽으로 꺾고 오른쪽으로 이동
                        else:
                            angle = 0
                            if avg_cx - self.zandi_x > 30:
                                res = 72 # 왼쪽 이동
                            elif avg_cx - self.zandi_x < -30:
                                res = 73 # 오른쪽 이동
                            else:
                                res = 12

                        self.get_logger().info(f"[Hoop] Done: Near by | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.valid_list)}, wall= {process_time*1000:.1f} ms")
                    
                    else: # 거리가 0.5m 이내 > 
                        if avg_yaw > 30:
                            angle = int(round(avg_yaw))
                            res = 82 # 오ㅗ른쪽으로 꺾고 왼쪽으로 이동
                        elif avg_yaw < -30:
                            angle = int(round(avg_yaw))
                            res = 83 # 왼쪽으로 꺾고 오른쪽으로 이동
                        else:
                            angle = 0
                            if avg_cx - self.zandi_x > 50:
                                res = 74 # 왼쪽 반 이동
                            elif avg_cx - self.zandi_x < -50:
                                res = 75 # 오른쪽 반 이동
                            else:
                                res = 77 # 던져 던져

                        if res == 77:
                            self.get_logger().info(f"[Hoop] Shoot !! | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.valid_list)}, wall= {process_time*1000:.1f} ms")    
                        else: 
                            self.get_logger().info(f"[Hoop] Done: Near by | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.valid_list)}, wall= {process_time*1000:.1f} ms")

                else:
                    if self.hoop_count >= 1: # 그 전까지 공을 보고 있었다면
                        res = 55 # 다시 찾는 모션 >> 제자리에서? 한발짝 뒤에서?
                        angle = 0

                        self.get_logger().info(f"[Hoop] Missed,,, | frames= {len(self.valid_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.hoop_count = 0
                        self.last_position_text = "No hoop"

                    else: 
                        self.get_logger().info(f"[Hoop] No hoop detected | frames= {len(self.valid_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.last_position_text = "No hoop"
                        self.hoop_count = 0

                        res = 99 # 라인 걸으세요
                        angle = 0

                # 퍼블리시
                msg_out = HoopResult()
                msg_out.res = res
                msg_out.angle = abs(angle)
                self.hoop_result_pub.publish(msg_out)

                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0
                    
        # 출력
        cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
        cv2.rectangle(frame, (int((self.roi_x_start + self.roi_x_end) / 2) - 50, self.roi_y_start), (int((self.roi_x_start + self.roi_x_end) / 2) + 50, self.roi_y_end), 
                      (255, 120, 150), 1)
        cv2.circle(frame, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
                
        t6 = time.time()        

        # 시간
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time  += elapsed
        now = time.time()

        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / max(1, self.frame_count)
            fps = self.frame_count / max(1e-6, (now - self.last_report_time))
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.score_text, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        
        # self.get_logger().info(f"ROI = {(t1-start_time)*1000:.3f}, Conv = {(t2-t1)*1000:.3f}, InRange = {(t3 - t2)*1000:.3f},"
        #                        f"Mophology = {(t4 - t3)*1000:.3f}, "
        #                        f"Contour = {(t5 - t4)*1000:.3f}, Decision = {(t6 - t5)*1000:.3f}, Show = {(now - t6)*1000:.3f}")

        if self.collecting:
            cv2.imshow('Red Mask', red_mask)
            cv2.imshow('White Mask', white_mask)
        cv2.imshow('Band Mask', self.last_band_mask)
        cv2.imshow('Hoop Detection', frame)
        cv2.waitKey(1)

def main():

    rclpy.init()
    node = HoopDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
