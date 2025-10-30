#!/usr/bin/env python3

# 내부 흰색 수정
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rcl_interfaces.msg import SetParametersResult

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
    
        self.roi_x_start = self.image_width * 1 // 5
        self.roi_x_end   = self.image_width * 4 // 5
        self.roi_y_start = self.image_height * 1 // 12
        self.roi_y_end   = self.image_height * 11 // 12

        # 잔디
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = 'AVG: --- ms | FPS: --'
        self.last_position_text = 'Miss'

        # 추적
        self.last_cx = None
        self.last_cy = None
        self.last_w = None
        self.last_h = None
        self.last_z = None
        self.last_score = None
        self.lost = 0
        self.last_box = None

        self.last_band_mask = None

        # 변수
        self.fx, self.fy = 607.0, 606.0
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.last_band_mask = np.zeros((self.image_width, self.image_height), dtype=np.uint8)

        # 깊이
        self.depth_scale = 0.001  # mm -> m
        self.depth_min = 100.0
        self.depth_max = 3000.0

        self.bridge = CvBridge()

        color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.pub_hoop = self.create_publisher(PointStamped, '/hoop/position', 10)

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
   
    def on_click(self, event, x, y, _, __):  # 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end:
            H, S, V = [int(v) for v in self.hsv[y - self.roi_y_start, x - self.roi_x_start]]
            self.get_logger().info(f"[Pos] x={x-self.zandi_x}, y={-(y-self.zandi_y)} | HSV=({H},{S},{V})")

    def make_band_masks(self, src_pts, width, height, t, s, roi_shape):
        H, W = roi_shape[:2]
        top_mask = left_mask = right_mask = np.zeros((H, W), dtype=np.uint8)
        inner_mask = np.zeros((H, W), dtype=np.uint8)

        dst_rect = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
        M_inv = cv2.getPerspectiveTransform(dst_rect, src_pts.astype(np.float32))

        # 박스 평면의 밴드, 내부 duddur
        top_dst   = np.array([[0,0],[width,0],[width,t],[0,t]], dtype=np.float32)
        left_dst  = np.array([[0,0],[s,0],[s,height],[0,height]], dtype=np.float32)
        right_dst = np.array([[width-s,0],[width,0],[width,height],[width-s,height]], dtype=np.float32)
        inner_dst = np.array([[s,t],[width-s,t],[width-s,height],[s,height]], dtype=np.float32)

        top_src = np.round(cv2.perspectiveTransform(top_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
        left_src = np.round(cv2.perspectiveTransform(left_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
        right_src = np.round(cv2.perspectiveTransform(right_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
        inner_src = np.round(cv2.perspectiveTransform(inner_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)

        cv2.fillPoly(top_mask, [top_src], 255, lineType=cv2.LINE_8)
        cv2.fillPoly(left_mask, [left_src], 255, lineType=cv2.LINE_8)
        cv2.fillPoly(top_mask, [right_src], 255, lineType=cv2.LINE_8)
        cv2.fillPoly(right_mask, [inner_src], 255, lineType=cv2.LINE_8)
        return top_mask, left_mask, right_mask, inner_mask


    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)  # mm

        # ROI
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]  # mm
        
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

        best_cnt = best_cx = best_cy = best_w = best_h = best_depth = best_box = None
        best_score = 0.5  # 빨강 비율 최소치

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
          
                width, height = int(round(width)), int(round(height))

                # 밴드 두께
                t = int(round(height * self.band_top_ratio))
                s = int(round(width * self.band_side_ratio))
                top_mask, left_mask, right_mask, inner_mask = self.make_band_masks(src, width, height, t, s, roi_color.shape)

                area_top = cv2.countNonZero(top_mask)
                area_left = cv2.countNonZero(left_mask)
                area_right = cv2.countNonZero(right_mask)
                if area_top == 0 or area_left == 0 or area_right == 0:
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

                    # 깊이 > 워프 없이 ROI 깊이에서 마스크로
                    inner_depth_vals = roi_depth[inner_mask.astype(bool)]
                    valid = np.isfinite(inner_depth_vals) & (inner_depth_vals > self.depth_min) & (inner_depth_vals < self.depth_max)
                    if np.count_nonzero(valid) <= 30:
                        continue
                    depth_med = float(np.median(inner_depth_vals[valid])) * self.depth_scale

                    # 스코어/최적 후보 갱신 (기존과 동일)
                    if best_score < ratio_band:
                        best_score = ratio_band
                        best_cnt = cnt
                        best_cx, best_cy = int(round(cx + self.roi_x_start)), int(round(cy + self.roi_y_start))
                        best_w, best_h = width, height
                        best_depth = depth_med
                        best_box = (box + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)
                        best_band_mask = cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_or(top_mask, left_mask,),right_mask),red_mask)

        t5 = time.time()
                
        # 좌표 갱신
        if best_cnt is not None:   
            self.lost = 0
            self.last_cx, self.last_cy = best_cx, best_cy
            self.last_w, self.last_h = best_w, best_h
            self.last_z = best_depth
            self.last_score = best_score
            self.last_box = best_box if 'best_box' in locals() else None
            self.rect_color = (0, 255, 0)

            self.last_band_mask = best_band_mask

        elif self.lost < 10 and self.last_cx is not None:            
            self.lost += 1
            best_cx, best_cy = self.last_cx, self.last_cy 
            best_w, best_h = self.last_w, self.last_h 
            best_depth = self.last_z 
            best_score = self.last_score
            best_box = self.last_box
            self.rect_color = (255, 0, 0)

        else:
            self.lost = 10
            best_cx = best_cy = best_w = best_h = best_depth = best_box = None
            self.rect_color = (0, 0, 255)
            self.last_position_text = 'Miss'

            self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)

        # 최종 출력
        if best_cx is not None:

            
            X = (best_cx - self.cx_intr) * best_depth / self.fx
            Y = (best_cy - self.cy_intr) * best_depth / self.fy
                    
            msg = PointStamped()
            msg.header = color_msg.header
            msg.point.x, msg.point.y, msg.point.z = X, Y, best_depth
            self.pub_hoop.publish(msg)

            cv2.polylines(frame, [best_box], True, self.rect_color, 2)
            cv2.circle(frame, (best_cx, best_cy), 5, (0, 0, 255), -1)

            self.last_position_text = f'Dist: {best_depth:.2f}m | Pos: {best_cx}, {-best_cy}, | Acc: {best_score:.2f}, | Ang: {yaw_deg}'
                
        # 출력
        cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
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
        cv2.putText(frame, self.last_position_text, (self.roi_x_start - 125, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        
        # self.get_logger().info(f"ROI = {(t1-start_time)*1000:.3f}, Conv = {(t2-t1)*1000:.3f}, InRange = {(t3 - t2)*1000:.3f},"
        #                        f"Mophology = {(t4 - t3)*1000:.3f}, "
        #                        f"Contour = {(t5 - t4)*1000:.3f}, Decision = {(t6 - t5)*1000:.3f}, Show = {(now - t6)*1000:.3f}")

        cv2.imshow('Red Mask', red_mask)
        cv2.imshow('White Mask', white_mask)
        cv2.imshow('Band Mask', self.last_band_mask)
        cv2.imshow('Hoop Detection', frame)
        cv2.waitKey(1)

def main():
    # cv2.setUseOptimized(True)
    # cv2.setNumThreads(0)

    rclpy.init()
    node = HoopDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()