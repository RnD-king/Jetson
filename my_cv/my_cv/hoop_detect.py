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

        # 변수
        self.fx, self.fy = 607.0, 606.0
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        
        self.declare_parameter('white_s_max', 0) # 하양
        self.declare_parameter('white_v_min', 255)

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
        self.red_mask = np.empty((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)       # 빨강 마스크 (재사용)
        self.tmp_mask = np.empty((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)       # 임시 마스크 (재사용)

        # 클릭
        cv2.namedWindow('Hoop Detection')
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
            if param.name == "white_s_min":
                if param.value >= 0 and param.value <= 255:
                    self.white_s_min = param.value
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

    def rect_sum(self, ii, x, y, w, h): # 사각형 적분값
        x2, y2 = x + w, y + h
        return int(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])

    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)  # mm

        # ROI
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        
        t1 = time.time()
        
        cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV, dst=self.hsv)

        t2 = time.time()

        # 빨강 마스킹
        cv2.inRange(self.hsv, (self.red_h1_low, self.red_s_min, self.red_v_min), (self.red_h1_high, 255, 255), dst=self.red_mask)
        cv2.inRange(self.hsv, (self.red_h2_low, self.red_s_min, self.red_v_min), (self.red_h2_high, 255, 255), dst=self.tmp_mask)
        cv2.bitwise_or(self.red_mask, self.tmp_mask, dst=self.red_mask)
        
        t3 = time.time()

        # 모폴로지
        cv2.morphologyEx(self.red_mask, cv2.MORPH_OPEN,  self.kernel, dst=self.red_mask)
        cv2.morphologyEx(self.red_mask, cv2.MORPH_CLOSE, self.kernel, dst=self.red_mask)

        t4 = time.time()

        # 컨투어
        contours, _ = cv2.findContours(self.red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = best_cx = best_cy = best_w = best_h = best_depth = best_box = None
        best_score = 0.5  # 빨강 비율 최소치

        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]  # mm

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.backboard_area: # 1. 일정 넓이 이상
                rect = cv2.minAreaRect(cnt)  # ((cx,cy),(W,H),angle in [-90,0))
                (cx, cy), (W, H), angle = rect
                if W < 1 or H < 1: continue
                if W < H:
                    W, H = H, W
                    angle += 90.0

                box = cv2.boxPoints(((cx, cy), (W, H), angle)).astype(np.float32)

                def order_box(pts):
                    s = pts.sum(axis=1)
                    diff = np.diff(pts, axis=1).ravel()
                    tl = pts[np.argmin(s)]
                    br = pts[np.argmax(s)]
                    tr = pts[np.argmin(diff)]
                    bl = pts[np.argmax(diff)]
                    return np.array([tl, tr, br, bl], dtype=np.float32)

                src = order_box(box)
                W_i, H_i = int(round(W)), int(round(H))
                if W_i <= 0 or H_i <= 0: continue
                dst = np.array([[0,0],[W_i,0],[W_i,H_i],[0,H_i]], dtype=np.float32)

                # 원근변환 (HSV/Depth 동일 M)
                M = cv2.getPerspectiveTransform(src, dst)
                hsv_warp = cv2.warpPerspective(self.hsv, M, (W_i, H_i), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                depth_warp = cv2.warpPerspective(roi_depth, M, (W_i, H_i), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

                # 빨강 마스킹 (warp 공간)
                red1 = cv2.inRange(hsv_warp, (self.red_h1_low, self.red_s_min, self.red_v_min), (self.red_h1_high, 255, 255))
                red2 = cv2.inRange(hsv_warp, (self.red_h2_low, self.red_s_min, self.red_v_min), (self.red_h2_high, 255, 255))
                red_warp = cv2.bitwise_or(red1, red2)
                cv2.morphologyEx(red_warp, cv2.MORPH_OPEN,  self.kernel, dst=red_warp)
                cv2.morphologyEx(red_warp, cv2.MORPH_CLOSE, self.kernel, dst=red_warp)

                # 밴드 두께
                t = int(round(H_i * self.band_top_ratio))
                s = int(round(W_i * self.band_side_ratio))
                if t <= 0 or s <= 0 or W_i <= 2*s or H_i <= t: continue

                # 밴드 빨강 비율 (뒤집은 U: top+left+right - 겹침2개)
                area_band = float(t*W_i + s*H_i + s*H_i - 2*s*t)
                if area_band <= 0.0: continue

                red01_warp = (red_warp != 0).astype(np.uint8)
                ii = cv2.integral(red01_warp, sdepth=cv2.CV_32S)

                def isum(ii, x, y, w, h):
                    x2, y2 = x + w, y + h
                    return int(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])

                red_top   = isum(ii, 0, 0, W_i, t)
                red_left  = isum(ii, 0, 0, s, H_i)
                red_right = isum(ii, W_i - s, 0, s, H_i)
                red_tl    = isum(ii, 0, 0, s, t)
                red_tr    = isum(ii, W_i - s, 0, s, t)
                red_U = float(red_top + red_left + red_right - red_tl - red_tr)
                ratio_band = red_U / area_band

                if (ratio_band >= self.red_ratio_min): # 2. 일정 빨강 이상
                    # 내부(하양)
                    inner_y1, inner_y2 = t, H_i
                    inner_x1, inner_x2 = s, W_i - s
                    if inner_y2 <= inner_y1 or inner_x2 <= inner_x1: continue
                    inner_depth = depth_warp[inner_y1:inner_y2, inner_x1:inner_x2]
                    valid = (np.isfinite(inner_depth)) & (inner_depth > self.depth_min) & (inner_depth < self.depth_max)
                    valid_pixels = inner_depth[valid]
                    if valid_pixels.size > 30: # 4. 거리 조건 만족한 픽셀 수
                        depth_med = float(np.median(valid_pixels)) * self.depth_scale # 깊이

                        if best_score < ratio_band: # 최종
                            best_score = ratio_band
                            best_cnt = cnt
                            best_cx, best_cy = int(round(cx + self.roi_x_start)), int(round(cy + self.roi_y_start))
                            best_w, best_h = W_i, H_i
                            best_depth = depth_med
                            best_box = (box + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)
                    # inner_y1, inner_y2 = t, H_i
                    # inner_x1, inner_x2 = s, W_i - s
                    # if inner_y2 <= inner_y1 or inner_x2 <= inner_x1: continue
                    # inner_hsv = hsv_warp[inner_y1:inner_y2, inner_x1:inner_x2]
                    # inner_white = cv2.inRange(inner_hsv, (0, 0, self.white_v_min), (180, self.white_s_max, 255))
                    # area_inner = inner_white.size
                    # if area_inner <= 0: continue
                    # ratio_inner = float(cv2.countNonZero(inner_white)) / float(area_inner)
                    

                    # if ratio_inner > self.white_min_inner: # 3. 일정 하양 이상
                    #     # 깊이(m)
                    #     inner_depth = depth_warp[inner_y1:inner_y2, inner_x1:inner_x2]
                    #     valid = (np.isfinite(inner_depth)) & (inner_depth > self.depth_min) & (inner_depth < self.depth_max)
                    #     valid_pixels = inner_depth[valid]
                    #     if valid_pixels.size > 30: # 4. 거리 조건 만족한 픽셀 수
                    #         depth_med = float(np.median(valid_pixels)) * self.depth_scale # 깊이

                    #         if best_score < ratio_band: # 최종
                    #             best_score = ratio_band
                    #             best_cnt = cnt
                    #             best_cx, best_cy = int(round(cx + self.roi_x_start)), int(round(cy + self.roi_y_start))
                    #             best_w, best_h = W_i, H_i
                    #             best_depth = depth_med
                    #             best_box = (box + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)


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
            best_box = 0.5
            self.rect_color = (0, 0, 255)
            self.last_position_text = 'Miss'

        # 최종 출력
        if best_cx is not None:
            # 좌우 깊이로 yaw 추정
            delta_px = 20
            u_l = max(0, min(self.image_width - 1, best_cx - delta_px))
            u_r = max(0, min(self.image_width - 1, best_cx + delta_px))
            v   = max(0, min(self.image_height - 1, best_cy))

            z_l_mm = depth[v, u_l]
            z_r_mm = depth[v, u_r]

            yaw_deg = None
            if np.isfinite(z_l_mm) and np.isfinite(z_r_mm) and self.depth_min < z_l_mm < self.depth_max and self.depth_min < z_r_mm < self.depth_max:
                Z_l = float(z_l_mm) * self.depth_scale
                Z_r = float(z_r_mm) * self.depth_scale
                X_l = (float(u_l) - self.cx_intr) * Z_l / self.fx
                X_r = (float(u_r) - self.cx_intr) * Z_r / self.fx

                dx = X_r - X_l
                dz = Z_l - Z_r
                if abs(dx) > 1e-6:
                    yaw_rad = math.atan2(dz, dx)  # +면 오른쪽이 더 가깝다(카메라 기준 양의 yaw)
                    yaw_deg = round(math.degrees(yaw_rad), 2)

                    # 시각화(좌우 샘플점)
                    cv2.circle(frame, (u_l, v), 3, (255, 200, 0), -1)
                    cv2.circle(frame, (u_r, v), 3, (255, 200, 0), -1)
            
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

        cv2.imshow('Red Mask', self.red_mask)
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