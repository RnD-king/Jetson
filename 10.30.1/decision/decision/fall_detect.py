# ROS 2 Python 인터페이스 사용을 위한 기본 모듈
import rclpy
from rclpy.node import Node

# IMU 센서 메시지 타입 (쿼터니언 포함)
from sensor_msgs.msg import Imu
from geometry_msgs.msg import QuaternionStamped #/filter/quaternion 메세지 타입

from robot_msgs.msg import FallResult

# 쿼터니언 → 오일러각 변환 함수
from tf_transformations import euler_from_quaternion

# 각도 계산에 필요한 수학 함수들
import math

# 시간 측정용 (넘어짐 조건 지속 시간 체크)
import time


class XsensFallDetector(Node):
    def __init__(self):
        super().__init__('xsens_fall_detector')  # 노드 이름 설정

        # '/filter/quaternion' 토픽을 구독하여 필터된 Orientation을 수신
        self.subscription = self.create_subscription(
            QuaternionStamped,
            '/filter/quaternion',
            self.OrientationCallback,
            10)
        
        # '/imu/data'를 이용해 az 추출
        self.subscription_acceleration = self.create_subscription(
            Imu,
            '/imu/data',
            self.AccelerationCallback,
            10)

        self.get_logger().info('✅ Xsens Fall Detector Node Started.')

        #fall, 안정된 yaw 각도 decision으로 publish
        self.fall_publisher = self.create_publisher(FallResult, '/fall_result', 10)


        self.fall_detected_ = False  # 넘어짐 상태 저장 변수 (True일 경우 넘어짐 상태)
        self.condition_start_time_ = None # 넘어짐 조건이 처음 만족된 시점 기록
        self.kFallTime = 1  # 넘어짐 확정을 위한 지속 시간 임계값 (초 단위)
        self.prev_yaw_deg_ = 0.0 # 넘어지기 전 안정적인 yaw 값 저장용
        self.last_stable_time_ = None # 마지막으로 안정 상태에 있었던 시간


        # 최신 센서 값 저장 (초기화 필수)
        self.roll_deg_ = 0.0
        self.pitch_deg_ = 0.0
        self.yaw_deg_ = 0.0
        self.az_ = 0.0

    # 필터된 Orientation을 수신할 때 호출되는 콜백 함수
    def OrientationCallback(self, msg):
        # orientation 필드에서 쿼터니언 추출
        quat = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]

        # 쿼터니언을 오일러 각도로 변환 (라디안 단위)
        roll, pitch, yaw = euler_from_quaternion(quat)

        # 각도(degree) 단위로 변환
        self.roll_deg_ = math.degrees(roll)
        self.pitch_deg_ = math.degrees(pitch)
        self.yaw_deg_ = (math.degrees(yaw) + 360) % 360  # yaw는 0~360도로 wrap

        # 넘어짐 판단 수행 (조건 충족 여부 확인 및 상태 업데이트)
        
        # az 추출 전용 콜백
    def AccelerationCallback(self, msg):
        self.az_ = msg.linear_acceleration.z   # 선형 가속도 추출

        # self.StableYaw(self.roll_deg_, self.pitch_deg_, self.yaw_deg_)
        # 넘어짐 판단 수행 (조건 충족 여부 확인 및 상태 업데이트)
        self.CheckFallCondition(self.roll_deg_, self.pitch_deg_, self.az_, self.yaw_deg_)

        # yaw 값을 이용해 텍스트 방향(N, NE, ...) 계산
        direction = self.GetDirectionFromYaw(self.yaw_deg_)

        # 현재 센서 상태 및 넘어짐 여부 출력
        self.get_logger().info(
            f"Yaw: {self.yaw_deg_:.1f}° → 방향: {direction}, "
            f"Roll: {self.roll_deg_:.1f}°, Pitch: {self.pitch_deg_:.1f}°, "
            f"az: {self.az_:.2f}, 넘어짐: {self.fall_detected_}")

    # yaw 값을 방향 문자열로 변환하는 함수
    def GetDirectionFromYaw(self, yaw_deg):
        if 337.5 <= yaw_deg or yaw_deg < 22.5:
            return '북 (N)'
        elif 22.5 <= yaw_deg < 67.5:
            return '북동 (NE)'
        elif 67.5 <= yaw_deg < 112.5:
            return '동 (E)'
        elif 112.5 <= yaw_deg < 157.5:
            return '남동 (SE)'
        elif 157.5 <= yaw_deg < 202.5:
            return '남 (S)'
        elif 202.5 <= yaw_deg < 247.5:
            return '남서 (SW)'
        elif 247.5 <= yaw_deg < 292.5:
            return '서 (W)'
        elif 292.5 <= yaw_deg < 337.5:
            return '북서 (NW)'
        else:
            return '알 수 없음'
        
     # 넘어짐 판단 조건 확인, 안정된 yaw 저장 및 상태 갱신 함수
    def CheckFallCondition(self, roll_deg, pitch_deg, az, yaw_deg):
        # 조건 1: roll 또는 pitch가 지나치게 크면 기울어진 상태로 판단 / abs는 절댓값

        fall_msg = FallResult()

        tilt = abs(roll_deg) > 60 or abs(pitch_deg) > 65

        # 조건 2: Z축 가속도가 매우 작으면 누운 상태일 수 있음
        z_fallen = abs(az) < 3.0

        # 세 조건이 모두 만족되면 넘어짐 후보로 간주
        if tilt and z_fallen:
            # self.get_logger().warn('💥 넘어짐 확정됨!')
            if self.condition_start_time_ is None:
                # 처음 조건을 만족한 시점 기록
                self.condition_start_time_ = time.time()
            elif time.time() - self.condition_start_time_ > self.kFallTime:
                # 일정 시간 이상 조건이 유지되었을 때만 넘어짐 확정
                if not self.fall_detected_:
                    self.get_logger().warn('💥 넘어짐 확정됨!')
                self.fall_detected_ = True
        else:
            # 조건이 중간에 깨졌을 경우 초기화
            self.condition_start_time_ = None
            self.fall_detected_ = False

        #안정된 yaw 저장
        if abs(roll_deg) < 45 and abs(pitch_deg) < 30: 
            if self.last_stable_time_ is None:
                self.last_stable_time_ = time.time() # 처음으로 안정 상태에 진입한 경우 → 시간 기록
            elif time.time() - self.last_stable_time_ > 0.5: # 안정 상태가 0.5초 이상 지속되었으면 yaw 저장
                self.prev_yaw_deg_ = yaw_deg
                self.get_logger().info(f"안정된 yaw 저장됨: {yaw_deg:.1f}°")
        else:
            self.last_stable_time_ = None


        # fall 메시지에 상태 반영해서 항상 publish
        fall_msg.fall_detect = self.fall_detected_  
        fall_msg.prev_yaw_deg = int(self.prev_yaw_deg_)
        self.fall_publisher.publish(fall_msg)

    # def RecoverPose():
        # 여기에는 다이나믹셀을 원래 yaw로 돌리기 위한 코드

# 노드 실행 진입점
def main(args=None):
    rclpy.init(args=args)
    node = XsensFallDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

