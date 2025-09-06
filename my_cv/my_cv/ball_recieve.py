#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

class BallListenerNode(Node):
    def __init__(self):
        super().__init__('ball_listener')
        # QoS는 퍼블리셔와 같은 depth(10)로 맞춰 줍니다.
        self.sub = self.create_subscription(
            PointStamped,                   # 메시지 타입
            '/basketball/position',         # 토픽 이름
            self.ball_callback,             # 콜백 함수
            10                              # 큐 깊이
        )

    def ball_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        # 원하는 처리: 로그로 출력하거나, 다른 로직으로 연결
        self.get_logger().info(f'Ball at X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m')

def main():
    rclpy.init()
    node = BallListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
