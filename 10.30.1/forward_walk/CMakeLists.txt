

cmake_minimum_required(VERSION 3.5)
project(forward_walk)

# 이 줄을 추가하여 ROS 2 패키지 기능을 사용
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(dynamixel_sdk REQUIRED)  # 다이나믹셀 SDK 의존성
find_package(Eigen3 REQUIRED)         # Eigen3 의존성
find_package(std_msgs REQUIRED)  # std_msgs 패키지 의존성 추가
find_package(tf2 REQUIRED)



# 메시지 타입과 서비스가 필요하면 이 줄을 추가
find_package(rosidl_default_generators REQUIRED)



find_package(robot_msgs REQUIRED)

# 타겟에 대한 포함 디렉토리
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)


# 빌드 타겟 설정
add_executable(main_node src/main.cpp src/callback.cpp src/dynamixel_controller.cpp src/dynamixel.cpp src/NewPattern2.cpp src/BRP_Kinematics.cpp)

# 실행 파일에 필요한 ROS 2 라이브러리 링크
ament_target_dependencies(main_node
  rclcpp
  std_msgs
  sensor_msgs
  dynamixel_sdk
  Eigen3
  tf2  # tf2 의존성 추가
  robot_msgs
)

# ===== 새 센서 노드 =====
add_executable(sensor_node
  src/sensor.cpp
)
target_include_directories(sensor_node PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(sensor_node
  rclcpp
  std_msgs
  sensor_msgs
  tf2
  Eigen3
)

# 설치 설정
install(TARGETS main_node
  sensor_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
