cmake_minimum_required(VERSION 3.18)  # CUDA Modern 방식은 최소 3.18 이상 권장
project(my_cv)

# CUDA 언어 활성화
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_PATH "/usr/local/lib")

# Jetson Orin Nano: Compute Capability 8.7
set(CMAKE_CUDA_ARCHITECTURES 87)

# ROS 2 빌드 시스템
find_package(ament_cmake REQUIRED)

# ROS 2 의존성
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(robot_msgs REQUIRED)

# OpenCV with CUDA
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaimgproc cudaarithm cudawarping)

# Include 디렉토리 설정
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# CUDA 전처리 커널 라이브러리
add_library(cuda_preprocess STATIC
  include/yolo_preprocess.cu
)
set_target_properties(cuda_preprocess PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# YOLO TensorRT 커스텀 플러그인
add_library(yolo_plugin STATIC
  include/yololayer.cu
)
set_target_properties(yolo_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# 메인 실행 노드
add_executable(yolo_cpp
  my_cv/yolo_cpp.cpp
  my_cv/yolo_trt.cpp
)

target_include_directories(yolo_cpp PUBLIC 
  include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# 링크 라이브러리 설정
target_link_libraries(yolo_cpp
  cuda_preprocess
  yolo_plugin
  ${OpenCV_LIBS}
  nvinfer
  nvonnxparser
  cudart
)

# ROS 2 의존성
ament_target_dependencies(yolo_cpp
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  robot_msgs
)

# 실행 파일 설치
install(TARGETS yolo_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Python 실행 스크립트 설치
install(PROGRAMS
  my_cv/ball_detect.py
  my_cv/ball_and_hoop.py
  my_cv/ball_recieve.py
  my_cv/image_saver.py
  my_cv/hurdle_detect.py
  my_cv/hoop_detect.py
  my_cv/yolo_line_subscriber.py
  my_cv/color_mask_test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch 파일 설치
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Lint / Test 설정
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ament 마무리
ament_package()

