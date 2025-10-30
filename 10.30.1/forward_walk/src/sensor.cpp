#include "sensor.hpp"
#include <rclcpp/rclcpp.hpp>


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Sensor>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

Sensor::Sensor() : Node("sensor_node")
{
    // === Subscriber ===
    imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", 10, std::bind(&Sensor::IMUsensorCallback, this, std::placeholders::_1));

    // === Publishers ===
    // Angle
    angle_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Angle/x", 10);
    angle_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Angle/y", 10);
    angle_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Angle/z", 10);

    // Raw
    gyro_x_pub_  = this->create_publisher<std_msgs::msg::Float32>("/Gyro/x", 10);
    gyro_y_pub_  = this->create_publisher<std_msgs::msg::Float32>("/Gyro/y", 10);
    gyro_z_pub_  = this->create_publisher<std_msgs::msg::Float32>("/Gyro/z", 10);
    accel_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Accel/x", 10);
    accel_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Accel/y", 10);
    accel_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/Accel/z", 10);

    // Filtered
    gyro_lpf_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Gyro/x", 10);
    gyro_lpf_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Gyro/y", 10);
    gyro_lpf_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Gyro/z", 10);

    accel_hpf_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Accel/x", 10);
    accel_hpf_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Accel/y", 10);
    accel_hpf_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Accel/z", 10);

    // Velocity
    vel_int_x_pub_    = this->create_publisher<std_msgs::msg::Float32>("/Velocity/x", 10);
    vel_int_y_pub_    = this->create_publisher<std_msgs::msg::Float32>("/Velocity/y", 10);
    vel_int_z_pub_    = this->create_publisher<std_msgs::msg::Float32>("/Velocity/z", 10);
    vel_hpfint_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity/x", 10);
    vel_hpfint_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity/y", 10);
    vel_hpfint_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity/z", 10);

    vel_comp_x_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity_Complementary/x", 10);
    vel_comp_y_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity_Complementary/y", 10);
    vel_comp_z_pub_ = this->create_publisher<std_msgs::msg::Float32>("/filtered/Velocity_Complementary/z", 10);

    // 200 Hz 타이머
    publish_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(5),
        std::bind(&Sensor::SensorPublishTimerCallback, this));
}

Sensor::~Sensor() {}

void Sensor::IMUsensorCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    Accel(0) = msg->linear_acceleration.x;
    Accel(1) = msg->linear_acceleration.y;
    Accel(2) = msg->linear_acceleration.z;

    Gyro(0) = msg->angular_velocity.x;
    Gyro(1) = msg->angular_velocity.y;
    Gyro(2) = msg->angular_velocity.z;

    quaternion(0) = msg->orientation.x;
    quaternion(1) = msg->orientation.y;
    quaternion(2) = msg->orientation.z;
    quaternion(3) = msg->orientation.w;
}

void Sensor::Quaternion2RPY()
{
    tf2::Quaternion q(quaternion(0), quaternion(1), quaternion(2), quaternion(3));
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    RPY(0) = static_cast<float>(roll);
    RPY(1) = static_cast<float>(pitch);
    RPY(2) = static_cast<float>(yaw);
}

void Sensor::SensorPublishTimerCallback()
{
    // 필요한 전부 발행
    Publish_Angle();
    Publish_Gyro_Origin();
    Publish_Accel_Origin();
    Publish_Gyro_LPF();
    Publish_Accel_HPF();
    Publish_Velocity_Integral();
    Publish_Velocity_HPF_Integral();
    Publish_Velocity_Complementary();
}

// ===== Publish 구현 =====
void Sensor::Publish_Angle()
{
    Quaternion2RPY();

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = RPY(0);
    my.data = RPY(1);
    mz.data = RPY(2);

    angle_x_pub_->publish(mx);
    angle_y_pub_->publish(my);
    angle_z_pub_->publish(mz);  // Z(Yaw)도 발행 활성화
}

void Sensor::Publish_Gyro_Origin()
{
    std_msgs::msg::Float32 mx, my, mz;
    mx.data = Gyro(0);
    my.data = Gyro(1);
    mz.data = Gyro(2);
    gyro_x_pub_->publish(mx);
    gyro_y_pub_->publish(my);
    gyro_z_pub_->publish(mz);
}

void Sensor::Publish_Accel_Origin()
{
    std_msgs::msg::Float32 mx, my, mz;
    mx.data = Accel(0);
    my.data = Accel(1);
    mz.data = Accel(2);
    accel_x_pub_->publish(mx);
    accel_y_pub_->publish(my);
    accel_z_pub_->publish(mz);
}

void Sensor::Publish_Gyro_LPF()
{
    float gx = LPF(Gyro(0), lpf_y_pre_x, Ts, tau_LPF);
    float gy = LPF(Gyro(1), lpf_y_pre_y, Ts, tau_LPF);
    float gz = LPF(Gyro(2), lpf_y_pre_z, Ts, tau_LPF);

    lpf_y_pre_x = gx;
    lpf_y_pre_y = gy;
    lpf_y_pre_z = gz;

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = gx;
    my.data = gy;
    mz.data = gz;
    gyro_lpf_x_pub_->publish(mx);
    gyro_lpf_y_pub_->publish(my);
    gyro_lpf_z_pub_->publish(mz);
}

void Sensor::Publish_Accel_HPF()
{
    float ax_f = HPF(Accel(0), hpf_x_pre_x, hpf_y_pre_x, Ts, tau_HPF);
    float ay_f = HPF(Accel(1), hpf_x_pre_y, hpf_y_pre_y, Ts, tau_HPF);
    float az_f = HPF(Accel(2), hpf_x_pre_z, hpf_y_pre_z, Ts, tau_HPF);

    hpf_x_pre_x = Accel(0);  hpf_y_pre_x = ax_f;
    hpf_x_pre_y = Accel(1);  hpf_y_pre_y = ay_f;
    hpf_x_pre_z = Accel(2);  hpf_y_pre_z = az_f;

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = ax_f; my.data = ay_f; mz.data = az_f;
    accel_hpf_x_pub_->publish(mx);
    accel_hpf_y_pub_->publish(my);
    accel_hpf_z_pub_->publish(mz);
}

void Sensor::Publish_Velocity_Integral()
{
    float vx = Integral(Accel(0), hpf_yi_pre_x, Ts);
    float vy = Integral(Accel(1), hpf_yi_pre_y, Ts);
    float vz = Integral(Accel(2), hpf_yi_pre_z, Ts);

    hpf_yi_pre_x = vx; hpf_yi_pre_y = vy; hpf_yi_pre_z = vz;

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = vx; my.data = vy; mz.data = vz;
    vel_int_x_pub_->publish(mx);
    vel_int_y_pub_->publish(my);
    vel_int_z_pub_->publish(mz);
}

void Sensor::Publish_Velocity_HPF_Integral()
{
    float vx = HPF_Integral(Accel(0), hpf_y_pre_x, Ts, tau_HPF_Integral);
    float vy = HPF_Integral(Accel(1), hpf_y_pre_y, Ts, tau_HPF_Integral);
    float vz = HPF_Integral(Accel(2), hpf_y_pre_z, Ts, tau_HPF_Integral); // ✔ z축 이전값은 z로 (원본 코드 오탈자 수정)

    hpf_y_pre_x = vx; hpf_y_pre_y = vy; hpf_y_pre_z = vz;

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = vx; my.data = vy; mz.data = vz;
    vel_hpfint_x_pub_->publish(mx);
    vel_hpfint_y_pub_->publish(my);
    vel_hpfint_z_pub_->publish(mz);
}

void Sensor::Publish_Velocity_Complementary()
{
    // 가속도 → 속도(HPF_Integral)
    float vx = HPF_Integral(Accel(0), hpf_y_pre_x, Ts, tau_HPF_Integral);
    float vy = HPF_Integral(Accel(1), hpf_y_pre_y, Ts, tau_HPF_Integral);
    float vz = HPF_Integral(Accel(2), hpf_y_pre_z, Ts, tau_HPF_Integral);

    hpf_y_pre_x = vx; hpf_y_pre_y = vy; hpf_y_pre_z = vz;

    // 자이로 기반 속도 성분 보정 (축 매핑은 로봇 좌표 정의에 맞게 조정 가능)
    float cx = Complementary(Gyro(1), vx, alpha);
    float cy = Complementary(Gyro(2), vy, alpha);
    float cz = Complementary(Gyro(0), vz, alpha);


    // ★ 최신값 저장 (다른 cpp에서 getter로 읽음)
    vel_comp_x_.store(cx);
    vel_comp_y_.store(cy);
    vel_comp_z_.store(cz);

    std_msgs::msg::Float32 mx, my, mz;
    mx.data = cx; my.data = cy; mz.data = cz;
    vel_comp_x_pub_->publish(mx);
    vel_comp_y_pub_->publish(my);
    vel_comp_z_pub_->publish(mz);

}

// ===== 필터 코어 =====
float Sensor::LPF(float x_k, float y_pre, float Ts, float tau_LPF) {
    return (tau_LPF * y_pre + Ts * x_k) / (Ts + tau_LPF);
}
float Sensor::HPF(float x_k, float x_pre, float y_pre, float Ts, float tau_HPF) {
    return (tau_HPF / (tau_HPF + Ts) * y_pre) + (tau_HPF / (tau_HPF + Ts)) * (x_k - x_pre);
}
float Sensor::HPF_Integral(float x_k, float y_pre, float Ts, float tau_HPF_Integral) {
    return y_pre * (1.0f - tau_HPF_Integral * Ts) + (x_k * Ts);
}
float Sensor::Integral(float x_k, float y_pre, float Ts) {
    return y_pre + x_k * Ts;
}
float Sensor::Complementary(float gyro, float HPF_Int, float alpha) {
    return alpha * gyro * L + (1.0f - alpha) * HPF_Int;
}
