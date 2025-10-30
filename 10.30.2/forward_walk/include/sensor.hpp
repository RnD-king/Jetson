#ifndef SENSOR_HPP
#define SENSOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <atomic>

class Sensor : public rclcpp::Node
{
public:
    Sensor();
    ~Sensor();

    // === ★ 보정된 최종 속도 Getter ===
    float GetVelocityCompX() const { return vel_comp_x_.load(); }
    float GetVelocityCompY() const { return vel_comp_y_.load(); }
    float GetVelocityCompZ() const { return vel_comp_z_.load(); }

    Eigen::Vector3f GetVelocityComp() const {
        return Eigen::Vector3f(vel_comp_x_.load(), vel_comp_y_.load(), vel_comp_z_.load());
    }

private:
    // === Subscriber ===
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;

    // === Publishers ===
    // Angle
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_z_pub_;

    // Raw
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_z_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_z_pub_;

    // Filtered
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_lpf_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_lpf_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gyro_lpf_z_pub_;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_hpf_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_hpf_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr accel_hpf_z_pub_;

    // Velocity
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_int_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_int_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_int_z_pub_;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_hpfint_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_hpfint_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_hpfint_z_pub_;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_comp_x_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_comp_y_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr vel_comp_z_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // === Callback ===
    void IMUsensorCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void SensorPublishTimerCallback();

    // === Publish functions ===
    void Publish_Angle();
    void Publish_Gyro_Origin();
    void Publish_Accel_Origin();
    void Publish_Gyro_LPF();
    void Publish_Accel_HPF();
    void Publish_Velocity_Integral();
    void Publish_Velocity_HPF_Integral();
    void Publish_Velocity_Complementary();

    // === Utility ===
    void Quaternion2RPY();

    // === Filter core functions ===
    float LPF(float x_k, float y_pre, float Ts, float tau_LPF);
    float HPF(float x_k, float x_pre, float y_pre, float Ts, float tau_HPF);
    float HPF_Integral(float x_k, float y_pre, float Ts, float tau_HPF_Integral);
    float Integral(float x_k, float y_pre, float Ts);
    float Complementary(float gyro, float HPF_Int, float alpha);

    // === Member variables ===
    Eigen::Vector3f Accel = Eigen::Vector3f::Zero();
    Eigen::Vector3f Gyro  = Eigen::Vector3f::Zero();
    Eigen::Vector4f quaternion = Eigen::Vector4f::Zero();
    Eigen::Vector3f RPY   = Eigen::Vector3f::Zero();

    // === ★ 보정된 속도의 최신값(스레드 안전) ===
    std::atomic<float> vel_comp_x_{0.0f};
    std::atomic<float> vel_comp_y_{0.0f};
    std::atomic<float> vel_comp_z_{0.0f};

    // Filter states
    float lpf_y_pre_x = 0.0f, lpf_y_pre_y = 0.0f, lpf_y_pre_z = 0.0f;
    float hpf_x_pre_x = 0.0f, hpf_x_pre_y = 0.0f, hpf_x_pre_z = 0.0f;
    float hpf_y_pre_x = 0.0f, hpf_y_pre_y = 0.0f, hpf_y_pre_z = 0.0f;

    float hpf_yi_pre_x = 0.0f, hpf_yi_pre_y = 0.0f, hpf_yi_pre_z = 0.0f;

    // Parameters (초기값 필요 시 수정)
    float Ts = 0.005f;            // 샘플링 시간 (200Hz)
    float tau_LPF = 0.05f;        // LPF 시간상수
    float tau_HPF = 0.05f;        // HPF 시간상수
    float tau_HPF_Integral = 0.05f; // HPF적분 시간상수
    float alpha = 0.98f;          // 보정 가중치
    float L = 1.0f;               // 속도 보정 scale factor
};

#endif // SENSOR_HPP
