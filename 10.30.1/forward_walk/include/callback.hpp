#ifndef CALLBACK_H
#define CALLBACK_H



#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/bool.hpp"
// #include <tf2/LinearMath/Quaternion.h>
#include <boost/thread.hpp>
#include <Eigen/Dense>  // Eigen 관련 헤더 추가
#include "sensor_msgs/msg/imu.hpp" //
#include "robot_msgs/msg/line_result.hpp"
#include <atomic>

#include "dynamixel.hpp"
#include "BRP_Kinematics.hpp"
#include "NewPattern2.hpp"

using Eigen::VectorXd;  // Eigen 벡터를 간단히 사용하기 위한 선언

class Callback : public rclcpp::Node
{
private:
    // double turn_angle = 0;     
    // int arm_indext = 0;        
    double z_c = 1.2 * 0.28224; 
    double g = 9.81;           
    double omega_w;            
    double _dt = 0.01;         



    double kp_zmp_ = 0.05;


    Trajectory *trajectoryPtr;    // Trajectory 객체를 가리키는 포인터
    IK_Function *IK_Ptr;          // IK_Function 객체를 가리키는 포인터
    Dxl *dxlPtr;                  // Dxl 객체를 가리키는 포인터
    Pick *pick_Ptr;               // Pick 객체를 가리키는 포인터
    double Goal_joint_[NUMBER_OF_DYNAMIXELS];

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_; //


    // rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr Direction_subsciption_;
    
    using LineResult = robot_msgs::msg::LineResult;

    rclcpp::Subscription<LineResult>::SharedPtr line_subscriber_;
    

    std::atomic<bool> line_turn{false};
    std::atomic<int> turns_remaining_{0};

public:
    Callback(Trajectory *trajectoryPtr, IK_Function *IK_Ptr, Dxl *dxlPtr, Pick *pick_Ptr);
    

    void SetLineTurn(bool on);

    // ROS 메시지 콜백 함수
    // rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr set_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_subscriber_;

    // virtual void SetMode(const std_msgs::msg::Bool::SharedPtr set);
    virtual void StartMode(const std_msgs::msg::Bool::SharedPtr start);

    // void DirectionCallback(const sensor_msgs::msg::Imu::SharedPtr Direction_msg);

    // void CalculateTurn(int a);

    int GetIndexT() const { return indext; }
    int GetAngle() const { return angle111; } // CalculateTurn에서 계산한 각도를 뺴오기 위한 getter 함수

    double startRL_st[6] = { 0.0 };
    double startLL_st[6] = { 0.0 };

    int GetTurnsRemaining() const;
    const void* GetTurnsRemainingAddr() const;

    void OnLineResult(int angle);
    
    int  FetchSubTurnsRemaining(int delta = 1);  // 감소 전 값을 반환
    void SetTurnsRemaining(int v);

    // 모션 제어 함수들
    virtual void SelectMotion(int go); 
    virtual void Write_All_Theta();           
    void callbackThread();
    void Set();
    void ResetMotion();
    void TATA();
    //모션 종료 확인 여부
    bool IsMotionFinish();

    const int SPIN_RATE = 100;
    
            
    int go = 0;
    int go_ = 0;
    int re = 0;
    int emergency = 99;
    int indext = 0;
    int mode = 0;                   
    int index_angle = 0;
    int angle = 0;


    double step = 0;
    double RL_th2 = 0, LL_th2 = 0;
    double RL_th1 = 0, LL_th1 = 0;
    double HS = 0;  
    double SR = 0; 
    double turn_angle = 0;  
    double turnRL_st=0;
    double turnLL_st=0;   

    VectorXd All_Theta = MatrixXd::Zero(NUMBER_OF_DYNAMIXELS, 1);
    VectorXd initial_theta = VectorXd::Zero(NUMBER_OF_DYNAMIXELS);
    bool initial_theta_saved = false;


    bool initial_yaw = false;
    double initial_N = 0; //북
    double initial_S = 0; //남
    double initial_E = 0; //동
    double initial_W = 0; //서
    double yaw_now = 0;
    int turn_angle1 = 0;
    int angle111 = 0;



    double walkfreq = 1.48114;
    double walktime = 2 / walkfreq;
    int freq = 100;
    int walktime_n = walktime * freq;
















    // double rl_neckangle = 0;                
    // double ud_neckangle = 0;                
    // double tmp_turn_angle = 0;              
    // bool emergency_ = 1;                    
    // double vel_x = 0;
    // double vel_y = 0;
    // double vel_z = 0;
    // int error_counter = 0;
    // bool error_printed = false;

    // int8_t mode = 99;                       
    // double walkfreq = 1.48114;             
    // double walktime = 2 / walkfreq;        
    // int freq = 100;                        
    // int walktime_n = walktime * freq;      


    // int upper_indext = 0;                 
    // int check_indext = 0;                 
    // int stop_indext = 0;  

    // bool turn_left = false;
    // bool turn_right = false;

    // bool on_emergency = false;

    // double angle = 0;
    // int index_angle = 0;

    // double Real_CP_Y = 0;
    // double Real_CP_X = 0;
    // double xZMP_from_CP = 0;
    // double yZMP_from_CP = 0;
    // double Real_zmp_y_accel = 0;

    // MatrixXd RL_motion, LL_motion;
    // MatrixXd RL_motion0, LL_motion0;
    // MatrixXd RL_motion1, LL_motion1;
    // MatrixXd RL_motion2, LL_motion2;
    // MatrixXd RL_motion3, LL_motion3;
    // MatrixXd RL_motion4, LL_motion4;
    // MatrixXd RL_motion5, LL_motion5;
    // MatrixXd RL_motion6, LL_motion6;
    // MatrixXd RL_motion7, LL_motion7;

};

#endif // CALLBACK_H
