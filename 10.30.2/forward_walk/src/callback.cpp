#include "callback.hpp"
#include "sensor_msgs/msg/imu.hpp"  // IMU 메시지 타입 추가
#include <cmath>  
#include "robot_msgs/msg/line_result.hpp"
#include "sensor_msgs/msg/imu.hpp" //imu sensor
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

bool flgflg = 0;
FILE *Trajectory_all;

Callback::Callback(Trajectory *trajectoryPtr, IK_Function *IK_Ptr, Dxl *dxlPtr, Pick *pick_Ptr)
    : Node("callback_node_call"),  // Node 생성 시 노드 이름을 추가
      trajectoryPtr(trajectoryPtr),
      IK_Ptr(IK_Ptr),
      dxlPtr(dxlPtr),
      pick_Ptr(pick_Ptr),
      SPIN_RATE(100)     
{
    // ROS 2에서의 Node 객체 생성
    rclcpp::Node::SharedPtr nh = rclcpp::Node::make_shared("callback_node");
    
    // ROS 2에서 boost::thread 대신 std::thread 사용
    std::thread queue_thread(&Callback::callbackThread, this);
    queue_thread.detach();  // 비동기식 실행

    // Direction_subsciption_ = this->create_subscription<sensor_msgs::msg::Imu>(
    //     "/imu/data", 10, std::bind(&Callback::DirectionCallback, this, std::placeholders::_1));


    // set_subscriber_= this->create_subscription<std_msgs::msg::Bool>("/SET", 10, std::bind(&Callback::SetMode, this, std::placeholders::_1));

    // ROS 2의 subscription 생성
    start_subscriber_= this->create_subscription<std_msgs::msg::Bool>("/START", 10, std::bind(&Callback::StartMode, this, std::placeholders::_1));


    trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 675);
    trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 675);
    trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 675);
    trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 675);
    trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 675);
    trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 675);
    trajectoryPtr->Turn_Trajectory = VectorXd::Zero(135);
    omega_w = sqrt(g / z_c);

    pick_Ptr->Ref_WT_th = MatrixXd::Zero(1, 675);
    pick_Ptr->Ref_RA_th = MatrixXd::Zero(4, 675);
    pick_Ptr->Ref_LA_th = MatrixXd::Zero(4, 675);
    pick_Ptr->Ref_NC_th = MatrixXd::Zero(2, 675);

    
    emergency = 1;
    indext = 1;
    go = 0;

    RCLCPP_INFO(this->get_logger(), "Emergency value: %d", emergency);
    RCLCPP_INFO(this->get_logger(), "Callback activated");
}


void Callback::Set()
{
    All_Theta[0] = 0.0;
    All_Theta[1] = -0.050419;
    All_Theta[2] = -0.785155;
    All_Theta[3] = -0.327585;
    All_Theta[4] = 0.959987;
    All_Theta[5] = -0.032966;
    All_Theta[6] = 0.0;
    All_Theta[7] = 0.036848;
    All_Theta[8] = 0.785155;
    All_Theta[9] = 0.327585;
    All_Theta[10] = -0.907627;
    All_Theta[11] = -0.032966;

    // upper_body
    All_Theta[12] = pick_Ptr->WT_th[0] + step + 0 * DEG2RAD;  // waist
    All_Theta[13] = pick_Ptr->LA_th[0] + 90 * DEG2RAD; // L_arm
    All_Theta[14] = pick_Ptr->RA_th[0] - 90 * DEG2RAD; // R_arm
    All_Theta[15] = pick_Ptr->LA_th[1] - 60 * DEG2RAD; // L_arm
    All_Theta[16] = pick_Ptr->RA_th[1] + 60 * DEG2RAD; // R_arm
    All_Theta[17] = pick_Ptr->LA_th[2] - 90 * DEG2RAD; // L_elbow
    All_Theta[18] = pick_Ptr->RA_th[2] + 90 * DEG2RAD; // R_elbow
    All_Theta[19] = pick_Ptr->LA_th[3] - 0 * DEG2RAD; // L_hand
    All_Theta[20] = pick_Ptr->RA_th[3] + 0 * DEG2RAD; // R_hand
    All_Theta[21] = pick_Ptr->NC_th[0] + 0 * DEG2RAD; // neck_RL
    All_Theta[22] = pick_Ptr->NC_th[1] - 24 * DEG2RAD; // neck_UD
}


int Callback::GetTurnsRemaining() const {
    return turns_remaining_.load(std::memory_order_acquire);
}
const void* Callback::GetTurnsRemainingAddr() const {
    return static_cast<const void*>(&turns_remaining_);
}
// "남은 유닛 턴 수"를 delta만큼 감소시키고, '감소 전 값'을 반환.
// acq_rel: 감소 전 읽기(acquire)와 감소 쓰기(release) 모두의 메모리 순서 보장.
int Callback::FetchSubTurnsRemaining(int delta) {
    return turns_remaining_.fetch_sub(delta, std::memory_order_acq_rel);
}

// "남은 유닛 턴 수"를 v로 설정. release 오더: 이후 acquire 로드에서 이 쓰기 이전의 효과가 보장된다.
void Callback::SetTurnsRemaining(int v) {
    turns_remaining_.store(v, std::memory_order_release);
}


void Callback::OnLineResult(int angle)
{
    
    if (!line_turn)
    {
        RCLCPP_INFO(this->get_logger(), "[OnLineResult] ignored because line_turn==false");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "[OnLineResult] recv angle=%.2d, line_turn=%d",
            angle, (int)line_turn);

    double line_angle_ = angle;
    // 원래는 8도인데 실제 로봇이 8도를 넘게 돌아서 13으로 임의 설정
    const int stepdeg = 11;

    int turncount = static_cast<int>(line_angle_ / stepdeg);
    double extra_angle = static_cast<int>(line_angle_ - stepdeg * turncount);

    if (extra_angle > 9.5)
    {
        turncount += 1;
    }

    int before = turns_remaining_.load(std::memory_order_seq_cst);
    turns_remaining_.store(turncount, std::memory_order_seq_cst);
    int after  = turns_remaining_.load(std::memory_order_seq_cst);

    // RCLCPP_INFO(this->get_logger(),
    //   "[OnLineResult] turns_remaining_: %d -> %d (set=%d) @%p",
    //   before, after, turncount, static_cast<const void*>(&turns_remaining_));

    line_turn = false;

}


// // 현재 방향 기록
// void Callback::DirectionCallback(const sensor_msgs::msg::Imu::SharedPtr Direction_msg)
// {
//     tf2::Quaternion q(
//         Direction_msg->orientation.x,
//         Direction_msg->orientation.y,
//         Direction_msg->orientation.z,
//         Direction_msg->orientation.w);

//         // 쿼터니언 → roll, pitch, yaw 변환
//     double roll, pitch, yaw;
//     tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

//     yaw_now = yaw * 180.0 / M_PI; //라디안에서 dgree로
//     if (yaw_now < 0) yaw_now += 360; //0~360 보정
    
//     // 초기 방향 잡기 위해 한번만
//     if (!initial_yaw && yaw_now != 0){
//         initial_N = yaw_now; //북
//         //fmod(a,b) = a를 b로 나눈 나머지
//         initial_E = fmod(initial_N + 270, 360);   // 동
//         initial_W = fmod(initial_N + 90, 360);  // 서
//         initial_S = fmod(initial_N + 180, 360);  // 남
//         initial_yaw = true;
//         RCLCPP_INFO(this->get_logger(), "initial_N = %f, initial_E = %f, initial_W = %f, initial_S = %f",
//                 initial_N, initial_E, initial_W, initial_S);
//     }
// }

// 공 줍고 던진 후 몇도 돌아야할지 계산
// void Callback::CalculateTurn(int a)
// {
//     RCLCPP_INFO(this->get_logger(), "a = %d", a);
//     if(a == 1)
//     {
//         if(initial_E < yaw_now)
//         {
//             turn_angle1 = yaw_now - initial_E;
//         }
//         else if (initial_E > yaw_now)
//         {
//             turn_angle1 = yaw_now + 360 - initial_E;
//         }
//     }

//     else if(a == 2)
//     {
//         if(initial_S < yaw_now)
//         {
//             turn_angle1 = yaw_now - initial_S;
//         }
//         else if (initial_S > yaw_now)
//         {
//             turn_angle1 = yaw_now + 360 - initial_S;
//         }
//     }

//     else if(a == 3)
//     {
//         if(initial_E < yaw_now)
//         {
//             turn_angle1 = initial_E + 360 - yaw_now;
//         }
//         else if (initial_E > yaw_now)
//         {
//             turn_angle1 = initial_E - yaw_now;
//         }
//     }

//     else if(a == 4)
//     {
//         if(initial_N > yaw_now)
//         {
//             turn_angle1 = initial_N - yaw_now;
//         }
//         else if (initial_N < yaw_now)
//         {
//             turn_angle1 = initial_N + 360 - yaw_now;
//         }
//     }

//     angle111 = turn_angle1;
//     RCLCPP_INFO(this->get_logger(), "yaw_now = %f, turn_angle1 = %d", yaw_now, turn_angle1);

// }


// ros2 topic pub /START std_msgs/msg/Bool "data: true" -1

void Callback::StartMode(const std_msgs::msg::Bool::SharedPtr start)
{
    RCLCPP_DEBUG(this->get_logger(), "StartMode called with data: %d", start->data);
    if (start->data)
    {
        indext = 0;
        trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 30);
        trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 30);
        trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 30);
        trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 30);


        trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 30);
        trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 30);
        pick_Ptr->Ref_WT_th = MatrixXd::Zero(1, 30);
        pick_Ptr->Ref_RA_th = MatrixXd::Zero(4, 30);
        pick_Ptr->Ref_LA_th = MatrixXd::Zero(4, 30);
        pick_Ptr->Ref_NC_th = MatrixXd::Zero(2, 30);

        emergency = 0;
        // go = 1;

        RCLCPP_INFO(this->get_logger(), "StartMode activated with true data!");
    }
}


void Callback::TATA()
{
    double res_turn_angle = angle;

    if (res_turn_angle != 0)
    {
        turn_angle = res_turn_angle * DEG2RAD;
        trajectoryPtr->Make_turn_trajectory(turn_angle);
        // index_angle = 0;
    }
}


void Callback::callbackThread()
{
    // ROS 2의 spin 사용 대신 루프에서 메시지 처리
    rclcpp::Rate loop_rate(SPIN_RATE);
    
    while (rclcpp::ok())
    {
        rclcpp::spin_some(this->get_node_base_interface());
        loop_rate.sleep();
    }
}

void Callback::SetLineTurn(bool on)
{
    line_turn.store(on, std::memory_order_release);
    RCLCPP_INFO(this->get_logger(), "[line_turn] <- %d", static_cast<int>(on));
}

void Callback::SelectMotion(int go)
{

    go_ = go;

    // RCLCPP_WARN(this->get_logger(), "[SelectMotion] called: go=%d, re=%d", go, re);
    if(re == 0)
    {
        if(go == 0) //start_pose
        {
            startRL_st[0] = 0;
            startRL_st[1] = 0.001941;
            startRL_st[2] = 0.122416;
            startRL_st[3] = 0.196013;
            startRL_st[4] = -0.073595;//1.148133;
            startRL_st[5] = 0.001941;

            startLL_st[0] = 0;
            startLL_st[1] = 0.001941;
            startLL_st[2] = -0.122416;
            startLL_st[3] = -0.196013;
            startLL_st[4] = 0.073595;//-1.148133;
            startLL_st[5] = 0.001941;

        }
        if(go == 1)// 6step
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 0.3, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1, 2, 1, 0, -3); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        
        }

        else if(go == 2)//좌회전
        {
            re = 1;
            indext = 0;
            angle = 8;

            index_angle = 0; // ★ 턴 진행 인덱스 리셋 (로그용/안전) 
            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Step_in_place;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Step_in_place(0.05, 0.25, 0.025);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(1, 2, 0, -2, 1, 2, 0, -2);
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 3)//우회전
        {
            re = 1;
            indext = 0;
            angle = 8;

            index_angle = 0; // ★ 턴 진행 인덱스 리셋 (로그용/안전) 

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Step_in_place(0.05, 0.25, 0.025);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(1, 2, 0, -2, 0, 1, 0, -2);
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 4)//Back_2step
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Back_2step;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Back_Straight(-0.04, -0.18, 0.05);
            IK_Ptr->Change_Angle_Compensation(3, 4, 2, 0, 2, 2, 2, -4); 
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 5  || go == 22 || go == 28)//Back_Halfstep
        {
            re = 1;
            indext = 0;

            turnRL_st=6;
            // turnLL_st=0;



            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Back_Halfstep;
            IK_Ptr->Change_Com_Height(30);
            // trajectoryPtr->Go_Back_Straight(-0.04, -0.1, 0.05);
            trajectoryPtr->Go_Back_Straight(-0.04, -0.08, 0.05);
            // IK_Ptr->Change_Angle_Compensation(2.5, 3, 0, 0, 2.5, 3, 1.5, -4);   // right turn right
            // IK_Ptr->Change_Angle_Compensation(2.5, 3, 0, 0, 3, 4, 1.5, -4);  // right turn linear
            IK_Ptr->Change_Angle_Compensation(2.5, 3, 0, 0, 3, 4, 1.5, -4); // best
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Set_Angle_Compensation(135);
        }
                
        else if(go == 6)//Forward_Halfstep
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Forward_Halfstep;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight(0.01, 0.03, 0.025);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(1, 1, 0, -1, 1, 1, 0, -3);
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 7)//Left_Halfstep
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Left_Halfstep;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Side_Left1(0.0275);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            // IK_Ptr->Change_Angle_Compensation(-1, -1, -2, -6, 5, 7, -4, 0); //2,4,-3,1,2,7,2,-8
            IK_Ptr->Change_Angle_Compensation(2, 4, 2, 1, 2, 7, 0, -8);
            IK_Ptr->Set_Angle_Compensation(135);
        }


        else if(go == 8)//Right_Halfstep
        {
             re = 1;
            indext = 0;
            angle = 2;
            


            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Right_Halfstep;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Side_Right1(0.0275);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(3, 7, 7, -5, 1, 1, 0.5, -3);
            // IK_Ptr->Change_Angle_Compensation(2, 7, 4, -8, 2, 4, -4, 4);

            // IK_Ptr->Change_Angle_Compensation(3, 7, 0, -5, 1, 1, 1.3, -4);
            IK_Ptr->Set_Angle_Compensation(135);
        }



        else if(go == 9)//PickMotion
        {
            re = 1;
            indext = 0;
            // mode = Motion_Index::Picking_Ball;
            
            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr -> Picking_Motion(300, 150, 0.165);
        }
        
        else if(go == 10)   //공 다시 잡기
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1,200);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 200);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 200);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 200);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 200);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 200);

            pick_Ptr->WT_Trajectory(0,200);
            pick_Ptr->RA_Trajectory(0,0,0,0,200);
            pick_Ptr->LA_Trajectory(-70,0,0,0,200);
            pick_Ptr->NC_Trajectory(0,30,200); // 공 다시 잡으면서 고개도 올리기
            // pick_Ptr->NC_Trajectory(0,0,250);
        }

        else if(go == 11) //Catch_Finish
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 150);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 150);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 150);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 150);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 150);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 150);

            pick_Ptr->WT_Trajectory(0,150);
            pick_Ptr->RA_Trajectory(0,0,0,0,150);  
            pick_Ptr->LA_Trajectory(70,0,0,30,150);
            pick_Ptr->NC_Trajectory(0,0,150);
        }

        else if(go == 12) //1 step
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 0.15, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1.5, 2, 1, 0, -2.5); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 13) //forward_half_six
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight(0.02, 0.12, 0.03);    // basic
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            // IK_Ptr->Change_Angle_Compensation(3, 4, 2, 2, 2, 2, 2, -2); 
            IK_Ptr->Change_Angle_Compensation(1, 1, 0, -1, 1, 1, 0, -3);    // RL, LL plus outside
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 14) //forward_short
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight(0.01, 0.06, 0.03);   // closed

            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            // IK_Ptr->Change_Angle_Compensation(3, 4, 2, 2, 2, 2, 2, -2); 
            IK_Ptr->Change_Angle_Compensation(1, 1, 0, -1, 1, 1, 0, -3);    // RL, LL plus outside
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if (go == 17) //shoot_ready
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 500);

            pick_Ptr->WT_Trajectory(3,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->RA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->LA_Trajectory(-180,-22,50,-50,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->NC_Trajectory(0,0,trajectoryPtr->Ref_RL_x.cols());

        }

        else if (go == 18) //shoot
        {
            re = 1;            
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(10);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 400);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 400);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 400);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 400);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 400);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 400);
        }

        else if (go == 19) //shoot_finish
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 500);
            // trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 500);
            // trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_RL_z = trajectoryPtr->zsimulation_standup_Shoot_FINISH(500, -0.02);
            trajectoryPtr->Ref_LL_z = trajectoryPtr->zsimulation_standup_Shoot_FINISH(500, -0.02);

            pick_Ptr->WT_Trajectory(23,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->RA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->LA_Trajectory(150,22,-110,50,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->NC_Trajectory(0,-30,trajectoryPtr->Ref_RL_x.cols()); // 고개 내리기
            // pick_Ptr->NC_Trajectory(0,0,trajectoryPtr->Ref_RL_x.cols());

        }

        else if (go == 77) //recovery
        {
            re = 1;            
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 1900);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 1900);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 1900);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 1900);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 1900);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 1900);
        }

        else if(go == 20)//Huddle
        {
            // re = 1;
            // indext = 0;
            // trajectoryPtr->Change_Freq(2);
            // // mode = Motion_Index::Huddle;
            // IK_Ptr->Change_Com_Height(35);
            // trajectoryPtr->Huddle_Motion(0.22, 0.14, 0.05);
            // IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            // // IK_Ptr->Change_Angle_Compensation(6.5, 2.6, 0.0, 3.5, 6.5, 2.6, 0.0, -3.5);
            // IK_Ptr->Change_Angle_Compensation(6.5, 1.0, 0.0, 3.5, 6.5, 1.0, 0.0, 0.0);

            // IK_Ptr->Set_Angle_Compensation(135);
            re = 1;            
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 2700);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 2700);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 2700);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 2700);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 2700);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 2700);
        }

        else if(go == 21)//fast_forward_4step
        {


            re = 1;
            indext = 0;
            // angle = 1.5;


            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Step_in_place;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Freq_Change_Straight(0.05, 0.2, 0.05, 1);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 3, -1.5, 2, 1, -3, -2.5);   
            IK_Ptr->Set_Angle_Compensation(67);
        }

        else if (go == 23) //Up_Nc
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 500);

            pick_Ptr->WT_Trajectory(0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->RA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->LA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->NC_Trajectory(0,30,trajectoryPtr->Ref_RL_x.cols());

        }

        else if (go == 24) //Down_Nc
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 500);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 500);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 500);

            pick_Ptr->WT_Trajectory(0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->RA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->LA_Trajectory(0,0,0,0,trajectoryPtr->Ref_RL_x.cols());
            pick_Ptr->NC_Trajectory(0,-30,trajectoryPtr->Ref_RL_x.cols());

        }

        else if(go == 25)//40step
        {
            re = 1;
            indext = 0;
            // angle = 2.65;
            turnRL_st = 8;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 2.5, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1, 2, 1, 0, -3); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 26)//15step
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 1.5, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1, 2, 1, 0, -3); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 27)//2step
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 0.2, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1, 2, 1, 0, -3); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 29) //forward_left
        {
            re = 1;
            indext = 0;
            angle = 10;
            step = 0;            // ✅ 추가
            index_angle = 0;
            // index_angle = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 0.3, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, 1, 2, 1, 0, -5); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 30) //forward_right
        {
            re = 1;
            indext = 0;
            angle = 10;
            step = 0;            // ✅ 추가
            index_angle = 0;
                // index_angle = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Go_Straight_start(0.05, 0.3, 0.05);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, 0, 2, 1, 0, -4); 
            IK_Ptr->Set_Angle_Compensation(135);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 31) //Re_Catch2
        {
            re = 1;
            indext = 0;


            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Ref_RL_x = MatrixXd::Zero(1,150);
            trajectoryPtr->Ref_LL_x = MatrixXd::Zero(1, 150);
            trajectoryPtr->Ref_RL_y = -0.06 * MatrixXd::Ones(1, 150);
            trajectoryPtr->Ref_LL_y = 0.06 * MatrixXd::Ones(1, 150);
            trajectoryPtr->Ref_RL_z = MatrixXd::Zero(1, 150);
            trajectoryPtr->Ref_LL_z = MatrixXd::Zero(1, 150);

            pick_Ptr->WT_Trajectory(0,150);
            pick_Ptr->RA_Trajectory(0,0,0,0,150);
            pick_Ptr->LA_Trajectory(0,0,0,-30,150);
            pick_Ptr->NC_Trajectory(0,0,150); 
        }

        else if(go == 32)//40step fast
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Freq_Change_Straight(0.05, 2.5, 0.05, 1.5);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(2, 3, 0, -1, 2, 1, 0, -3);
            IK_Ptr->Set_Angle_Compensation(101);
            trajectoryPtr->Stop_Trajectory_straightwalk(0.05);
        }

        else if(go == 33)//Step_In_Place
        {
            re = 1;
            indext = 0;

            index_angle = 0; // ★ 턴 진행 인덱스 리셋 (로그용/안전) 
            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Step_in_place;
            IK_Ptr->Change_Com_Height(30);
            trajectoryPtr->Step_in_place(0.05, 0.25, 0.025);
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Change_Angle_Compensation(1, 2, 0, -2, 1, 2, 0, -2);
            IK_Ptr->Set_Angle_Compensation(135);
        }

        else if(go == 34)//Back_Halfstep
        {
            re = 1;
            indext = 0;

            trajectoryPtr->Change_Freq(2);
            // mode = Motion_Index::Back_Halfstep;
            IK_Ptr->Change_Com_Height(30);
            // trajectoryPtr->Go_Back_Straight(-0.04, -0.1, 0.05);
            trajectoryPtr->Go_Back_Straight(-0.04, -0.08, 0.05);
            IK_Ptr->Change_Angle_Compensation(2.5, 3, 0, 0, 2.5, 3, 1.5, -4);   // right turn right
            IK_Ptr->Get_Step_n(trajectoryPtr->Return_Step_n());
            IK_Ptr->Set_Angle_Compensation(135);
        }


    }

    
}

// 동작 중인지 판별
bool Callback::IsMotionFinish()
{
    return (indext >= trajectoryPtr->Ref_RL_x.cols());
}

void Callback::ResetMotion()
{
    indext = 0;
    re = 0;
    index_angle = 0;

    RCLCPP_INFO(rclcpp::get_logger("Callback"), "[ResetMotion] 인덱스 및 상태 초기화 완료");
}


void Callback::Write_All_Theta()
{
    if (emergency == 0)
    {
        if(re == 1)
        {
            int go = go_;
            RCLCPP_INFO(rclcpp::get_logger("Callback"), "[Write_All_Theta] indext = %d / step_n = %ld / go = %d", indext, trajectoryPtr->Ref_RL_x.cols(), go);
            
            if (go == 21) //fast_forward 4step
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Fast_Angle_Compensation(indext);
                if(indext>=67 && indext<=134)
                {
                    IK_Ptr->RL_th[0] = (trajectoryPtr->Turn_Trajectory(index_angle));
                    step = (IK_Ptr->RL_th[0])/2;
                    index_angle += 1;
                    // std::cout << "index_angle" << index_angle << "walktime_n" << walktime_n << std::endl;
                    if (index_angle > 66)
                    {
                        index_angle = 0;
                    }
                }
                // IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());
            }

            else if (go == 2 || go == 29)//Step_in_place 좌회전, forward_left
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());

                // std::cout << "indext" << indext << std::endl;
                if(indext > 135 && indext <= 270)
                {

                    IK_Ptr->LL_th[0] = trajectoryPtr->Turn_Trajectory(index_angle);
                    step = (IK_Ptr->LL_th[0])/2;
                    index_angle = index_angle + 1;
                    // std::cout << "index_angle" << index_angle << std::endl;
                    if (index_angle > walktime_n - 1)
                    {
                        index_angle = 0;
                    }
                }
            }

            else if (go == 3 || go == 30)//Step_in_place 우회전, forward_right
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());

                // std::cout << "indext" << indext << std::endl;
                if(indext>=67 && indext <=337)
                {
                    IK_Ptr->RL_th[0] = -(trajectoryPtr->Turn_Trajectory(index_angle));
                    step = (IK_Ptr->RL_th[0])/2;
                    index_angle += 1;
                    // std::cout << "index_angle" << index_angle << std::endl;
                    if (index_angle > walktime_n - 1)
                    {
                        index_angle = 0;
                    }
                }
            }

            else if (go == 4 || go == 5 || go == 22 || go == 28 || go == 33 || go == 34)//Back_Halfstep //Back_2step
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());

                // pick_Ptr->Back_Halfstep(trajectoryPtr->Ref_RL_x, indext);
                // std::cout << "indext" << indext << std::endl;
            }

            else if (go == 6)//Forward_Halfstep
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());
            }

            else if (go == 7)//Left_Halfstep
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation_Leftwalk(indext);
            }

            else if (go == 8)//Right_Halfstep
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation_Rightwalk(indext);

                if(indext>=130 && indext <=390)
                {
                    startRL_st[2] = angle * DEG2RAD;
                    startRL_st[4] = -angle * DEG2RAD;
                }

            }
            
            else if (go == 9) // Pick
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                
                pick_Ptr->Picking(trajectoryPtr->Ref_RL_x, indext, RL_th2, LL_th2);
            }

            else if (go == 10 || go == 11 || go == 17 || go == 19 || go == 24 || go == 23 || go == 31) //Re_Catch, Catch_Finish, shoot_ready, shoot_finish, Up_Nc, Down_Nc
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                pick_Ptr->UPBD_SET(pick_Ptr->Ref_WT_th, pick_Ptr->Ref_RA_th, pick_Ptr->Ref_LA_th, pick_Ptr->Ref_NC_th, indext);

            }
            else if (go == 12 || go == 1 || go == 25 || go ==26 || go == 27 || go == 13 || go == 14) // forward 모음
            {                
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Angle_Compensation(indext, trajectoryPtr->Ref_RL_x.cols());

                
                if(go == 25)
                {

                    // std::cout << "indext" << indext << std::endl;
                    int period = 405;
                    int local_index = indext % period;

                    if (local_index >= 67 && local_index <= 337)
                    {
                        IK_Ptr->RL_th[0] = -(trajectoryPtr->Turn_Trajectory(index_angle));
                        step = (IK_Ptr->RL_th[0]) / 2;
                        index_angle += 1;

                        if (index_angle > walktime_n - 1)
                        {
                            index_angle = 0;
                        }
                    }

                }
            }
            
            else if (go == 18) //shoot
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                pick_Ptr->WT_th[0] =  -23 * DEG2RAD;
                pick_Ptr->LA_th[0] = -150 * DEG2RAD;
                pick_Ptr->LA_th[1] = -22 * DEG2RAD;
                pick_Ptr->LA_th[2] = 110 * DEG2RAD;
                pick_Ptr->LA_th[3]  = -50 * DEG2RAD;
            }

            else if (go == 32) //forward 40step fast
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                IK_Ptr->Forward_40step_Angle_Compensation(indext);
            }

            else if (go == 77)
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                pick_Ptr->Stand_up(trajectoryPtr->Ref_RL_x, indext);
            }
            else if (go == 20)
            {
                IK_Ptr->BRP_Simulation(trajectoryPtr->Ref_RL_x, trajectoryPtr->Ref_RL_y, trajectoryPtr->Ref_RL_z, trajectoryPtr->Ref_LL_x, trajectoryPtr->Ref_LL_y, trajectoryPtr->Ref_LL_z, indext);
                
                pick_Ptr->hhhh(trajectoryPtr->Ref_RL_x, indext);
            }
        }

    }

    indext += 1;

    if (indext >= trajectoryPtr->Ref_RL_x.cols() && indext != 0)
    {

        re = 0;


        turnRL_st=0;
    }



    // All_Theta 계산 및 저장
    All_Theta[0] = -IK_Ptr->RL_th[0] + pick_Ptr->RL_th_ALL[0] + startRL_st[0] + turnRL_st * DEG2RAD;
    All_Theta[1] = IK_Ptr->RL_th[1] + pick_Ptr->RL_th_ALL[1] +startRL_st[1] -RL_th1 * DEG2RAD - 3 * DEG2RAD;
    All_Theta[2] = IK_Ptr->RL_th[2] + pick_Ptr->RL_th_ALL[2] +startRL_st[2] -RL_th2 * DEG2RAD - 17 * DEG2RAD; //10.74 right
    All_Theta[3] = -IK_Ptr->RL_th[3] + pick_Ptr->RL_th_ALL[3] +startRL_st[3] + 40 * DEG2RAD; //38.34 * DEG2RAD;   
    All_Theta[4] = -IK_Ptr->RL_th[4] + pick_Ptr->RL_th_ALL[4] +startRL_st[4] + 24.22 * DEG2RAD;
    All_Theta[5] = -IK_Ptr->RL_th[5] + pick_Ptr->RL_th_ALL[5] +startRL_st[5] - 2* DEG2RAD;

    All_Theta[6] = -IK_Ptr->LL_th[0] + pick_Ptr->LL_th_ALL[0] +startLL_st[0];
    All_Theta[7] = IK_Ptr->LL_th[1] + pick_Ptr->LL_th_ALL[1] +startLL_st[1] +LL_th1 * DEG2RAD + 2 * DEG2RAD;
    All_Theta[8] = -IK_Ptr->LL_th[2] + pick_Ptr->LL_th_ALL[2] +startLL_st[2] +LL_th2 * DEG2RAD + 17 * DEG2RAD; //left
    All_Theta[9] = IK_Ptr->LL_th[3] + pick_Ptr->LL_th_ALL[3] +startLL_st[3] - 40 * DEG2RAD; //40.34 * DEG2RAD;  
    All_Theta[10] = IK_Ptr->LL_th[4] + pick_Ptr->LL_th_ALL[4] +startLL_st[4] - HS * DEG2RAD - 21.22 * DEG2RAD;
    All_Theta[11] = -IK_Ptr->LL_th[5] + pick_Ptr->LL_th_ALL[5] +startLL_st[5] - 2 * DEG2RAD;

    // upper_body
    All_Theta[12] = pick_Ptr->WT_th[0] + step + 0 * DEG2RAD;  // waist
    All_Theta[13] = pick_Ptr->LA_th[0] + 90 * DEG2RAD; // L_arm
    All_Theta[14] = pick_Ptr->RA_th[0] - 90 * DEG2RAD; // R_arm
    All_Theta[15] = pick_Ptr->LA_th[1] - 60 * DEG2RAD; // L_arm
    All_Theta[16] = pick_Ptr->RA_th[1] + 60 * DEG2RAD; // R_arm
    All_Theta[17] = pick_Ptr->LA_th[2] - 90 * DEG2RAD; // L_elbow
    All_Theta[18] = pick_Ptr->RA_th[2] + 90 * DEG2RAD; // R_elbow
    All_Theta[19] = pick_Ptr->LA_th[3] - 0 * DEG2RAD; // L_hand
    All_Theta[20] = pick_Ptr->RA_th[3] + 0 * DEG2RAD; // R_hand
    All_Theta[21] = pick_Ptr->NC_th[0] + 0 * DEG2RAD; // neck_RL
    All_Theta[22] = pick_Ptr->NC_th[1] - 24 * DEG2RAD; // neck_UD


    if(go == 0)
    {
        for (int i = 0; i < 6; i++) 
        {
            startRL_st[i] = 0.0;
            startLL_st[i] = 0.0;
        }
    }

    // 디버깅 정보 출력
//     for (int i = 0; i < All_Theta.size(); ++i)
//     {
//         RCLCPP_INFO(this->get_logger(), "All_Theta[%d] = %f", i, All_Theta[i]);
//     }
}