
#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <span>
#include "BRP_Kinematics.hpp"
using namespace Eigen;
using namespace std;
#define DEG2rad 0.017453292519943

class Trajectory {
private:
	//Com member
	double walkfreq;
	double walktime;
	int walktime_n;
	double freq;
	double step;
	int step_n;
	double del_t;
	double z_c;
	double g;
	double T_prev;
	int NL;
	int sim_n;
	Matrix3d A;
	Vector3d B;
	RowVector3d C;
	double Qe;
	Matrix3d Qx;
	Matrix4d Q_p;
	Matrix<double, 1, 1> R;
	Vector4d I_p;
	Vector4d B_p;
	Matrix<double, 4, 3> F_p;
	Matrix4d A_p;
	Matrix4d K_p;
	double Gi;
	Matrix<double, 1, 3> Gx;
	Matrix4d Ac_p;
	MatrixXd Gd;
	double sim_time;
	RowVectorXd xzmp_ref;
	RowVectorXd yzmp_ref;

	
	VectorXd zmp_ref;
	VectorXd zmp_ref_fifo;
	VectorXd u;
	VectorXd zmp;
	VectorXd zmp_ref_final;
	VectorXd ref_xCP;
	VectorXd ref_yCP;
	double zmp_err_int;
	double u_prev;
	VectorXd Ref_Xpos;
	VectorXd Ref_Ypos;
	bool check_DD = true;
	Matrix<double, 6, 1> Angle_trajectorty_turn;
	Matrix<double, 6, 1> Angle_trajectorty_back;
	//Footmember
	Matrix<double, 6, 1> XStep;
	Matrix<double, 6, 1> XStride;
    Matrix<double, 6, 1> Xsn;
	double L0 = 0.06;



public:

	Trajectory();
	void Change_Freq(double f);
	// for Com
	void Set_step(double a);
	void Set_distance(double Goal_distance, double YPos);
	void Set_distance_start(double Goal_distance);

	void Set_distance_back(double Goal_distance);
	MatrixXd PreviewGd();
	double Return_Step_n();
	int Return_Walktime_n();
	int Return_Sim_n();
	VectorXd Get_xCP();
	VectorXd Get_yCP();
	RowVectorXd Get_xZMP();
	RowVectorXd Get_yZMP();
	MatrixXd YComSimulation();
	MatrixXd XComSimulation();

	MatrixXd Equation_solver(double t0, double t1, double start, double end);
	double Sinusoidal(double t, double T, double h);
	double Step(double t);
	double Stride(double t);
	double XSN(double t);
	MatrixXd RF_xsimulation_straightwalk();
	MatrixXd LF_xsimulation_straightwalk();
	MatrixXd RF_zsimulation_straightwalk(double h);
	MatrixXd LF_zsimulation_straightwalk(double h);
	MatrixXd YComSimulation_Sidewalk_half(double a, double b, double c, double d);

	void Go_Straight(double step, double distance, double height);
	void Go_Straight_start(double step, double distance, double height);
	void Go_Back_Straight(double step, double distance, double height);
	void Go_Back_Straight2(double step, double distance, double height);
	void Go_Back(double step, double distance, double height);
	void Freq_Change_Straight(double step, double distance, double height, double freq);
	void Stop_Trajectory_straightwalk(double step);
	void Step_in_place(double step, double distance, double height);
	void Side_Left1(double step);
	void Side_Right1(double step);
	void Picking_Motion(int FB_sim_n, int M_sim_n, double COM_h);
	
	
	
	
	MatrixXd RF_ysimulation_leftwalk_halfstep();
	MatrixXd LF_ysimulation_leftwalk_halfstep();
	MatrixXd RF_zsimulation_leftwalk_halfstep();
	MatrixXd LF_zsimulation_leftwalk_halfstep();
	MatrixXd RF_ysimulation_rightwalk_halfstep();
	MatrixXd LF_ysimulation_rightwalk_halfstep();
	MatrixXd RF_zsimulation_rightwalk_halfstep();
	MatrixXd LF_zsimulation_rightwalk_halfstep();
	
	
	
	
	void Huddle_Motion(double step, double height, double COM_h);
	MatrixXd RF_xsimulation_huddle();
	MatrixXd LF_xsimulation_huddle();
	MatrixXd RF_zsimulation_huddle(double h, double COM_h);
	MatrixXd LF_zsimulation_huddle(double h, double COM_h);
	MatrixXd Huddle_Xcom();
	MatrixXd Huddle_Ycom();
	MatrixXd RF_zsimulation_sitdown(double h);
	MatrixXd LF_zsimulation_sitdown(double h);
	MatrixXd RF_zsimulation_standup(double h);
	MatrixXd LF_zsimulation_standup(double h);

	MatrixXd Foot_zsimulation_sitdown(double h);
	MatrixXd Foot_zsimulation_standup(double h);



	void Side_Left2();
	MatrixXd YComSimulation_Sidewalk(double a, double b, double c, double d, double e, double f);
	MatrixXd RF_ysimulation_leftwalk();
	MatrixXd LF_ysimulation_leftwalk();
	MatrixXd RF_zsimulation_leftwalk();
	MatrixXd LF_zsimulation_leftwalk();



	MatrixXd z_position_sitdown(int M_sim_n, double h);
	MatrixXd zsimulation_sitdown_pick(int FB_sim_n, double h);
	MatrixXd zsimulation_standup_pick(int FB_sim_n, double h);

	void Make_turn_trajectory(double angle);
	double Return_turn_trajectory(double t);
	double Return_back_trajectory(double t);





	MatrixXd RF_zsimulation_sitdown3(double h);
	MatrixXd LF_zsimulation_sitdown3(double h);




	MatrixXd zsimulation_standup_Shoot_FINISH(int FB_sim_n, double h);

	MatrixXd Ref_RL_x;
	MatrixXd Ref_RL_y;
	MatrixXd Ref_RL_z;
	MatrixXd Ref_LL_x;
	MatrixXd Ref_LL_y;
	MatrixXd Ref_LL_z;

	MatrixXd lsRef_RL_x;
	MatrixXd lsRef_RL_y;
	MatrixXd lsRef_RL_z;
	MatrixXd lsRef_LL_x;
	MatrixXd lsRef_LL_y;
	MatrixXd lsRef_LL_z;
	MatrixXd rsRef_RL_x;
	MatrixXd rsRef_RL_y;
	MatrixXd rsRef_RL_z;
	MatrixXd rsRef_LL_x;
	MatrixXd rsRef_LL_y;
	MatrixXd rsRef_LL_z;
/////////////////////////////////////////////////////////////////com and foot position
	MatrixXd Xcom;
	MatrixXd Ycom;
	MatrixXd LF_xFoot;
	MatrixXd RF_xFoot;
	MatrixXd RF_yFoot;
	MatrixXd LF_yFoot;


	VectorXd Turn_Trajectory;
};














































class IK_Function
{
private:
	double walkfreq;
	double walktime;
	int walktime_n;
	double freq;
	double step;
	int step_n;
	double sim_time;
	int sim_n;

	//Inverse_kinematics member
	double L0 = 60;
	double L1 = 35.64;
	double L2 = 36.07;
	double L3 = 121.89;
	double L4 = 111.76;
	double L5 = 36.10;
	double L6 = 42.58;


	// double L0 = 60;
	// double L1 = 0;
	// double L2 = 0;
	// double L3 = 250;
	// double L4 = 250;
	// double L5 = 318;
	// double L6 = 77;



	double FW = 92.8;
	double FL = 137.8;
	double RL_th_IK[6] = { 0.,0.,0.,0.,0.,0. }, LL_th_IK[6] = { 0.,0.,0.,0.,0.,0. };
	double Ref_RL_PR[6] = { 0.,0.,0.,0.,0.,0. }, Ref_LL_PR[6] = { 0.,0.,0.,0.,0.,0. };
	double RL_th_FK[6] = { 0.,0.,0.,0.,0.,0. }, RL_PR_FK[6] = { 0.,0.,0.,0.,0.,0. };
	double LL_th_FK[6] = { 0.,0.,0.,0.,0.,0. }, LL_PR_FK[6] = { 0.,0.,0.,0.,0.,0. };
	double Foot_Height = 0;
	Matrix<double, 6, 1> RL_Compensation_Support_Leg_up;
	Matrix<double, 6, 1> RL_Compensation_Support_Leg_down;
	Matrix<double, 6, 1> RL_Compensation_Swing_Leg_up;
	Matrix<double, 6, 1> RL_Compensation_Swing_Leg_down;
	Matrix<double, 6, 1> RL_Compensation_Support_knee_up;
	Matrix<double, 6, 1> RL_Compensation_Support_knee_down;
	Matrix<double, 6, 1> RL_Compensation_Support_ankle_up;
	Matrix<double, 6, 1> RL_Compensation_Support_ankle_down;
	double RL_Swing_Leg_Compensation_up(double t);
	double RL_Swing_Leg_Compensation_down(double t);
	double RL_Support_Leg_Compensation_up(double t);
	double RL_Support_Leg_Compensation_down(double t);
	double RL_Support_Knee_Compensation_up(double t);
	double RL_Support_Knee_Compensation_down(double t);
	double RL_Support_Ankle_Compensation_up(double t);
	double RL_Support_Ankle_Compensation_down(double t);
	double RL_Support_Leg;
	double RL_Swing_Leg;
	double RL_Support_Knee;
	double RL_Support_Ankle;

	Matrix<double, 6, 1> LL_Compensation_Support_Leg_up;
	Matrix<double, 6, 1> LL_Compensation_Support_Leg_down;
	Matrix<double, 6, 1> LL_Compensation_Swing_Leg_up;
	Matrix<double, 6, 1> LL_Compensation_Swing_Leg_down;
	Matrix<double, 6, 1> LL_Compensation_Support_knee_up;
	Matrix<double, 6, 1> LL_Compensation_Support_knee_down;
	Matrix<double, 6, 1> LL_Compensation_Support_ankle_up;
	Matrix<double, 6, 1> LL_Compensation_Support_ankle_down;
	double LL_Swing_Leg_Compensation_up(double t);
	double LL_Swing_Leg_Compensation_down(double t);
	double LL_Support_Leg_Compensation_up(double t);
	double LL_Support_Leg_Compensation_down(double t);
	double LL_Support_Knee_Compensation_up(double t);
	double LL_Support_Knee_Compensation_down(double t);
	double LL_Support_Ankle_Compensation_up(double t);
	double LL_Support_Ankle_Compensation_down(double t);
	double LL_Support_Leg;
	double LL_Swing_Leg;
	double LL_Support_Knee;
	double LL_Support_Ankle;
	double Com_Height;

public:
	IK_Function();
	void Get_Step_n(double a);
	MatrixXd BRP_RL_Simulation(MatrixXd RFx, MatrixXd RFy, MatrixXd RFz);
	MatrixXd BRP_LL_Simulation(MatrixXd RFx, MatrixXd RFy, MatrixXd RFz);
    void BRP_Simulation(const MatrixXd& RFx, const MatrixXd& RFy, const MatrixXd& RFz, const MatrixXd& LFx, const MatrixXd& LFy, const MatrixXd& LFz, int Index_CNT);
	void Angle_Compensation(int indext, int size);
	void Fast_Angle_Compensation(int indext);
	void Forward_40step_Angle_Compensation(int indext);
	void Angle_Compensation_test(int indext);
	void Set_Angle_Compensation(int walktime_n);
	void Angle_Compensation_Huddle(int indext);










	void Change_Angle_Compensation(double RL_Support, double RL_Swing, double RL_Knee , double RL_Ankle, double LL_Support, double LL_Swing, double LL_Knee ,double LL_Ankle);
	double RL_th[6] = { 0.,0.,-0.610865,1.22173,0.610865,0. }, LL_th[6] = { 0.,0.,-0.610865,1.22173,0.610865, 0.};
	void Change_Com_Height(double h);
    
	double check_index;





	void Angle_Compensation_Leftwalk(int indext);
	void Angle_Compensation_Rightwalk(int indext);

};







































class Pick
{
private:
    double freq;
    double del_t;

public:
    // Constructor
    Pick();

    // Member functions
    void WT_Trajectory(double WT_th, int length);
    void RA_Trajectory(double RA_th1, double RA_th2, double RA_th3, double RA_th4, int length);
    void LA_Trajectory(double LA_th1, double LA_th2, double LA_th3, double LA_th4, int length);
    void NC_Trajectory(double NC_th1, double NC_th2, int length);
	void RL_Trajectory(double th1, double th2, double th3, double th4, double th5, double th6, int length);
	void LL_Trajectory(double th1, double th2, double th3, double th4, double th5, double th6, int length);

    void UPBD_SET(const MatrixXd& WT, const MatrixXd& RA, const MatrixXd& LA, const MatrixXd& NC, int Index_CNT);
	void UPBD_SET_ALL(const MatrixXd& WT, const MatrixXd& RA, const MatrixXd& LA,const MatrixXd& NC, const MatrixXd& RL, const MatrixXd& LL,int Index_CNT);
    void Fast_Arm_Swing(int indext, int walktime_n, double sim_n);

    void Picking(const MatrixXd& RFx, int Index_CNT, double& RL, double& LL);
	void Stand_up(const MatrixXd& RFx, int Index_CNT);
    void Huddle(const MatrixXd& RFx, int Index_CNT,double& RL, double& LL, double& RL_2, double& LL_2,double& HS, double& SR);
	void hhhh(const MatrixXd& RFx, int Index_CNT);
	void Back_Halfstep(const MatrixXd& RFx, int Index_CNT);
    // Public arrays
    double WT_th[1];
    double RA_th[4];
    double LA_th[4];
    double NC_th[2];
	double RL_th_ALL[6] = {0};
    double LL_th_ALL[6] = {0};

	MatrixXd Ref_WT_th;
    MatrixXd Ref_RA_th;
    MatrixXd Ref_LA_th;
    MatrixXd Ref_NC_th;
	MatrixXd Ref_RL_th;
	MatrixXd Ref_LL_th;

};