// #include "BRP_Kinematics.hpp"
#include "NewPattern2.hpp"
#include <cmath>
using namespace Eigen;
using namespace std;
using namespace BRP_Kinematics;
#define deg2rad 0.017453292519943
#define rad2deg 57.295779513082323
#define PI 3.141592653589793


Trajectory::Trajectory()
{
	walkfreq = 1.48114; // 1초 걸음 수 (보행 주파수 Hz)
	walktime = 2 / walkfreq; // 왼발 오른발 나누어 두번 걸으므로..!
	freq = 100; // 제어 루프의 주파수!
	walktime_n = walktime * freq; // 한 걸음동안 제어 루프가 몇 번 도는지 (샘플 수)!
	del_t = 1 / freq; // 제어루프의 시간 간격!
	z_c = 1.2 * 0.28224;  // CoM 높이 -> 1.2는 scale factor!
	g = 9.81; // 중력가속도!
	T_prev = 1.5; // Preview 제어 시간!
	NL = T_prev * freq; // Preview 샘플 수!
	A << 1, del_t, del_t * del_t / 2, // 시스템의 상태 : 위치 속도 가속도
		0, 1, del_t,
		0, 0, 1;
	B << del_t * del_t * del_t / 6, del_t * del_t / 2, del_t; // 입력 행렬 - 가속도를 제어 입력 취급
	C << 1, 0, -z_c / g; // CoM 위치와 가속도로 ZMP 계산
	
	
	////////////////// 가중치 행렬 //////////////////
	Qe = 1; // ZMP 오차에 대한 가중치
	Qx = Matrix3d::Zero(); // 상태 가중치 (위치, 속도, 가속도에 대한 penalty=0)
	Q_p = Matrix4d::Zero(); // Preview 제어용 결합 가중치 행렬
	Q_p << Qe, 0, 0, 0,
		0, Qx(0, 0), Qx(0, 1), Qx(0, 2),
		0, Qx(1, 0), Qx(1, 1), Qx(1, 2),
		0, Qx(2, 0), Qx(2, 1), Qx(2, 2);	// zmp 오차와 상태 가중치 조합 행렬 
	R << pow(10, -6);
	I_p << 1, 0, 0, 0;
	
	//// Preview 제어용 B,A 행렬 확장 //////
	B_p.row(0) = C * B;
	B_p.row(1) = B.row(0);
	B_p.row(2) = B.row(1);
	B_p.row(3) = B.row(2);	// ZMP 오차를 포함한 입력 행렬

	F_p.row(0) = C * A;
	F_p.row(1) = A.row(0);
	F_p.row(2) = A.row(1);
	F_p.row(3) = A.row(2);	// ZMP 오차를 포함한 상태 행렬

	A_p = Matrix4d::Identity();
	A_p.block<4, 3>(0, 1) = F_p;	// Preview 제어용 확장 상태 행렬

	K_p << 41.1035354770549, 824.198546618308, 163.877879931218, 0.866971302114951,
		824.198546618308, 17489.6862079445, 3483.96465227876, 19.6487374770632,
		163.877879931218, 3483.96465227876, 694.154076281009, 3.94329494961909,
		0.866971302114951, 19.6487374770632, 3.94329494961909, 0.0282399782535132;		// 미리 계산된 LQR Gain 행렬

	/////////// Preview 제어기 이득 계산 /////////////
	Gi = (R + B_p.transpose() * K_p * B_p).inverse() * B_p.transpose() * K_p * I_p;		// 오차누적에 대한 이득 (zmp 오차 적분 -> 제어입력 변환)
	Gx = (R + B_p.transpose() * K_p * B_p).inverse() * B_p.transpose() * K_p * F_p;		// 현재 상태에 대한 이득
	Gd = MatrixXd::Zero(NL, 1);		// Preview Horizon에 대한 feedforward 이득
	Ac_p = A_p - B_p * (R + B_p.transpose() * K_p * B_p).inverse() * B_p.transpose() * K_p * A_p;
	zmp_err_int = 0;	// ZMP 오차 적분값 초기화
	u_prev = 0;		// 이전 제어 입력 초기화
	step = 0.02;	// 한 스텝 거리
	step_n = 0;		
	sim_n = 0;
	sim_time = 0;	// 시뮬레이션 관련 카운터 초기화
	Angle_trajectorty_turn << 0, 0, 0, 0, 0, 0;	
	Angle_trajectorty_back << 0, 0, 0, 0, 0, 0;		// 보행 중 회전 및 후진 동작용 궤적 초기화

	XStep << 0, 0, 0, 0, 0, 0;
	XStride << 0, 0, 0, 0, 0, 0;		// 보행 스텝 별 위치 및 보폭 값 초기화
};

void Trajectory::Change_Freq(double f)		// 보폭 속도를 조절할때 사용
{
	walktime = f / walkfreq;
	walktime_n = walktime * freq;
}		// 외부 인자 f를 기준으로 walktime(보행주기), walktime_n(제어 샘플 수) 다시 계산

void Trajectory::Set_step(double a)		// 보폭 크기를 바꾸고 싶을 때 사용 
{
	step = a;
}

void Trajectory::Set_distance(double Goal_distance, double YPos)		// ZMP 및 CoM 궤적 생성을 위한 기준 위치 벡터 Ref_X,Ypos 설정
{ // ROS_WARN("tmp_img_proc_line_det_flg_ : %d", tmp_img_proc_line_det_flg_);
        // ROS_WARN("tmp_img_proc_no_line_det_flg_ : %d", tmp_img_proc_no_line_det_flg_);
        // ROS_WARN("tmp_img_proc_huddle_det_flg_2d_ : %d", tmp_img_proc_huddle_det_flg_2d_);
        // ROS_WARN("tmp_img_proc_ball_det_flg_ : %d", tmp_img_proc_ball_det_flg_);
        // ROS_WARN("is_in_pick_mode_ : %d", is_in_pick_mode_);
        // ROS_WARN("is_in_huddle_mode_ : %d", is_in_huddle_mode_);
        // ROS_WARN("tmp_pick_seq : %d", tmp_pick_seq);
        // ROS_WARN("tmp_huddle_seq : %d", tmp_huddle_seq);
        // ROS_WARN("is_in_shoot_mode_ : %d", is_in_shoot_mode_);
	step_n = (Goal_distance + 2 * step) / (2 * step);	// 스텝 수 계산
	sim_n = walktime_n * step_n;	// 시뮬레이션 프레임 수 - zmp, com 궤적 생성 수

	Ref_Xpos = VectorXd::Zero(2 * step_n);	// 전진방향 기준 위치 (보폭 기준)
	for (int i = 0; i < (2 * step_n); i++){
		if (i == 0) Ref_Xpos(i) = 0;
		else if (i < 2 * step_n - 1) Ref_Xpos(i) = step * (i - 1);
		else if (i == 2 * step_n - 1) Ref_Xpos(i) = step * (i - 2);
	}

	Ref_Ypos = VectorXd::Zero(2 * step_n);		// 좌우 보행 기준 위치
	for (int i = 0; i < (2 * step_n); i++){
		if (i < 1) Ref_Ypos(i) = 0;
		else if (i > 2 * step_n - 2) Ref_Ypos(i) = 0;
		else if ((i + 1) % 2 == 0) Ref_Ypos(i) = YPos;
		else if ((i + 1) % 2 == 1) Ref_Ypos(i) = -0.055;
	}
}

void Trajectory::Set_distance_start(double Goal_distance) // 위의 함수와 같으나, 시작할때 사용인지, 보행중 사용인지 차이
{
	step_n = (Goal_distance + 2 * step) / (2 * step);
	sim_n = walktime_n * step_n;

	Ref_Xpos = VectorXd::Zero(2 * step_n);
	for (int i = 0; i < (2 * step_n); i++){
		if (i == 0) Ref_Xpos(i) = 0;
		else if (i < 2 * step_n - 1) Ref_Xpos(i) = step * (i - 1);
		else if (i == 2 * step_n - 1) Ref_Xpos(i) = step * (i - 2);
	}

	Ref_Ypos = VectorXd::Zero(2 * step_n);
	for (int i = 0; i < (2 * step_n); i++){
		if (i < 1) Ref_Ypos(i) = 0;
		else if (i > 2 * step_n - 2) Ref_Ypos(i) = 0;
		else if ((i + 1) % 2 == 0) Ref_Ypos(i) = 0.057;
		else if ((i + 1) % 2 == 1) Ref_Ypos(i) = -0.06;
	}
}

void Trajectory::Set_distance_back(double Goal_distance)	// 뒤로 걷기를 위한,,
{
	step_n = (Goal_distance + 2 * step) / (2 * step);
	sim_n = walktime_n * step_n;

	Ref_Xpos = VectorXd::Zero(2 * step_n);
	for (int i = 0; i < (2 * step_n); i++){
		if (i == 0) Ref_Xpos(i) = 0;
		else if (i < 2 * step_n - 1) Ref_Xpos(i) = (step-0.007) * (i - 1);
		else if (i == 2 * step_n - 1) Ref_Xpos(i) = (step) * (i - 2);
	}

	Ref_Ypos = VectorXd::Zero(2 * step_n);
	for (int i = 0; i < (2 * step_n); i++){
		if (i < 1) Ref_Ypos(i) = 0;
		else if (i > 2 * step_n - 2) Ref_Ypos(i) = 0;
		else if ((i + 1) % 2 == 0) Ref_Ypos(i) = 0.055; //0.065
		else if ((i + 1) % 2 == 1) Ref_Ypos(i) = -0.05; //-0.05
	}
}

double Trajectory::Return_Step_n(){		// 현재 보행 거리 기준으로 계산된 step 수
	return step_n;
}
int Trajectory::Return_Walktime_n(){ return walktime_n; } // 한 걸음 동안 제어 루프가 몇 번 도는지 반환
int Trajectory::Return_Sim_n(){	return sim_n; } // 전체 보행 시뮬레이션 프레임 수 (걸음 수 * walktime_n) \

MatrixXd Trajectory::PreviewGd(){		// Preview control의 이득 행렬 Gd 계산
	for (int l = 0; l < NL; l++)
	{

		Matrix4d temp = Ac_p.transpose();
		for (int i = 1; i < l; i++){
			temp = temp * Ac_p.transpose();
		}
		Gd(l, 0) = (R + B_p.transpose() * K_p * B_p).inverse() * B_p.transpose() * temp * K_p * I_p;
		if (l == 0)
			Gd(l, 0) = (R + B_p.transpose() * K_p * B_p).inverse() * B_p.transpose() * K_p * I_p;
	}
	return Gd;
};

MatrixXd Trajectory::XComSimulation()		// X축 방향으로 com이 어떻게 이동해야 zmp 기준에 맞춰 움직일 수 있을지 궤적 계산
{
	PreviewGd();
	zmp_err_int = 0;
	u_prev = 0;
	int Com_flag = 0;
	float float_Com_flag = 0;
	float float_walktime_n = walktime_n;
	RowVectorXd zmp_ref(sim_n + 19);
	for (int i = 0; i < sim_n + 19; i++)
	{
		if (i < float_walktime_n) zmp_ref(i) = 0;
		else if (i > sim_n - 1) zmp_ref(i) = Ref_Xpos(2 * step_n - 1);
		else if (i < (float_Com_flag * 0.5 + 0.5) * float_walktime_n) zmp_ref(i) = Ref_Xpos(Com_flag);
		else if (i < (float_Com_flag * 0.5 + 1) * float_walktime_n) zmp_ref(i) = Ref_Xpos(Com_flag + 1);
		
		if ((i + 1) % walktime_n == 0){
			Com_flag = Com_flag + 2;
			float_Com_flag = float_Com_flag + 2;
		}
	}

	VectorXd zmp_ref_fifo(NL);
	VectorXd u(sim_n + 19);
	VectorXd zmp(sim_n + 19);
	VectorXd zmp_ref_final(sim_n + 19);
	VectorXd CP(sim_n + 19);
	MatrixXd XCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	XCom.setZero();
	double w = sqrt(g / z_c);
	for (int i = 0; i < sim_n + 19; i++){
		for (int j = 0; j < NL; j++){
			if (i + j < sim_n + 19) zmp_ref_fifo[j] = zmp_ref[i + j];
			else zmp_ref_fifo[j] = zmp_ref[sim_n + 18];
		}
		u_prev = 0;
		for (int j = 0; j < NL; j++){
			u_prev += Gd(j, 0) * zmp_ref_fifo[j];
		}
		u[i] = Gi * zmp_err_int - Gx * XCom.col(i) + u_prev;
		XCom.col(i + 1) = A * XCom.col(i) + B * u[i];
		zmp[i] = C * XCom.col(i);
		zmp_err_int += (zmp_ref[i] - zmp[i]);
		CP[i] = XCom(0, i) + 1 / w * XCom(1, i);
		zmp_ref_final[i] = zmp_ref[i];
	}
	xzmp_ref = zmp_ref;
	ref_xCP = CP.block(18, 0, sim_n, 1);
	MatrixXd XCom_ = XCom.block(0, 18, 1, sim_n);
	return XCom_;
};

MatrixXd Trajectory::YComSimulation()		// Y축 방향으로 com이 어떻게 이동해야 zmp 기준에 맞춰 움직일 수 있을지 궤적 계산
{
	PreviewGd();
	zmp_err_int = 0;
	u_prev = 0;
	int Com_flag = 0;
	float float_Com_flag = 0;
	float float_walktime_n = walktime_n;
	RowVectorXd zmp_ref(sim_n + 19);

	for (int i = 0; i < sim_n + 19; i++) {
		if (i > sim_n - 1) zmp_ref(i) = 0;
		else if (i < (float_Com_flag * 0.5 + 0.5) * float_walktime_n) zmp_ref(i) = Ref_Ypos(Com_flag);
		else if (i < (float_Com_flag * 0.5 + 1) * float_walktime_n) zmp_ref(i) = Ref_Ypos(Com_flag + 1);

		if ((i + 1) % walktime_n == 0) {
			Com_flag = Com_flag + 2;
			float_Com_flag = float_Com_flag + 2;
		}
	}

	VectorXd zmp_ref_fifo(NL);
	VectorXd u(sim_n + 19);
	VectorXd zmp(sim_n + 19);
	VectorXd zmp_ref_final(sim_n + 19);
	VectorXd CP(sim_n + 19);
	MatrixXd YCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	YCom.setZero();
	double w = sqrt(g / z_c);
	for (int i = 0; i < sim_n + 19; i++){
		for (int j = 0; j < NL; j++){
			if (i + j < sim_n + 19) zmp_ref_fifo[j] = zmp_ref[i + j];
			else zmp_ref_fifo[j] = zmp_ref[sim_n + 18];
		}
		u_prev = 0;
		for (int j = 0; j < NL; j++){
			u_prev = u_prev + Gd(j, 0) * zmp_ref_fifo[j];
		}
		u[i] = Gi * zmp_err_int - Gx * YCom.col(i) + u_prev;
		YCom.col(i + 1) = A * YCom.col(i) + B * u[i];
		zmp[i] = C * YCom.col(i);
		zmp_err_int += (zmp_ref[i] - zmp[i]);
		CP[i] = YCom(0, i) + 1.0 / w * YCom(1, i);
		zmp_ref_final[i] = zmp_ref[i];
	}
	yzmp_ref = zmp_ref;
	ref_yCP = CP.block(18, 0, sim_n, 1);
	MatrixXd YCom_ = YCom.block(0, 18, 1, sim_n);
	return YCom_;
}




MatrixXd Trajectory::YComSimulation_Sidewalk_half(double a, double b, double c, double d)
{
	step_n = 3;
	sim_n = walktime_n * step_n;
	PreviewGd();
	RowVectorXd zmp_ref(sim_n + 19);
	for (int i = 0; i < sim_n + 19; i++)
	{
		double time = i * del_t;
		if (time < 1 * walktime)
		{
			zmp_ref[i] = 0;
		}
		else if (time < 1.5 * walktime)
		{
			zmp_ref[i] = a;
		}
		else if (time < 2 * walktime)
		{
			zmp_ref[i] = b;
		}
		else if (time < 2.5 * walktime)
		{
			zmp_ref[i] = c;
		}
		else if (time < 3 * walktime)
		{
			zmp_ref[i] = d;
		}
		else
		{
			zmp_ref[i] = d;
		}
	}
	VectorXd zmp_ref_fifo(NL);
	VectorXd u(sim_n + 19);
	VectorXd zmp(sim_n + 19);
	VectorXd zmp_ref_final(sim_n + 19);
	VectorXd CP(sim_n + 19);
	MatrixXd YCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	YCom.setZero();
	double w = sqrt(g / z_c);
	for (int i = 0; i < sim_n + 19; i++)
	{
		for (int j = 0; j < NL; j++)
		{
			if (i + j < sim_n + 19)
			{
				zmp_ref_fifo[j] = zmp_ref[i + j];
			}
			else
			{
				zmp_ref_fifo[j] = zmp_ref[sim_n + 18];
			}
		}
		u_prev = 0;
		for (int j = 0; j < NL; j++)
		{
			u_prev = u_prev + Gd(j, 0) * zmp_ref_fifo[j];
		}
		u[i] = Gi * zmp_err_int - Gx * YCom.col(i) + u_prev;

		YCom.col(i + 1) = A * YCom.col(i) + B * u[i];

		zmp[i] = C * YCom.col(i);

		zmp_err_int += (zmp_ref[i] - zmp[i]);

		CP[i] = YCom(0, i) + 1.0 / w * YCom(1, i);

		zmp_ref_final[i] = zmp_ref[i];
	}
	MatrixXd YCom_ = YCom.block(0, 18, 1, sim_n);
	return YCom_;
}
//////////////////////////////////////////////////////////
VectorXd Trajectory::Get_xCP() {return ref_xCP;}

VectorXd Trajectory::Get_yCP() {return ref_yCP;}

RowVectorXd Trajectory::Get_xZMP() {return xzmp_ref;}

RowVectorXd Trajectory::Get_yZMP() {return yzmp_ref;}


//////////////////// 5차 방정식 궤적 계산기 /////////////////////
MatrixXd Trajectory::Equation_solver(double t0, double t1, double start, double end)
{
	Matrix<double, 6, 6> A;
	Matrix<double, 6, 1> B;
	Matrix<double, 6, 1> X;
	A << 1, t0, pow(t0, 2), pow(t0, 3), pow(t0, 4), pow(t0, 5),
		0, 1, 2 * t0, 3 * pow(t0, 2), 4 * pow(t0, 3), 5 * pow(t0, 4),
		0, 0, 2, 6 * t0, 12 * pow(t0, 2), 20 * pow(t0, 3),
		1, t1, pow(t1, 2), pow(t1, 3), pow(t1, 4), pow(t1, 5),
		0, 1, 2 * t1, 3 * pow(t1, 2), 4 * pow(t1, 3), 5 * pow(t1, 4),
		0, 0, 2, 6 * t1, 12 * pow(t1, 2), 20 * pow(t1, 3);
	B << start, 0, 0, end, 0, 0;
	X = A.colPivHouseholderQr().solve(B);
	return X;
};

double Trajectory::Sinusoidal(double t, double T, double h) {return h * 0.5 * (1 - cos(PI * t / T));};

double Trajectory::Step(double t){
	double X = XStep(0) + XStep(1) * t + XStep(2) * pow(t, 2) + XStep(3) * pow(t, 3) + XStep(4) * pow(t, 4) + XStep(5) * pow(t, 5);
	return X;
};

double Trajectory::Stride(double t){
	double X = XStride(0) + XStride(1) * t + XStride(2) * pow(t, 2) + XStride(3) * pow(t, 3) + XStride(4) * pow(t, 4) + XStride(5) * pow(t, 5);
	return X;
};

double Trajectory::XSN(double t){
	double X = Xsn(0) + Xsn(1) * t + Xsn(2) * pow(t, 2) + Xsn(3) * pow(t, 3) + Xsn(4) * pow(t, 4) + Xsn(5) * pow(t, 5);
	return X;
};

/////////////////// 발 경로 계산기 /////////////////////////////////////////


void Trajectory::Step_in_place(double step, double distance, double height)
{
	Set_step(step);
	Set_distance(distance, 0.05);
	Ycom = YComSimulation();
	RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = MatrixXd::Zero(1, sim_n);
	Ref_LL_x = MatrixXd::Zero(1, sim_n);
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
}

void Trajectory::Huddle_Motion(double step, double height, double COM_h)
{
	// 100 810 100 1010
	Set_step(step);
	MatrixXd Ref_RL_x_t = RF_xsimulation_huddle() - Huddle_Xcom();
	MatrixXd Ref_LL_x_t = LF_xsimulation_huddle() - Huddle_Xcom();
	MatrixXd Ref_RL_y_t =  -L0 * MatrixXd::Ones(1, sim_n) - Huddle_Ycom();
	MatrixXd Ref_LL_y_t = + L0 * MatrixXd::Ones(1, sim_n) - Huddle_Ycom();
	MatrixXd Ref_RL_z_t = RF_zsimulation_huddle(height, COM_h);
	MatrixXd Ref_LL_z_t = LF_zsimulation_huddle(height, COM_h);
	MatrixXd Temp_RL_x(Ref_RL_x_t.rows(), MatrixXd::Zero(1, 100).cols() + Ref_RL_x_t.cols() + MatrixXd::Zero(1, 100).cols());
	MatrixXd Temp_LL_x(Ref_LL_x_t.rows(), MatrixXd::Zero(1, 100).cols() + Ref_LL_x_t.cols() + MatrixXd::Zero(1, 100).cols());
	MatrixXd Temp_RL_y(Ref_RL_y_t.rows(), MatrixXd::Ones(1, 100).cols() + Ref_RL_y_t.cols() + MatrixXd::Ones(1, 100).cols());
	MatrixXd Temp_LL_y(Ref_LL_y_t.rows(), MatrixXd::Ones(1, 100).cols() + Ref_LL_y_t.cols() + MatrixXd::Ones(1, 100).cols());
	MatrixXd Temp_RL_z(Ref_RL_z_t.rows(), Foot_zsimulation_sitdown(COM_h).cols() + Ref_RL_z_t.cols() + Foot_zsimulation_standup(COM_h).cols());
	MatrixXd Temp_LL_z(Ref_LL_z_t.rows(), Foot_zsimulation_sitdown(COM_h).cols() + Ref_LL_z_t.cols() + Foot_zsimulation_standup(COM_h).cols());
	Temp_RL_x << MatrixXd::Zero(1, 100), Ref_RL_x_t, MatrixXd::Zero(1, 100);
	Temp_LL_x << MatrixXd::Zero(1, 100), Ref_LL_x_t, MatrixXd::Zero(1, 100);
	Temp_RL_y << -L0 * MatrixXd::Ones(1, 100), Ref_RL_y_t, -L0 * MatrixXd::Ones(1, 100);
	Temp_LL_y << L0 * MatrixXd::Ones(1, 100), Ref_LL_y_t, L0 * MatrixXd::Ones(1, 100);
	Temp_RL_z << Foot_zsimulation_sitdown(COM_h), Ref_RL_z_t, Foot_zsimulation_standup(COM_h);
	Temp_LL_z << Foot_zsimulation_sitdown(COM_h), Ref_LL_z_t, Foot_zsimulation_standup(COM_h);
	Ref_RL_x = Temp_RL_x;
	Ref_LL_x = Temp_LL_x;
	Ref_RL_y = Temp_RL_y;
	Ref_LL_y = Temp_LL_y;
	Ref_RL_z = Temp_RL_z;
	Ref_LL_z = Temp_LL_z;
}

void Trajectory::Side_Left1(double step)
{
	Set_step(step);
	double aaa=0.055;
	// MatrixXd Ycom = YComSimulation_Sidewalk_half(-L0, -L0, L0 + step, L0 - step);
	MatrixXd Ycom = YComSimulation_Sidewalk_half(-aaa, -aaa,  aaa + step, aaa - step);
	MatrixXd LF_yFoot = LF_ysimulation_leftwalk_halfstep();
	MatrixXd RF_yFoot = RF_ysimulation_leftwalk_halfstep();
	Ref_RL_x = MatrixXd::Zero(1, sim_n);
	Ref_LL_x = MatrixXd::Zero(1, sim_n);
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_leftwalk_halfstep();
	Ref_LL_z = LF_zsimulation_leftwalk_halfstep();
}

void Trajectory::Side_Right1(double step)
{
	Set_step(-step);
	double aaa=0.055;
	// MatrixXd Ycom = YComSimulation_Sidewalk_half(L0, L0, - L0 - step, -L0 + step);
	MatrixXd Ycom = YComSimulation_Sidewalk_half(aaa, aaa, - aaa - step, -aaa + step);
	MatrixXd LF_yFoot = LF_ysimulation_rightwalk_halfstep();
	MatrixXd RF_yFoot = RF_ysimulation_rightwalk_halfstep();
	Ref_RL_x = MatrixXd::Zero(1, sim_n);
	Ref_LL_x = MatrixXd::Zero(1, sim_n);
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_rightwalk_halfstep();
	Ref_LL_z = LF_zsimulation_rightwalk_halfstep();
}


void Trajectory::Go_Straight(double step, double distance, double height)
{
	Set_step(step);
	Set_distance(distance, 0.06);
	Xcom = XComSimulation();
	Ycom = YComSimulation();
	LF_xFoot = LF_xsimulation_straightwalk();
	RF_xFoot = RF_xsimulation_straightwalk();
	RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = RF_xFoot - Xcom;
	Ref_LL_x = LF_xFoot - Xcom;
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
}

void Trajectory::Go_Straight_start(double step, double distance, double height)
{
	Set_step(step);
	Set_distance_start(distance);
	Xcom = XComSimulation();
	Ycom = YComSimulation();
	LF_xFoot = LF_xsimulation_straightwalk();
	RF_xFoot = RF_xsimulation_straightwalk();
	RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = RF_xFoot - Xcom;
	Ref_LL_x = LF_xFoot - Xcom;
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
}


void Trajectory::Freq_Change_Straight(double step, double distance, double height, double freq)
{
	Change_Freq(freq);
	Set_step(step);
	Set_distance(distance, 0.06);

	MatrixXd Xcom_ = XComSimulation();
	MatrixXd Ycom_ = YComSimulation();
	MatrixXd LF_xFoot = LF_xsimulation_straightwalk();
	MatrixXd RF_xFoot = RF_xsimulation_straightwalk();
	MatrixXd RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	MatrixXd LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = RF_xFoot - Xcom_;
	Ref_LL_x = LF_xFoot - Xcom_;
	Ref_RL_y = RF_yFoot - Ycom_;
	Ref_LL_y = LF_yFoot - Ycom_;
	// Ref_RL_x = RF_xFoot.block(0, 0, RF_xFoot.rows(), sim_n) - Xcom.block(0, 0, RF_xFoot.rows(), sim_n);
	// Ref_LL_x = LF_xFoot.block(0, 0, LF_xFoot.rows(), sim_n) - Xcom.block(0, 0, LF_xFoot.rows(), sim_n);
	// Ref_RL_y = RF_yFoot.block(0, 0, RF_yFoot.rows(), sim_n) - Ycom.block(0, 0, RF_yFoot.rows(), sim_n);
	// Ref_LL_y = LF_yFoot.block(0, 0, LF_yFoot.rows(), sim_n) - Ycom.block(0, 0, LF_yFoot.rows(), sim_n);
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
};

void Trajectory::Stop_Trajectory_straightwalk(double step)
{
	Set_step(step);
	Set_distance(0.2, 0.06);
	MatrixXd sXcom = XComSimulation();
	MatrixXd sYcom = YComSimulation();
	MatrixXd sLF_xFoot = LF_xsimulation_straightwalk();
	MatrixXd sRF_xFoot = RF_xsimulation_straightwalk();
	MatrixXd sRF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	MatrixXd sLF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	int numCols = sRF_xFoot.cols() / 3;
	lsRef_RL_x = sRF_xFoot.block(0, numCols, 1, 2 * numCols) - sXcom.block(0, numCols, 1, 2 * numCols);
	lsRef_LL_x = sLF_xFoot.block(0, numCols, 1, 2 * numCols) - sXcom.block(0, numCols, 1, 2 * numCols);
	lsRef_RL_y = sRF_yFoot.block(0, numCols, 1, 2 * numCols) - sYcom.block(0, numCols, 1, 2 * numCols);
	lsRef_LL_y = sLF_yFoot.block(0, numCols, 1, 2 * numCols) - sYcom.block(0, numCols, 1, 2 * numCols);
	lsRef_RL_z = RF_zsimulation_straightwalk(0.05).block(0, numCols, 1, 2 * numCols);
	lsRef_LL_z = LF_zsimulation_straightwalk(0.05).block(0, numCols, 1, 2 * numCols);
	lsRef_RL_x = rsRef_RL_x.rowwise().reverse();
	lsRef_LL_x = rsRef_LL_x.rowwise().reverse();
	lsRef_RL_y = rsRef_RL_y.rowwise().reverse();
	lsRef_LL_y = rsRef_LL_y.rowwise().reverse();
	lsRef_RL_z = rsRef_RL_z.rowwise().reverse();
	lsRef_LL_z = rsRef_LL_z.rowwise().reverse();

	rsRef_RL_x = sRF_xFoot.block(0, 0, 1, numCols) - sXcom.block(0, 0, 1, numCols);
	rsRef_LL_x = sLF_xFoot.block(0, 0, 1, numCols) - sXcom.block(0, 0, 1, numCols);
	rsRef_RL_y = sRF_yFoot.block(0, 0, 1, numCols) - sYcom.block(0, 0, 1, numCols);
	rsRef_LL_y = sLF_yFoot.block(0, 0, 1, numCols) - sYcom.block(0, 0, 1, numCols);
	rsRef_RL_z = RF_zsimulation_straightwalk(0.05).block(0, 0, 1, numCols);
	rsRef_LL_z = LF_zsimulation_straightwalk(0.05).block(0, 0, 1, numCols);
}

void Trajectory::Side_Left2()
{
	Set_step(0.06);
	MatrixXd Ycom_ = YComSimulation_Sidewalk(-L0, -L0, 2 * L0, 0, 3 * L0, 2 * L0);
	MatrixXd RF_yFoot = RF_ysimulation_leftwalk();
	MatrixXd LF_yFoot = LF_ysimulation_leftwalk();
	Ref_RL_x = MatrixXd::Zero(1, sim_n);
	Ref_LL_x = MatrixXd::Zero(1, sim_n);
	Ref_RL_y = RF_yFoot - Ycom_;
	Ref_LL_y = LF_yFoot - Ycom_;
	Ref_RL_z = RF_zsimulation_leftwalk();
	Ref_LL_z = LF_zsimulation_leftwalk();
}

void Trajectory::Go_Back_Straight(double step, double distance, double height)
{
	Set_step(step);
	Set_distance_back(distance);
	Xcom = XComSimulation();
	Ycom = YComSimulation();
	LF_xFoot = LF_xsimulation_straightwalk();
	RF_xFoot = RF_xsimulation_straightwalk();
	RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = RF_xFoot - Xcom;
	Ref_LL_x = LF_xFoot - Xcom;
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
}

void Trajectory::Go_Back_Straight2(double step, double distance, double height)
{
	Set_step(step);
	Set_distance_back(distance);
	double sim_all = sim_n*2 + 100;
	Xcom = XComSimulation();
	Ycom = YComSimulation();

	MatrixXd Ref_RL_x_t = MatrixXd::Zero(1, 100);
	MatrixXd Ref_LL_x_t = MatrixXd::Zero(1, 100);
	MatrixXd Ref_RL_y_t = -L0 * MatrixXd::Ones(1, 100);
	MatrixXd Ref_LL_y_t = L0 * MatrixXd::Ones(1, 100);
	MatrixXd Ref_RL_z_t = MatrixXd::Zero(1, 100);
	MatrixXd Ref_LL_z_t = MatrixXd::Zero(1, 100);

	LF_xFoot = LF_xsimulation_straightwalk();
	RF_xFoot = RF_xsimulation_straightwalk();
	RF_yFoot = -L0 * MatrixXd::Ones(1, sim_n);
	LF_yFoot = L0 * MatrixXd::Ones(1, sim_n);
	Ref_RL_x = RF_xFoot - Xcom;
	Ref_LL_x = LF_xFoot - Xcom;
	Ref_RL_y = RF_yFoot - Ycom;
	Ref_LL_y = LF_yFoot - Ycom;
	Ref_RL_z = RF_zsimulation_straightwalk(height);
	Ref_LL_z = LF_zsimulation_straightwalk(height);
}




// // standard : body.. so, foot -> body  = sit motion 
// // For Pick and huddle 
MatrixXd Trajectory::zsimulation_sitdown_pick(int FB_sim_n, double h)
{
	RowVectorXd Footpos(FB_sim_n);
	double T = 0.6 * FB_sim_n;
	for (int i = 0; i < FB_sim_n; i++)
	{
		if (i < FB_sim_n * 0.2) Footpos[i] = 0;
		else if (i < FB_sim_n * 0.8){
			double t_rel = i - FB_sim_n*0.2;
			Footpos[i] = Sinusoidal(t_rel, T, h);
		}
		else if (i < FB_sim_n) Footpos[i] = h;
		else Footpos[i] = h;
	};
	return Footpos;
}

MatrixXd Trajectory::zsimulation_standup_pick(int FB_sim_n, double h)
{
	RowVectorXd Footpos(FB_sim_n);
	double T = 0.6 * FB_sim_n;
	for (int i = 0; i < FB_sim_n; i++)
	{
		if (i < FB_sim_n * 0.2) Footpos[i] = h;
		else if (i < FB_sim_n * 0.8) {
			double t_rel = i - FB_sim_n*0.2;
			Footpos[i] = h - Sinusoidal(t_rel, T, h);
		}
		else if (i < FB_sim_n) Footpos[i] = 0;
		else Footpos[i] = 0;
	};
	return Footpos;
}

MatrixXd Trajectory::z_position_sitdown(int M_sim_n, double h)
{
	RowVectorXd Footpos(M_sim_n);
	for (int i = 0; i < M_sim_n; i++)
	{
		Footpos[i] = h;
	};
	return Footpos;
}

void Trajectory::Picking_Motion(int FB_sim_n, int M_sim_n, double COM_h)
{
	MatrixXd Ref_RL_x_t = MatrixXd::Zero(1, M_sim_n);
	MatrixXd Ref_LL_x_t = MatrixXd::Zero(1, M_sim_n);
	MatrixXd Ref_RL_y_t = -L0 * MatrixXd::Ones(1, M_sim_n);
	MatrixXd Ref_LL_y_t = L0 * MatrixXd::Ones(1, M_sim_n);
	MatrixXd Ref_RL_z_t = z_position_sitdown(M_sim_n, COM_h);
	MatrixXd Ref_LL_z_t = z_position_sitdown(M_sim_n, COM_h);
	MatrixXd Temp_RL_x(Ref_RL_x_t.rows(), MatrixXd::Zero(1, FB_sim_n).cols() + Ref_RL_x_t.cols() + MatrixXd::Zero(1, FB_sim_n).cols());
	MatrixXd Temp_LL_x(Ref_LL_x_t.rows(), MatrixXd::Zero(1, FB_sim_n).cols() + Ref_LL_x_t.cols() + MatrixXd::Zero(1, FB_sim_n).cols());
	MatrixXd Temp_RL_y(Ref_RL_y_t.rows(), MatrixXd::Ones(1, FB_sim_n).cols() + Ref_RL_y_t.cols() + MatrixXd::Ones(1, FB_sim_n).cols());
	MatrixXd Temp_LL_y(Ref_LL_y_t.rows(), MatrixXd::Ones(1, FB_sim_n).cols() + Ref_LL_y_t.cols() + MatrixXd::Ones(1, FB_sim_n).cols());
	MatrixXd Temp_RL_z(Ref_RL_z_t.rows(), zsimulation_sitdown_pick(FB_sim_n, COM_h).cols() + Ref_RL_z_t.cols() + zsimulation_standup_pick(FB_sim_n, COM_h).cols());
	MatrixXd Temp_LL_z(Ref_LL_z_t.rows(), zsimulation_sitdown_pick(FB_sim_n, COM_h).cols() + Ref_LL_z_t.cols() + zsimulation_standup_pick(FB_sim_n, COM_h).cols());
	Temp_RL_x << MatrixXd::Zero(1, FB_sim_n), Ref_RL_x_t, MatrixXd::Zero(1, FB_sim_n);
	Temp_LL_x << MatrixXd::Zero(1, FB_sim_n), Ref_LL_x_t, MatrixXd::Zero(1, FB_sim_n);
	Temp_RL_y << -L0 * MatrixXd::Ones(1, FB_sim_n), Ref_RL_y_t, -L0 * MatrixXd::Ones(1, FB_sim_n);
	Temp_LL_y << L0 * MatrixXd::Ones(1, FB_sim_n), Ref_LL_y_t, L0 * MatrixXd::Ones(1, FB_sim_n);
	Temp_RL_z << zsimulation_sitdown_pick(FB_sim_n, COM_h), Ref_RL_z_t, zsimulation_standup_pick(FB_sim_n, COM_h);
	Temp_LL_z << zsimulation_sitdown_pick(FB_sim_n, COM_h), Ref_LL_z_t, zsimulation_standup_pick(FB_sim_n, COM_h);
	Ref_RL_x = Temp_RL_x;
	Ref_LL_x = Temp_LL_x;
	Ref_RL_y = Temp_RL_y;
	Ref_LL_y = Temp_LL_y;
	Ref_RL_z = Temp_RL_z;
	Ref_LL_z = Temp_LL_z;

}



//////////////


/////////shoot


MatrixXd Trajectory::zsimulation_standup_Shoot_FINISH(int FB_sim_n, double h)
{
	XStep = Equation_solver(0, FB_sim_n * 0.6, h, 0);
	RowVectorXd Footpos(FB_sim_n);
	for (int i = 0; i < FB_sim_n; i++)
	{
		if (i < FB_sim_n * 0.2)
		{
			Footpos[i] = h;
		}
		else if (i < FB_sim_n * 0.8)
		{
			Footpos[i] = Step(i - FB_sim_n * 0.2);
		}
		else if (i < FB_sim_n)
		{
			Footpos[i] = 0;
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
}






MatrixXd Trajectory::RF_xsimulation_straightwalk(){
	int sim_n = step_n * walktime_n;
	double dwalktime_n = walktime_n;
	double Rfootflag = 0;

	// double Trajectory::Sinusoidal(double t, double T, double h) {return h * 0.5 * (1 - cos(PI * t / T));}

	RowVectorXd Footpos(sim_n);
	Footpos.setZero();
	
	for (int i = 0; i < sim_n; i++)
	{
		
		if (i < 0.6 * dwalktime_n) Footpos[i] = 0;
		
		else if (i < 0.9 * dwalktime_n) {
			double t_rel = i - 0.6 * walktime_n;
			Footpos[i] = Sinusoidal(t_rel, walktime_n*0.3, step);
		}
		else if (i < 1.1 * dwalktime_n) Footpos[i] = step;
		else if (i < (Rfootflag + 0.6) * dwalktime_n) Footpos[i] = (2 * Rfootflag - 1) * step;
		else if (i < (Rfootflag + 0.9) * dwalktime_n) {
			double t_rel = i - (0.6 + Rfootflag) * walktime_n;
			Footpos[i] = Sinusoidal(t_rel, walktime_n *0.3, 2*step) + (2 * Rfootflag - 1) * step;
		}
		
		else if (i < (Rfootflag + 1.1) * dwalktime_n) Footpos[i] = (2 * Rfootflag + 1) * step;
		else Footpos[i] = (2 * step_n - 3) * step;

		if ((i + 1) % walktime_n == 0)
		{
			if (Rfootflag < step_n - 2) Rfootflag = Rfootflag + 1;
		}
	};
	return Footpos;
};

MatrixXd Trajectory::LF_xsimulation_straightwalk()
{
	int sim_n = step_n * walktime_n;
	double dwalktime_n = walktime_n;
	double Lfootflag = 0;

	// double Trajectory::Sinusoidal(double t, double T, double h) {return h * 0.5 * (1 - cos(PI * t / T));}

	RowVectorXd Footpos(sim_n);
	Footpos.setZero();

	for (int i = 0; i < sim_n; i++)
	{
		if (i < 1.1 * walktime_n) Footpos[i] = 0;
		
		else if (i < (Lfootflag + 0.1) * walktime_n) Footpos[i] = (2 * Lfootflag - 2) * step;
		
		else if (i < (Lfootflag + 0.4) * walktime_n){
			double t_rel = i - (Lfootflag + 0.1) * dwalktime_n;
			Footpos[i] = Sinusoidal(t_rel, walktime_n * 0.3, 2* step) + (2 * Lfootflag - 2) * step;
		}

		else if (i < (Lfootflag + 1.1) * walktime_n) Footpos[i] = (2 * Lfootflag) * step;

		else if (i < (step_n - 0.6) * walktime_n) {
			double t_rel = i - (Lfootflag + 1.1) * dwalktime_n;
			Footpos[i] = Sinusoidal(t_rel, walktime_n * 0.3, step) + (2 * step_n - 4) * step;
		}

		else Footpos[i] = (2 * step_n - 3) * step;

		if ((i + 1) % walktime_n == 0)
		{
			if (Lfootflag < step_n - 2) Lfootflag = Lfootflag + 1;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::RF_zsimulation_straightwalk(double h) 
{
    int sim_n = step_n * walktime_n;
    double dwalktime_n = walktime_n;
    double Rfootflag = 0;

    RowVectorXd Footpos(sim_n);
    Footpos.setZero();

    for (int i = 0; i < sim_n; i++) {
        double rise_start = (Rfootflag + 0.55) * dwalktime_n;
        double rise_end   = (Rfootflag + 0.75) * dwalktime_n;
        double fall_end   = (Rfootflag + 0.95) * dwalktime_n;

        double T = 0.2 * dwalktime_n;

        if (i < rise_start) Footpos[i] = 0;

        else if (i >= rise_start && i < rise_end) {
            double phase_time = i - rise_start;
            Footpos[i] = Sinusoidal(phase_time, T, h);  // 올림
        }
        else if (i >= rise_end && i < fall_end) {
            double phase_time = fall_end - i;
            Footpos[i] = Sinusoidal(phase_time, T, h);  // 내림
        }
        else if (i  < (Rfootflag + 1) * dwalktime_n) Footpos[i] = 0;
        else Footpos[i] = 0;

        if ((i + 1) % walktime_n == 0) {
            if (Rfootflag < step_n - 2) Rfootflag += 1;
        }
    }

    return Footpos;
}

MatrixXd Trajectory::LF_zsimulation_straightwalk(double h) {
    int sim_n = step_n * walktime_n;
    double dwalktime_n = walktime_n;
    double Lfootflag = 0;

    RowVectorXd Footpos(sim_n);
    Footpos.setZero();

    for (int i = 0; i < sim_n; i++) {
        double rise_start = (Lfootflag + 0.05) * dwalktime_n;
        double rise_end   = (Lfootflag + 0.25) * dwalktime_n;
        double fall_end   = (Lfootflag + 0.45) * dwalktime_n;

        double T = 0.2 * dwalktime_n;  // cos traj. 

        if (i < 1.05 * dwalktime_n) Footpos[i] = 0;

        else if (i >= rise_start && i < rise_end) {
            double phase_time = i - rise_start;
            Footpos[i] = Sinusoidal(phase_time, T, h);  // 올림
        }
        else if (i >= rise_end && i < fall_end) {
            double phase_time = fall_end - i;
            Footpos[i] = Sinusoidal(phase_time, T, h);  // 내림
        }
        else if (i < (Lfootflag + 0.5) * dwalktime_n) Footpos[i] = 0;

        else Footpos[i] = 0;

        if ((i + 1) % walktime_n == 0) {
            if (Lfootflag < step_n - 1) Lfootflag += 1;
        }
    }

    return Footpos;
}






MatrixXd Trajectory::YComSimulation_Sidewalk(double a, double b, double c, double d, double e, double f)
{
	step_n = 5;
	sim_n = walktime_n * step_n;
	PreviewGd();
	RowVectorXd zmp_ref(sim_n + 19);
	for (int i = 0; i < sim_n + 19; i++)
	{
		double time = i * del_t;
		if (time < 1 * walktime)
		{
			zmp_ref[i] = 0;
		}
		else if (time < 1.5 * walktime)
		{
			zmp_ref[i] = a;
		}
		else if (time < 2 * walktime)
		{
			zmp_ref[i] = b;
		}
		else if (time < 2.5 * walktime)
		{
			zmp_ref[i] = c;
		}
		else if (time < 3 * walktime)
		{
			zmp_ref[i] = d;
		}
		else if (time < 3.5 * walktime)
		{
			zmp_ref[i] = e;
		}
		else if (time < 4 * walktime)
		{
			zmp_ref[i] = f;
		}
		else
			zmp_ref[i] = f;
	}
	VectorXd zmp_ref_fifo(NL);
	VectorXd u(sim_n + 19);
	VectorXd zmp(sim_n + 19);
	VectorXd zmp_ref_final(sim_n + 19);
	VectorXd CP(sim_n + 19);
	MatrixXd YCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	YCom.setZero();
	double w = sqrt(g / z_c);
	for (int i = 0; i < sim_n + 19; i++)
	{
		for (int j = 0; j < NL; j++)
		{
			if (i + j < sim_n + 19)
			{
				zmp_ref_fifo[j] = zmp_ref[i + j];
			}
			else
			{
				zmp_ref_fifo[j] = zmp_ref[sim_n + 18];
			}
		}
		u_prev = 0;
		for (int j = 0; j < NL; j++)
		{
			u_prev = u_prev + Gd(j, 0) * zmp_ref_fifo[j];
		}
		u[i] = Gi * zmp_err_int - Gx * YCom.col(i) + u_prev;

		YCom.col(i + 1) = A * YCom.col(i) + B * u[i];

		zmp[i] = C * YCom.col(i);

		zmp_err_int += (zmp_ref[i] - zmp[i]);

		CP[i] = YCom(0, i) + 1.0 / w * YCom(1, i);

		zmp_ref_final[i] = zmp_ref[i];
	}
	MatrixXd YCom_ = YCom.block(0, 18, 1, sim_n);
	return YCom_;
}

MatrixXd Trajectory::RF_ysimulation_leftwalk()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 5;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n); // rightfoot motion
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;
		if (time < 1.1 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 1.4 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 2.1 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 2.4 * walktime)
		{
			Footpos[i] = Step(time - 2.1 * walktime) - 0.06;
		}
		else if (time < 3.1 * walktime)
		{
			Footpos[i] = -0.06 + step;
		}
		else if (time < 3.4 * walktime)
		{
			Footpos[i] = Step(time - 3.1 * walktime) - 0.06 + step;
		}
		else
		{
			Footpos[i] = -0.06 + 2 * step;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::LF_ysimulation_leftwalk()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 5;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.6 * walktime)
		{
			Footpos[i] = 0.06;
		}
		else if (time < 1.9 * walktime)
		{
			Footpos[i] = Step(time - 1.6 * walktime) + 0.06;
		}
		else if (time < 2.6 * walktime)
		{
			Footpos[i] = 0.06 + step;
		}
		else if (time < 2.9 * walktime)
		{
			Footpos[i] = Step(time - 2.6 * walktime) + 0.06 + step;
		}
		else if (time < 3.6 * walktime)
		{
			Footpos[i] = 0.06 + 2 * step;
		}
		else if (time < 3.9 * walktime)
		{
			Footpos[i] = 0.06 + 2 * step;
		}
		else
		{
			Footpos[i] = 0.06 + 2 * step;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::RF_zsimulation_leftwalk()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 5;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.25 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.45 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.25 * walktime)
		{
			Footpos[i] = Step(time - 2.05 * walktime);
		}
		else if (time < 2.45 * walktime)
		{
			Footpos[i] = Stride(time - 2.05 * walktime);
		}
		else if (time < 3.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 3.25 * walktime)
		{
			Footpos[i] = Step(time - 3.05 * walktime);
		}
		else if (time < 3.45 * walktime)
		{
			Footpos[i] = Stride(time - 3.05 * walktime);
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
};

MatrixXd Trajectory::LF_zsimulation_leftwalk()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 5;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.75 * walktime)
		{
			Footpos[i] = Step(time - 1.55 * walktime);
		}
		else if (time < 1.95 * walktime)
		{
			Footpos[i] = Stride(time - 1.55 * walktime);
		}
		else if (time < 2.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.75 * walktime)
		{
			Footpos[i] = Step(time - 2.55 * walktime);
		}
		else if (time < 2.95 * walktime)
		{
			Footpos[i] = Stride(time - 2.55 * walktime);
		}
		else if (time < 3.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 3.75 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 3.95 * walktime)
		{
			Footpos[i] = 0;
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
};

////////////////turn//////////////////////
void Trajectory::Make_turn_trajectory(double angle)
{
	int st;
	Angle_trajectorty_turn = Equation_solver(0, walktime_n * 0.3, 0, angle);
	Angle_trajectorty_back = Equation_solver(0, walktime_n * 0.3, angle, 0);
	Turn_Trajectory = VectorXd::Zero(walktime_n);
	for (int i = 0; i < walktime_n; i++)
	{
		if (i < 0.1 * walktime_n)
		{
			Turn_Trajectory(i) = 0;
		}
		else if (i < 0.4 * walktime_n)
		{
			Turn_Trajectory(i) = Return_turn_trajectory(i - 0.1 * walktime_n);
		}
		else if (i < 0.6 * walktime_n)
		{
			Turn_Trajectory(i) = angle;
		}
		else if (i < 0.9 * walktime_n)
		{
			Turn_Trajectory(i) = Return_back_trajectory(i - 0.6 * walktime_n);
		}
		else
			Turn_Trajectory(i) = 0;
	}
	
	// for (int i = 0; i < walktime_n; ++i) {
    // double deg = Turn_Trajectory(i) * 180.0 / M_PI; // 라디안→도
    // std::printf("[TurnTraj] i=%4d  %.3f deg\n", i, deg);
    // // 또는 RCLCPP_INFO(this->get_logger(), "[TurnTraj] i=%d %.3f deg", i, deg);
	// }

}

double Trajectory::Return_turn_trajectory(double t)
{
	double X = Angle_trajectorty_turn(0) + Angle_trajectorty_turn(1) * t + Angle_trajectorty_turn(2) * pow(t, 2) + Angle_trajectorty_turn(3) * pow(t, 3) + Angle_trajectorty_turn(4) * pow(t, 4) + Angle_trajectorty_turn(5) * pow(t, 5);
	return X;
}

double Trajectory::Return_back_trajectory(double t)
{
	double X = Angle_trajectorty_back(0) + Angle_trajectorty_back(1) * t + Angle_trajectorty_back(2) * pow(t, 2) + Angle_trajectorty_back(3) * pow(t, 3) + Angle_trajectorty_back(4) * pow(t, 4) + Angle_trajectorty_back(5) * pow(t, 5);
	return X;
}



////////////////Huddle this this this this this this this////////////////////////

MatrixXd Trajectory::RF_xsimulation_huddle()
{
    XStep = Equation_solver(0, walktime, 0, 1.2 * step);
    step_n = 6;
    sim_n = walktime_n * step_n;
    double del_t = 1 / freq;
    RowVectorXd Footpos(sim_n); // rightfoot motion
    for (int i = 0; i < sim_n; i++)
    {
        double time = i * del_t;
        if (time < 2.5 * walktime) Footpos[i] = 0;
        else if (time < 3.5 * walktime) Footpos[i] = Step(time - 2.5 * walktime);
        else Footpos[i] = 1.2 * step;
    };
    return Footpos;
};

MatrixXd Trajectory::RF_zsimulation_huddle(double h, double COM_h)
{
    XStep = Equation_solver(0, 0.2 * walktime, 0, h);
    XStride = Equation_solver(0, 0.2 * walktime, h, 0);
    step_n = 6;
    sim_n = walktime_n * step_n;
    double del_t = 1 / freq;
    RowVectorXd Footpos(sim_n);
    for (int i = 0; i < sim_n; i++)
    {
        double time = i * del_t;

        if (time < 0.8 * walktime) Footpos[i] = 0.5*COM_h;
        else if (time < 2.3 * walktime) Footpos[i] = 0.5*COM_h;
        else if (time < 2.7 * walktime) Footpos[i] = 0.75*Step(time - 2.3 * walktime) + 0.5*COM_h;
        else if (time < 3.5 * walktime) Footpos[i] = 0.75*h + 0.5*COM_h;
        else if (time < 3.9 * walktime) Footpos[i] = 0.75*Stride(time - 3.5 * walktime) + 0.5*COM_h;
        else Footpos[i] = 0.5*COM_h;
    };
    return Footpos;
};

MatrixXd Trajectory::LF_xsimulation_huddle()
{
    XStep = Equation_solver(0, walktime * 0.5, 0, step);
    XStride = Equation_solver(0, walktime * 0.3, 0, 0.2 * step);
    step_n = 6;
    sim_n = walktime_n * step_n;
    double del_t = 1 / freq;
    RowVectorXd Footpos(sim_n); // leftfoot motion
    for (int i = 0; i < sim_n; i++)
    {
        double time = i * del_t;
        if (time < 1 * walktime) Footpos[i] = 0;
        else if (time < 1.5 * walktime) Footpos[i] = Step(time - 1 * walktime);
        else if (time < 5.2 * walktime) Footpos[i] = step;
        else if (time < 5.5 * walktime) Footpos[i] = Stride(time - 5.2 * walktime) + step;
        else Footpos[i] = 1.2 * step;
    };
    return Footpos;
};

MatrixXd Trajectory::LF_zsimulation_huddle(double h, double COM_h)
{
    XStep = Equation_solver(0, 0.2 * walktime, 0, h);
    XStride = Equation_solver(0, 0.2 * walktime, h, 0);
    step_n = 6;
    sim_n = walktime_n * step_n;
    double del_t = 1 / freq;
    RowVectorXd Footpos(sim_n);
    for (int i = 0; i < sim_n; i++)
    {
        double time = i * del_t;

        if (time < 0.8 * walktime) Footpos[i] = 0.5*COM_h;
        else if (time < 1 * walktime) Footpos[i] = 0.75*Step(time - 0.8 * walktime) + 0.5*COM_h;
        else if (time < 1.5 * walktime) Footpos[i] = 0.75*h + 0.5*COM_h;
        else if (time < 1.7 * walktime) Footpos[i] = 0.75*Stride(time - 1.5 * walktime) + 0.5*COM_h;
        else if (time < 5.0 * walktime) Footpos[i] = 0.5*COM_h;
        else if (time < 5.2 * walktime) Footpos[i] = 0.2 * Step(time - 5 * walktime) + 0.5*COM_h;
        else if (time < 5.5 * walktime) Footpos[i] = 0.2 * h + 0.5*COM_h;
        else if (time < 5.7 * walktime) Footpos[i] = 0.2 * Stride(time - 5.5 * walktime) + 0.5*COM_h;
        else Footpos[i] = 0.5*COM_h;
    };
    return Footpos;
};

MatrixXd Trajectory::Huddle_Xcom()
{
	sim_n = walktime_n * 6;
	zmp_err_int = 0;
	u_prev = 0;
	PreviewGd();
	int Com_flag = 0;
	float float_Com_flag = 0;
	float float_walktime_n = walktime_n;
	RowVectorXd zmp_ref(sim_n + 19);
	for (int i = 0; i < sim_n + 19; i++)
	{
		if (i < 1.75 * float_walktime_n) zmp_ref[i] = 0;
		else if (i < 4.5 * float_walktime_n) zmp_ref[i] = step;
		else zmp_ref[i] = 1.2 * step;
	}

	RowVectorXd zmp_ref_fifo(NL);
	RowVectorXd u(sim_n + 19);
	RowVectorXd zmp(sim_n + 19);
	RowVectorXd zmp_ref_final(sim_n + 19);
	RowVectorXd CP(sim_n + 19);
	MatrixXd XCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	XCom.setZero();
	double w = sqrt(g / z_c - 1.2 * 0.05);
	for (int i = 0; i < sim_n + 19; i++)
	{
		for (int j = 0; j < NL; j++) zmp_ref_fifo[j] = (i + j < sim_n + 19) ? zmp_ref[i + j] : zmp_ref[sim_n + 18];
		u_prev = 0;
		for (int j = 0; j < NL; j++) u_prev += Gd(j, 0) * zmp_ref_fifo[j];

		u[i] = Gi * zmp_err_int - Gx * XCom.col(i) + u_prev;
		XCom.col(i + 1) = A * XCom.col(i) + B * u[i];
		zmp[i] = C * XCom.col(i);
		zmp_err_int += (zmp_ref[i] - zmp[i]);
		CP[i] = XCom(0, i) + 1 / w * XCom(1, i);
		zmp_ref_final[i] = zmp_ref[i];
	}
	MatrixXd XCom_ = XCom.block(0, 18, 1, sim_n);
	return XCom_;
}


MatrixXd Trajectory::Huddle_Ycom()
{
	sim_n = walktime_n * 6;
	PreviewGd();
	zmp_err_int = 0;
	u_prev = 0;
	int Com_flag = 0;
	float float_Com_flag = 0;
	float float_walktime_n = walktime_n;
	RowVectorXd zmp_ref(sim_n + 19);
	for (int i = 0; i < sim_n + 19; i++)
	{
		if (i < 0.75 * float_walktime_n) zmp_ref[i] = 0;
		else if (i < 1.75 * float_walktime_n) zmp_ref[i] = -0.05;
		else if (i < 4.5 * float_walktime_n) zmp_ref[i] = 0.056;
		else if (i < 5.7 * float_walktime_n) zmp_ref[i] = -0.053;
		else zmp_ref[i] = 0.0;
	}

	VectorXd zmp_ref_fifo(NL);
	VectorXd u(sim_n + 19);
	VectorXd zmp(sim_n + 19);
	VectorXd zmp_ref_final(sim_n + 19);
	VectorXd CP(sim_n + 19);
	MatrixXd YCom(3, sim_n + 20);
	zmp_ref_fifo.setZero();
	u.setZero();
	zmp.setZero();
	zmp_ref_final.setZero();
	CP.setZero();
	YCom.setZero();
	double w = sqrt(g / z_c - 1.2 * 0.05);
	for (int i = 0; i < sim_n + 19; i++)
	{
		for (int j = 0; j < NL; j++) zmp_ref_fifo[j] = (i + j < sim_n + 19) ? zmp_ref[i + j] : zmp_ref[sim_n + 18];
		u_prev = 0;
		for (int j = 0; j < NL; j++) u_prev = u_prev + Gd(j, 0) * zmp_ref_fifo[j];
		
		u[i] = Gi * zmp_err_int - Gx * YCom.col(i) + u_prev;
		YCom.col(i + 1) = A * YCom.col(i) + B * u[i];
		zmp[i] = C * YCom.col(i);
		zmp_err_int += (zmp_ref[i] - zmp[i]);
		CP[i] = YCom(0, i) + 1.0 / w * YCom(1, i);
		zmp_ref_final[i] = zmp_ref[i];
	}
	MatrixXd YCom_ = YCom.block(0, 18, 1, sim_n);
	return YCom_;
}



/////////////// huddle the end the end the end /////


// MatrixXd Trajectory::Foot_zsimulation_sitdown(double COM_h)
// {
// 	XStep = Equation_solver(0, 60, 0, 0.5*COM_h);
// 	RowVectorXd Footpos(100);
// 	for (int i = 0; i < 100; i++)
// 	{
// 		if (i < 20) Footpos[i] = 0;
// 		else if (i < 80) Footpos[i] = Step(i - 20);
// 		else if (i < 100) Footpos[i] = 0.5*COM_h;
// 		else Footpos[i] = 0.5*COM_h;
// 	};
// 	return Footpos;
// }

// MatrixXd Trajectory::Foot_zsimulation_standup(double COM_h)
// {
// 	XStep = Equation_solver(0, 60, 0.5*COM_h, 0);
// 	RowVectorXd Footpos(100);
// 	for (int i = 0; i < 100; i++)
// 	{
// 		if (i < 20) Footpos[i] = 0.5*COM_h;
// 		else if (i < 80) Footpos[i] = Step(i - 20);
// 		else if (i < 100) Footpos[i] = 0;
// 		else Footpos[i] = 0;
// 	};
// 	return Footpos;
// }



MatrixXd Trajectory::Foot_zsimulation_sitdown(double COM_h)
{
	RowVectorXd Footpos(100);
	double T = 60.0;
	for (int i = 0; i < 100; i++)
	{
		if (i < 20) Footpos[i] = 0;
		else if (i < 80) {
			double t_rel = i -20;
			Footpos[i] = 0.5*Sinusoidal(t_rel, T, COM_h);
		}
		else Footpos[i] = 0.5*COM_h;
	};
	return Footpos;
}

MatrixXd Trajectory::Foot_zsimulation_standup(double COM_h)
{
	RowVectorXd Footpos(100);
	double T = 60.0;
	for (int i = 0; i < 100; i++)
	{
		if (i < 20) Footpos[i] = 0.5*COM_h;
		else if (i < 80) {
			double t_rel = i -20;
			Footpos[i] = 0.5*COM_h - 0.5*Sinusoidal(t_rel, T, COM_h);
		}
		else if (i < 100) Footpos[i] =0;
		else Footpos[i] = 0;
	};
	return Footpos;
}




///////////////////////halfstep/////////////////////

MatrixXd Trajectory::RF_ysimulation_leftwalk_halfstep()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n); // rightfoot motion
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;
		if (time < 1.1 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 1.4 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 2.1 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 2.4 * walktime)
		{
			Footpos[i] = Step(time - 2.1 * walktime) - 0.06;
		}

		else
		{
			Footpos[i] = step - 0.06;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::LF_ysimulation_leftwalk_halfstep()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.6 * walktime)
		{
			Footpos[i] = 0.06;
		}
		else if (time < 1.9 * walktime)
		{
			Footpos[i] = Step(time - 1.6 * walktime) + 0.06;
		}
		else
		{
			Footpos[i] = step + 0.06;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::RF_zsimulation_leftwalk_halfstep()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.25 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.45 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.25 * walktime)
		{
			Footpos[i] = Step(time - 2.05 * walktime);
		}
		else if (time < 2.45 * walktime)
		{
			Footpos[i] = Stride(time - 2.05 * walktime);
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::LF_zsimulation_leftwalk_halfstep()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.75 * walktime)
		{
			Footpos[i] = Step(time - 1.55 * walktime);
		}
		else if (time < 1.95 * walktime)
		{
			Footpos[i] = Stride(time - 1.55 * walktime);
		}
		else if (time < 2.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::RF_ysimulation_rightwalk_halfstep()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.6 * walktime)
		{
			Footpos[i] = -0.06;
		}
		else if (time < 1.9 * walktime)
		{
			Footpos[i] = Step(time - 1.6 * walktime) - 0.06;
		}
		else
		{
			Footpos[i] = -0.06 + step;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::LF_ysimulation_rightwalk_halfstep()
{
	XStep = Equation_solver(0, walktime * 0.3, 0, step);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n); // rightfoot motion
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;
		if (time < 1.1 * walktime)
		{
			Footpos[i] = 0.06;
		}
		else if (time < 1.4 * walktime)
		{
			Footpos[i] = 0.06;
		}
		else if (time < 2.1 * walktime)
		{
			Footpos[i] = 0.06;
		}
		else if (time < 2.4 * walktime)
		{
			Footpos[i] = Step(time - 2.1 * walktime) + 0.06;
		}
		else
		{
			Footpos[i] = 0.06 + step;
		}
	};
	return Footpos;
}

MatrixXd Trajectory::RF_zsimulation_rightwalk_halfstep()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.75 * walktime)
		{
			Footpos[i] = Step(time - 1.55 * walktime);
		}
		else if (time < 1.95 * walktime)
		{
			Footpos[i] = Stride(time - 1.55 * walktime);
		}
		else if (time < 2.55 * walktime)
		{
			Footpos[i] = 0;
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
};

MatrixXd Trajectory::LF_zsimulation_rightwalk_halfstep()
{
	XStep = Equation_solver(0, 0.2 * walktime, 0, 0.05);
	XStride = Equation_solver(0.2 * walktime, 0.4 * walktime, 0.05, 0);
	step_n = 3;
	sim_n = walktime_n * step_n;
	double del_t = 1 / freq;
	RowVectorXd Footpos(sim_n);
	for (int i = 0; i < sim_n; i++)
	{
		double time = i * del_t;

		if (time < 1.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.25 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 1.45 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.05 * walktime)
		{
			Footpos[i] = 0;
		}
		else if (time < 2.25 * walktime)
		{
			Footpos[i] = Step(time - 2.05 * walktime);
		}
		else if (time < 2.45 * walktime)
		{
			Footpos[i] = Stride(time - 2.05 * walktime);
		}
		else
		{
			Footpos[i] = 0;
		}
	};
	return Footpos;
};













IK_Function::IK_Function()
{
	walkfreq = 1.48114;
	walktime = 2 / walkfreq;
	freq = 100;
	walktime_n = walktime * freq;
	step = 0.05;
	freq = 100;
	step_n = 0;
	sim_n = 0;
	sim_time = 0;

	RL_th[0] = 0. * deg2rad;   // RHY
	RL_th[1] = 0. * deg2rad;   // RHR
	RL_th[2] = -35. * deg2rad; // RHP
	RL_th[3] = 70. * deg2rad;  // RKN
	RL_th[4] = -35. * deg2rad; // RAP
	RL_th[5] = 0. * deg2rad;   // RAR

	LL_th[0] = 0. * deg2rad;   // LHY
	LL_th[1] = 0. * deg2rad;   // LHR
	LL_th[2] = -35. * deg2rad; // LHP
	LL_th[3] = 70. * deg2rad;  // LKN
	LL_th[4] = -35. * deg2rad; // LAP
	LL_th[5] = 0. * deg2rad;   // LAR

	Ref_RL_PR[0] = 40.;
	Ref_RL_PR[1] = -L0;
	Ref_RL_PR[2] = -L1 - L2 - L3 - L4 - L5 - L6 + 40.;
	Ref_RL_PR[3] = 0 * deg2rad;
	Ref_RL_PR[4] = 0 * deg2rad;
	Ref_RL_PR[5] = 0 * deg2rad;

	Ref_LL_PR[0] = 40.;
	Ref_LL_PR[1] = L0;
	Ref_LL_PR[2] = -L1 - L2 - L3 - L4 - L5 - L6 + 40.;
	Ref_LL_PR[3] = 0 * deg2rad;
	Ref_LL_PR[4] = 0 * deg2rad;
	Ref_LL_PR[5] = 0 * deg2rad;

	RL_Support_Leg = 0;
	RL_Swing_Leg = 0;
	RL_Support_Knee = 0;
	LL_Support_Leg = 0;
	LL_Swing_Leg = 0;
	LL_Support_Knee = 0;
	Com_Height = 300;
}

void IK_Function::Get_Step_n(double a)
{
	step_n = a;
	sim_n = walktime_n * step_n;
}

void IK_Function::BRP_Simulation(const MatrixXd& RFx, const MatrixXd& RFy, const MatrixXd& RFz, 
                                 const MatrixXd& LFx, const MatrixXd& LFy, const MatrixXd& LFz, 
                                 int Index_CNT)
{
    // Index_CNT가 유효한 범위 내에 있는지 확인하고 조정
    int idx = std::clamp(Index_CNT, 0, static_cast<int>(RFx.cols()) - 1);

	// Ref_RL_PR와 Ref_LL_PR 배열 초기화
    Ref_RL_PR[0] = 1000 * RFx(0, idx);
    Ref_RL_PR[1] = 1000 * RFy(0, idx);
    Ref_RL_PR[2] = -L1 - L2 - L3 - L4 - L5 - L6 + Com_Height + 1000 * RFz(0, idx);
    Ref_RL_PR[3] = 0 * deg2rad;
    Ref_RL_PR[4] = 0 * deg2rad;
    Ref_RL_PR[5] = 0 * deg2rad;

    Ref_LL_PR[0] = 1000 * LFx(0, idx);
    Ref_LL_PR[1] = 1000 * LFy(0, idx);
    Ref_LL_PR[2] = -L1 - L2 - L3 - L4 - L5 - L6 + Com_Height + 1000 * LFz(0, idx);
    Ref_LL_PR[3] = 0 * deg2rad;
    Ref_LL_PR[4] = 0 * deg2rad;
    Ref_LL_PR[5] = 0 * deg2rad;

	Eigen::Map<Eigen::VectorXd> RL_th_vec(RL_th, 6);
    Eigen::Map<Eigen::VectorXd> LL_th_vec(LL_th, 6);
    Eigen::Map<Eigen::VectorXd> Ref_RL_PR_vec(Ref_RL_PR, 6);
    Eigen::Map<Eigen::VectorXd> Ref_LL_PR_vec(Ref_LL_PR, 6);

	Eigen::VectorXd link(7);
	link << L0, L1, L2, L3, L4, L5, L6;
 	
    Eigen::VectorXd RL_th_IK_vec(6), LL_th_IK_vec(6);

	BRP_Kinematics::BRP_RL_IK(Ref_RL_PR_vec, RL_th_vec, link, RL_th_IK_vec);
    BRP_Kinematics::BRP_LL_IK(Ref_LL_PR_vec, LL_th_vec, link, LL_th_IK_vec);

	std::memcpy(RL_th, RL_th_IK_vec.data(), 6 * sizeof(double));
    std::memcpy(LL_th, LL_th_IK_vec.data(), 6 * sizeof(double));

}


void IK_Function::Change_Com_Height(double h)
{
	Com_Height = h;
}

void IK_Function::Set_Angle_Compensation(int walktime_n)
{
	Trajectory traj;

	RL_Compensation_Support_Leg_up = traj.Equation_solver(0, walktime_n * 0.075, 0, RL_Support_Leg);
	RL_Compensation_Support_Leg_down = traj.Equation_solver(0, walktime_n * 0.075, RL_Support_Leg, 0);
	RL_Compensation_Swing_Leg_up = traj.Equation_solver(0, walktime_n * 0.075, 0, RL_Swing_Leg);
	RL_Compensation_Swing_Leg_down = traj.Equation_solver(0, walktime_n * 0.075, RL_Swing_Leg, 0);
	RL_Compensation_Support_knee_up = traj.Equation_solver(0, walktime_n * 0.075, 0, RL_Support_Knee);
	RL_Compensation_Support_knee_down = traj.Equation_solver(0, walktime_n * 0.075, RL_Support_Knee, 0);
	RL_Compensation_Support_ankle_up = traj.Equation_solver(0, walktime_n * 0.075, 0, RL_Support_Ankle);
	RL_Compensation_Support_ankle_down = traj.Equation_solver(0, walktime_n * 0.075, RL_Support_Ankle, 0);

	LL_Compensation_Support_Leg_up = traj.Equation_solver(0, walktime_n * 0.075, 0, LL_Support_Leg);
	LL_Compensation_Support_Leg_down = traj.Equation_solver(0, walktime_n * 0.075, LL_Support_Leg, 0);
	LL_Compensation_Swing_Leg_up = traj.Equation_solver(0, walktime_n * 0.075, 0, LL_Swing_Leg);
	LL_Compensation_Swing_Leg_down = traj.Equation_solver(0, walktime_n * 0.075, LL_Swing_Leg, 0);
	LL_Compensation_Support_knee_up = traj.Equation_solver(0, walktime_n * 0.075, 0, LL_Support_Knee);
	LL_Compensation_Support_knee_down = traj.Equation_solver(0, walktime_n * 0.075, LL_Support_Knee, 0);
	LL_Compensation_Support_ankle_up = traj.Equation_solver(0, walktime_n * 0.075, 0, LL_Support_Ankle);
	LL_Compensation_Support_ankle_down = traj.Equation_solver(0, walktime_n * 0.075, LL_Support_Ankle, 0);
}

void IK_Function::Change_Angle_Compensation(double RL_Support, double RL_Swing, double RL_Knee, double RL_Ankle, double LL_Support, double LL_Swing, double LL_Knee, double LL_Ankle)
{
	RL_Support_Leg = RL_Support * deg2rad;
	RL_Swing_Leg = RL_Swing * deg2rad;
	RL_Support_Knee = RL_Knee * deg2rad; //-5 * deg2rad;
	RL_Support_Ankle = RL_Ankle * deg2rad;

	LL_Support_Leg = LL_Support * deg2rad;
	LL_Swing_Leg = LL_Swing * deg2rad;
	LL_Support_Knee = LL_Knee * deg2rad; //-5 * deg2rad;
	LL_Support_Ankle = LL_Ankle * deg2rad;
}

void IK_Function::Angle_Compensation(int indext, int size)
{

	double check_index = indext % walktime_n;
	double dwalktime = walktime_n;
	if (indext > 74 && indext < size + walktime_n)
	{
		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // swing
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg;
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		}
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;
		}
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) { // support
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
		}

		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) { // support
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		}
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime){
			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime){
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		}

		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(check_index - 0.85 * dwalktime);
	}
}

void IK_Function::Fast_Angle_Compensation(int indext)
{
	int fwalktime_n = walktime_n * 0.5;
	double dwalktime = fwalktime_n;
	double check_index = indext % fwalktime_n;

	if (indext > 0.5 * dwalktime && indext < sim_n + dwalktime)
	{
		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // swing
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg;
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		}
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;
		}
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
		}

		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // support
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		}
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		}

		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(check_index - 0.85 * dwalktime);
	}
}

void IK_Function::Forward_40step_Angle_Compensation(int indext)
{
	int fwalktime_n = walktime_n * 0.75;
	double dwalktime = fwalktime_n;
	double check_index = indext % fwalktime_n;

	if (indext > 0.75 * dwalktime && indext < sim_n + dwalktime)
	{
		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // swing
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg;
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(check_index - 0.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		}
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;
		}
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(check_index - 0.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(check_index - 0.85 * dwalktime);
		}

		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // support
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(check_index - 0.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(check_index - 0.075 * dwalktime);
		}
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
		{
			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(check_index - 0.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(check_index - 0.35 * dwalktime);
		}

		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(check_index - 0.575 * dwalktime);
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // swing
			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(check_index - 0.85 * dwalktime);
	}
}

void IK_Function::Angle_Compensation_Leftwalk(int indext)
{
	double dwalktime = walktime_n;
	if (1.575 * dwalktime < indext && indext < 1.65 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 1.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 1.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 1.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 1.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 1.575 * dwalktime);
	}
	else if (1.65 * dwalktime < indext && indext < 1.85 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (1.85 * dwalktime < indext && indext < 1.925 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 1.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 1.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 1.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 1.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 1.85 * dwalktime);
	}

	else if (2.075 * dwalktime < indext && indext < 2.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 2.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 2.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 2.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 2.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 2.075 * dwalktime);
	}
	else if (2.15 * dwalktime < indext && indext < 2.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (2.35 * dwalktime < indext && indext < 2.425 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 2.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 2.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 2.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 2.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 2.35 * dwalktime);
	}
	
	else if (2.575 * dwalktime < indext && indext < 2.65 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 2.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 2.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 2.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 2.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 2.575 * dwalktime);
	}
	else if (2.65 * dwalktime < indext && indext < 2.85 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (2.85 * dwalktime < indext && indext < 2.925 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 2.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 2.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 2.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 2.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 2.85 * dwalktime);
	}
	
	else if (3.075 * dwalktime < indext && indext < 3.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 3.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 3.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 3.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 3.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 3.075 * dwalktime);
	}
	else if (3.15 * dwalktime < indext && indext < 3.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (3.35 * dwalktime < indext && indext < 3.425 * walktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 3.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 3.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 3.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 3.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 3.35 * dwalktime);
	}

	else if (3.575 * dwalktime < indext && indext < 3.65 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 3.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 3.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 3.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 3.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 3.575 * dwalktime);
	}
	else if (3.65 * dwalktime < indext && indext < 3.85 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (3.85 * dwalktime < indext && indext < 3.925 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 3.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 3.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 3.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 3.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 3.85 * dwalktime);
	}

	else if (4.075 * dwalktime < indext && indext < 4.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 4.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 4.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 4.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 4.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 4.075 * dwalktime);
	}
	else if (4.15 * dwalktime < indext && indext < 4.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (4.35 * dwalktime < indext && indext < 4.425 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 4.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 4.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 4.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 4.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 4.35 * dwalktime);
	}
	
	else if (4.575 * dwalktime < indext && indext < 4.65 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 4.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 4.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 4.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 4.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 4.575 * dwalktime);
	}
	else if (4.65 * dwalktime < indext && indext < 4.85 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (4.85 * dwalktime < indext && indext < 4.925 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 4.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 4.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 4.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 4.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 4.85 * dwalktime);
	}
	
	else if (5.075 * dwalktime < indext && indext < 5.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 5.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 5.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 5.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 5.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 5.075 * dwalktime);
	}
	else if (5.15 * dwalktime < indext && indext < 5.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (5.35 * dwalktime < indext && indext < 5.425 * walktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 5.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 5.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 5.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 5.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 5.35 * dwalktime);
	}

	else if (6.575 * dwalktime < indext && indext < 6.65 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 6.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 6.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 6.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 6.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 6.575 * dwalktime);
	}
	else if (6.65 * dwalktime < indext && indext < 6.85 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (6.85 * dwalktime < indext && indext < 6.925 * dwalktime) // support
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 6.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 6.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 6.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 6.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 6.85 * dwalktime);
	}

	else if (7.075 * dwalktime < indext && indext < 7.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 7.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 7.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 7.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 7.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 7.075 * dwalktime);
	}
	else if (7.15 * dwalktime < indext && indext < 7.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (7.35 * dwalktime < indext && indext < 7.425 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 7.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 7.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 7.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 7.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 7.35 * dwalktime);
	}
	
	else if (7.575 * dwalktime < indext && indext < 7.65 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 7.575 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 7.575 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 7.575 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 7.575 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 7.575 * dwalktime);
	}
	else if (7.65 * dwalktime < indext && indext < 7.85 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;

		LL_th[1] = LL_th[1] + LL_Swing_Leg;
	}
	else if (7.85 * dwalktime < indext && indext < 7.925 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 7.85 * dwalktime);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 7.85 * dwalktime);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 7.85 * dwalktime);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 7.85 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 7.85 * dwalktime);
	}
	
	else if (8.075 * dwalktime < indext && indext < 8.15 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 8.075 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 8.075 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 8.075 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 8.075 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 8.075 * dwalktime);
	}
	else if (8.15 * dwalktime < indext && indext < 8.35 * dwalktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg;

		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;
	}
	else if (8.35 * dwalktime < indext && indext < 8.425 * walktime)
	{
		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 8.35 * dwalktime);

		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 8.35 * dwalktime);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 8.35 * dwalktime);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 8.35 * dwalktime);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 8.35 * dwalktime);
	}

}

void IK_Function::Angle_Compensation_Rightwalk(int indext)
{
	double dwalktime = walktime_n;
	if (indext > 0.5 * walktime_n && indext < sim_n - 0.5*dwalktime)
	{
		if (1.575 * dwalktime < indext && indext < 1.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 1.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 1.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 1.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 1.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 1.575 * dwalktime);
		}
		else if (1.65 * dwalktime < indext && indext < 1.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (1.85 * dwalktime < indext && indext < 1.925 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 1.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 1.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 1.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 1.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 1.85 * dwalktime);
		}

		else if (2.075 * dwalktime < indext && indext < 2.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 2.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 2.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 2.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 2.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 2.075 * dwalktime);
		}
		else if (2.15 * dwalktime < indext && indext < 2.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (2.35 * dwalktime < indext && indext < 2.425 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 2.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 2.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 2.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 2.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 2.35 * dwalktime);
		}
		
		else if (2.575 * dwalktime < indext && indext < 2.65 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 2.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 2.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 2.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 2.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 2.575 * dwalktime);
		}
		else if (2.65 * dwalktime < indext && indext < 2.85 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (2.85 * dwalktime < indext && indext < 2.925 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 2.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 2.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 2.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 2.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 2.85 * dwalktime);
		}
		
		else if (3.075 * dwalktime < indext && indext < 3.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 3.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 3.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 3.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 3.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 3.075 * dwalktime);
		}
		else if (3.15 * dwalktime < indext && indext < 3.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (3.35 * dwalktime < indext && indext < 3.425 * walktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 3.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 3.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 3.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 3.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 3.35 * dwalktime);
		}

		else if (3.575 * dwalktime < indext && indext < 3.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 3.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 3.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 3.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 3.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 3.575 * dwalktime);
		}
		else if (3.65 * dwalktime < indext && indext < 3.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (3.85 * dwalktime < indext && indext < 3.925 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 3.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 3.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 3.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 3.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 3.85 * dwalktime);
		}

		else if (4.075 * dwalktime < indext && indext < 4.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 4.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 4.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 4.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 4.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 4.075 * dwalktime);
		}
		else if (4.15 * dwalktime < indext && indext < 4.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (4.35 * dwalktime < indext && indext < 4.425 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 4.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 4.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 4.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 4.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 4.35 * dwalktime);
		}
		
		else if (4.575 * dwalktime < indext && indext < 4.65 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 4.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 4.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 4.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 4.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 4.575 * dwalktime);
		}
		else if (4.65 * dwalktime < indext && indext < 4.85 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (4.85 * dwalktime < indext && indext < 4.925 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 4.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 4.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 4.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 4.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 4.85 * dwalktime);
		}
		
		else if (5.075 * dwalktime < indext && indext < 5.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 5.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 5.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 5.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 5.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 5.075 * dwalktime);
		}
		else if (5.15 * dwalktime < indext && indext < 5.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (5.35 * dwalktime < indext && indext < 5.425 * walktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 5.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 5.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 5.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 5.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 5.35 * dwalktime);
		}

		else if (5.575 * dwalktime < indext && indext < 5.65 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 5.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 5.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 5.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 5.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 5.575 * dwalktime);
		}
		else if (5.65 * dwalktime < indext && indext < 5.85 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (5.85 * dwalktime < indext && indext < 5.925 * dwalktime) // support
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 5.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 5.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 5.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 5.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 5.85 * dwalktime);
		}

		else if (6.075 * dwalktime < indext && indext < 6.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 6.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 6.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 6.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 6.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 6.075 * dwalktime);
		}
		else if (6.15 * dwalktime < indext && indext < 6.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (6.35 * dwalktime < indext && indext < 6.425 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 6.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 6.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 6.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 6.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 6.35 * dwalktime);
		}
		
		else if (6.575 * dwalktime < indext && indext < 6.65 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 6.575 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 6.575 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 6.575 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 6.575 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 6.575 * dwalktime);
		}
		else if (6.65 * dwalktime < indext && indext < 6.85 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg;
			LL_th[3] = LL_th[3] + LL_Support_Knee;
			LL_th[4] = LL_th[4] + LL_Support_Ankle;
			LL_th[5] = LL_th[5] + LL_Support_Leg;

			RL_th[1] = RL_th[1] - RL_Swing_Leg;
		}
		else if (6.85 * dwalktime < indext && indext < 6.925 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 6.85 * dwalktime);
			LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 6.85 * dwalktime);
			LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 6.85 * dwalktime);
			LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 6.85 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 6.85 * dwalktime);
		}
		
		else if (7.075 * dwalktime < indext && indext < 7.15 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 7.075 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 7.075 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 7.075 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 7.075 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 7.075 * dwalktime);
		}
		else if (7.15 * dwalktime < indext && indext < 7.35 * dwalktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg;

			RL_th[1] = RL_th[1] - RL_Support_Leg;
			RL_th[3] = RL_th[3] - RL_Support_Knee;
			RL_th[4] = RL_th[4] - RL_Support_Ankle;
			RL_th[5] = RL_th[5] - RL_Support_Leg;
		}
		else if (7.35 * dwalktime < indext && indext < 7.425 * walktime)
		{
			LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 7.35 * dwalktime);

			RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 7.35 * dwalktime);
			RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 7.35 * dwalktime);
			RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 7.35 * dwalktime);
			RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 7.35 * dwalktime);
		}

	
	
	}
}

void IK_Function::Angle_Compensation_Huddle(int indext)
{
	double dwalktime = walktime_n;
	if (indext > 0.8 * dwalktime + 100 && indext < 0.875 * dwalktime + 100) // swing
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 0.8 * dwalktime - 100);
		LL_th[5] = LL_th[5] - LL_Swing_Leg_Compensation_up(indext - 0.8 * dwalktime - 100);

		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 0.8 * dwalktime - 100);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 0.8 * dwalktime - 100);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 0.8 * dwalktime - 100);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 0.8 * dwalktime - 100);
	}
	else if (indext > 0.875 * dwalktime + 100 && indext < 1.625 * dwalktime + 100)
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg;
		LL_th[5] = LL_th[5] - LL_Swing_Leg;

		RL_th[1] = RL_th[1] - RL_Support_Leg;
		RL_th[3] = RL_th[3] - RL_Support_Knee;
		RL_th[4] = RL_th[4] - RL_Support_Ankle;
		RL_th[5] = RL_th[5] - RL_Support_Leg;
	}
	else if (indext > 1.625 * dwalktime + 100 && indext < 1.7 * dwalktime + 100)
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 1.625 * dwalktime - 100);
		LL_th[5] = LL_th[5] - 1*LL_Swing_Leg_Compensation_down(indext - 1.625 * dwalktime - 100);

		RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 1.625 * dwalktime - 100);
		RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 1.625 * dwalktime - 100);
		RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 1.625 * dwalktime - 100);
		RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 1.625 * dwalktime - 100);
	}
	else if (indext > 2.3 * dwalktime + 100 && indext < 2.375 * dwalktime + 100) // support
	{
		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_up(indext - 2.3 * dwalktime - 100);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_up(indext - 2.3 * dwalktime - 100);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_up(indext - 2.3 * dwalktime - 100);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_up(indext - 2.3 * dwalktime - 100);

		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_up(indext - 2.3 * dwalktime - 100);
		// RL_th[5] = RL_th[5] + RL_Swing_Leg_Compensation_up(indext - 2.3 * dwalktime - 100);
	}
	else if (indext > 2.375 * dwalktime + 100 && indext < 3.625 * dwalktime + 100) // support
	{
		LL_th[1] = LL_th[1] + LL_Support_Leg;
		LL_th[3] = LL_th[3] + LL_Support_Knee;
		LL_th[4] = LL_th[4] + LL_Support_Ankle;
		LL_th[5] = LL_th[5] + LL_Support_Leg;

		RL_th[1] = RL_th[1] - RL_Swing_Leg;
		// RL_th[5] = RL_th[5] + RL_Swing_Leg;
	}
	else if (indext > 3.625 * dwalktime + 100 && indext < 3.7 * dwalktime + 100) // support
	{
		LL_th[1] = LL_th[1] + LL_Support_Leg_Compensation_down(indext - 3.625 * dwalktime - 100);
		LL_th[3] = LL_th[3] + LL_Support_Knee_Compensation_down(indext - 3.625 * dwalktime - 100);
		LL_th[4] = LL_th[4] + LL_Support_Ankle_Compensation_down(indext - 3.625 * dwalktime - 100);
		LL_th[5] = LL_th[5] + LL_Support_Leg_Compensation_down(indext - 3.625 * dwalktime - 100);

		RL_th[1] = RL_th[1] - RL_Swing_Leg_Compensation_down(indext - 3.625 * dwalktime - 100);
		// RL_th[5] = RL_th[5] + RL_Swing_Leg_Compensation_down(indext - 3.625 * dwalktime - 100);
	}

	else if (indext > 5 * dwalktime + 100 && indext < 5.075 * dwalktime + 100) // support
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_up(indext - 5 * dwalktime - 100);
		// LL_th[5] = LL_th[5] - LL_Swing_Leg_Compensation_up(indext - 4 * dwalktime - 100);

		// RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_up(indext - 4 * dwalktime - 100);
		// RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_up(indext - 4 * dwalktime - 100);
		// RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_up(indext - 34 * dwalktime - 100);
		// RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_up(indext - 4 * dwalktime - 100);
	}
	else if (indext > 5.075 * dwalktime + 100 && indext < 5.625 * dwalktime + 100)
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg;
		// LL_th[5] = LL_th[5] - LL_Swing_Leg;

		// RL_th[1] = RL_th[1] - RL_Support_Leg;
		// RL_th[3] = RL_th[3] - RL_Support_Knee;
		// RL_th[4] = RL_th[4] - RL_Support_Ankle;
		// RL_th[5] = RL_th[5] - RL_Support_Leg;
	}
	else if (indext > 5.625 * dwalktime + 100 && indext < 5.7 * dwalktime + 100)
	{
		LL_th[1] = LL_th[1] + LL_Swing_Leg_Compensation_down(indext - 5.625 * dwalktime - 100);
		// LL_th[5] = LL_th[5] - LL_Swing_Leg_Compensation_down(indext - 4.625 * dwalktime - 100);

		// RL_th[1] = RL_th[1] - RL_Support_Leg_Compensation_down(indext - 4.625 * dwalktime - 100);
		// RL_th[3] = RL_th[3] - RL_Support_Knee_Compensation_down(indext - 4.625 * dwalktime - 100);
		// RL_th[4] = RL_th[4] - RL_Support_Ankle_Compensation_down(indext - 4.625 * dwalktime - 100);
		// RL_th[5] = RL_th[5] - RL_Support_Leg_Compensation_down(indext - 4.625 * dwalktime - 100);
	}
}

double IK_Function::RL_Swing_Leg_Compensation_up(double t)
{
	double X = RL_Compensation_Swing_Leg_up(0) + RL_Compensation_Swing_Leg_up(1) * t + RL_Compensation_Swing_Leg_up(2) * pow(t, 2) + RL_Compensation_Swing_Leg_up(3) * pow(t, 3) + RL_Compensation_Swing_Leg_up(4) * pow(t, 4) + RL_Compensation_Swing_Leg_up(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Leg_Compensation_up(double t)
{
	double X = RL_Compensation_Support_Leg_up(0) + RL_Compensation_Support_Leg_up(1) * t + RL_Compensation_Support_Leg_up(2) * pow(t, 2) + RL_Compensation_Support_Leg_up(3) * pow(t, 3) + RL_Compensation_Support_Leg_up(4) * pow(t, 4) + RL_Compensation_Support_Leg_up(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Swing_Leg_Compensation_down(double t)
{
	double X = RL_Compensation_Swing_Leg_down(0) + RL_Compensation_Swing_Leg_down(1) * t + RL_Compensation_Swing_Leg_down(2) * pow(t, 2) + RL_Compensation_Swing_Leg_down(3) * pow(t, 3) + RL_Compensation_Swing_Leg_down(4) * pow(t, 4) + RL_Compensation_Swing_Leg_down(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Leg_Compensation_down(double t)
{
	double X = RL_Compensation_Support_Leg_down(0) + RL_Compensation_Support_Leg_down(1) * t + RL_Compensation_Support_Leg_down(2) * pow(t, 2) + RL_Compensation_Support_Leg_down(3) * pow(t, 3) + RL_Compensation_Support_Leg_down(4) * pow(t, 4) + RL_Compensation_Support_Leg_down(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Knee_Compensation_up(double t)
{
	double X = RL_Compensation_Support_knee_up(0) + RL_Compensation_Support_knee_up(1) * t + RL_Compensation_Support_knee_up(2) * pow(t, 2) + RL_Compensation_Support_knee_up(3) * pow(t, 3) + RL_Compensation_Support_knee_up(4) * pow(t, 4) + RL_Compensation_Support_knee_up(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Knee_Compensation_down(double t)
{
	double X = RL_Compensation_Support_knee_down(0) + RL_Compensation_Support_knee_down(1) * t + RL_Compensation_Support_knee_down(2) * pow(t, 2) + RL_Compensation_Support_knee_down(3) * pow(t, 3) + RL_Compensation_Support_knee_down(4) * pow(t, 4) + RL_Compensation_Support_knee_down(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Ankle_Compensation_up(double t)
{
	double X = RL_Compensation_Support_ankle_up(0) + RL_Compensation_Support_ankle_up(1) * t + RL_Compensation_Support_ankle_up(2) * pow(t, 2) + RL_Compensation_Support_ankle_up(3) * pow(t, 3) + RL_Compensation_Support_ankle_up(4) * pow(t, 4) + RL_Compensation_Support_ankle_up(5) * pow(t, 5);
	return X;
};
double IK_Function::RL_Support_Ankle_Compensation_down(double t)
{
	double X = RL_Compensation_Support_ankle_down(0) + RL_Compensation_Support_ankle_down(1) * t + RL_Compensation_Support_ankle_down(2) * pow(t, 2) + RL_Compensation_Support_ankle_down(3) * pow(t, 3) + RL_Compensation_Support_ankle_down(4) * pow(t, 4) + RL_Compensation_Support_ankle_down(5) * pow(t, 5);
	return X;
};

double IK_Function::LL_Swing_Leg_Compensation_up(double t)
{
	double X = LL_Compensation_Swing_Leg_up(0) + LL_Compensation_Swing_Leg_up(1) * t + LL_Compensation_Swing_Leg_up(2) * pow(t, 2) + LL_Compensation_Swing_Leg_up(3) * pow(t, 3) + LL_Compensation_Swing_Leg_up(4) * pow(t, 4) + LL_Compensation_Swing_Leg_up(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Leg_Compensation_up(double t)
{
	double X = LL_Compensation_Support_Leg_up(0) + LL_Compensation_Support_Leg_up(1) * t + LL_Compensation_Support_Leg_up(2) * pow(t, 2) + LL_Compensation_Support_Leg_up(3) * pow(t, 3) + LL_Compensation_Support_Leg_up(4) * pow(t, 4) + LL_Compensation_Support_Leg_up(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Swing_Leg_Compensation_down(double t)
{
	double X = LL_Compensation_Swing_Leg_down(0) + LL_Compensation_Swing_Leg_down(1) * t + LL_Compensation_Swing_Leg_down(2) * pow(t, 2) + LL_Compensation_Swing_Leg_down(3) * pow(t, 3) + LL_Compensation_Swing_Leg_down(4) * pow(t, 4) + LL_Compensation_Swing_Leg_down(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Leg_Compensation_down(double t)
{
	double X = LL_Compensation_Support_Leg_down(0) + LL_Compensation_Support_Leg_down(1) * t + LL_Compensation_Support_Leg_down(2) * pow(t, 2) + LL_Compensation_Support_Leg_down(3) * pow(t, 3) + LL_Compensation_Support_Leg_down(4) * pow(t, 4) + LL_Compensation_Support_Leg_down(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Knee_Compensation_up(double t)
{
	double X = LL_Compensation_Support_knee_up(0) + LL_Compensation_Support_knee_up(1) * t + LL_Compensation_Support_knee_up(2) * pow(t, 2) + LL_Compensation_Support_knee_up(3) * pow(t, 3) + LL_Compensation_Support_knee_up(4) * pow(t, 4) + LL_Compensation_Support_knee_up(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Knee_Compensation_down(double t)
{
	double X = LL_Compensation_Support_knee_down(0) + LL_Compensation_Support_knee_down(1) * t + LL_Compensation_Support_knee_down(2) * pow(t, 2) + LL_Compensation_Support_knee_down(3) * pow(t, 3) + LL_Compensation_Support_knee_down(4) * pow(t, 4) + LL_Compensation_Support_knee_down(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Ankle_Compensation_up(double t)
{
	double X = LL_Compensation_Support_ankle_up(0) + LL_Compensation_Support_ankle_up(1) * t + LL_Compensation_Support_ankle_up(2) * pow(t, 2) + LL_Compensation_Support_ankle_up(3) * pow(t, 3) + LL_Compensation_Support_ankle_up(4) * pow(t, 4) + LL_Compensation_Support_ankle_up(5) * pow(t, 5);
	return X;
};
double IK_Function::LL_Support_Ankle_Compensation_down(double t)
{
	double X = LL_Compensation_Support_ankle_down(0) + LL_Compensation_Support_ankle_down(1) * t + LL_Compensation_Support_ankle_down(2) * pow(t, 2) + LL_Compensation_Support_ankle_down(3) * pow(t, 3) + LL_Compensation_Support_ankle_down(4) * pow(t, 4) + LL_Compensation_Support_ankle_down(5) * pow(t, 5);
	return X;
};











Pick::Pick() : freq(100), del_t(1.0 / freq),
               Ref_WT_th(RowVectorXd::Zero(0)),
               Ref_RA_th(RowVectorXd::Zero(0)),
               Ref_LA_th(RowVectorXd::Zero(0)),
               Ref_NC_th(RowVectorXd::Zero(0))
{
    // Initialize default values for trajectory arrays
    for (int i = 0; i < 1; ++i) {
        WT_th[i] = 0.0;
    }
    for (int i = 0; i < 4; ++i) {
        RA_th[i] = 0.0;
        LA_th[i] = 0.0;
    }
    for (int i = 0; i < 2; ++i) {
        NC_th[i] = 0.0;
    }
}

void Pick::WT_Trajectory(double WT_th, int length)
{
    RowVectorXd WT(length);
    RowVectorXd WT_add(length);

    // Compute the WT values
    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1);  
        WT(i) = WT_th * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }
    // Compute WT_add
    WT_add(0) = WT(0); // Initial value
    for (int i = 1; i < length; ++i)
    {
        WT_add(i) = WT(i) - WT(i - 1);
    }
    
    // Store the result in Ref_WT_th
    Ref_WT_th.resize(1, length);  
    Ref_WT_th.row(0) = WT_add * deg2rad;
}

void Pick::RA_Trajectory(double RA_th1, double RA_th2, double RA_th3, double RA_th4, int length)
{
    RowVectorXd RA_1(length);
    RowVectorXd RA_2(length);
    RowVectorXd RA_3(length);
    RowVectorXd RA_4(length);
    RowVectorXd RA_add_1(length);
    RowVectorXd RA_add_2(length);
    RowVectorXd RA_add_3(length);
    RowVectorXd RA_add_4(length);

    // Compute the RA values
    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1); 
        RA_1(i) = RA_th1 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RA_2(i) = RA_th2 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RA_3(i) = RA_th3 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RA_4(i) = RA_th4 * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }

    // Compute RA_add
    RA_add_1(0) = RA_1(0); // Initial value
    RA_add_2(0) = RA_2(0); // Initial value
    RA_add_3(0) = RA_3(0); // Initial value
    RA_add_4(0) = RA_4(0); // Initial value

    for (int i = 1; i < length; ++i)
    {
        RA_add_1(i) = RA_1(i) - RA_1(i - 1);
        RA_add_2(i) = RA_2(i) - RA_2(i - 1);
        RA_add_3(i) = RA_3(i) - RA_3(i - 1);
        RA_add_4(i) = RA_4(i) - RA_4(i - 1);
    }
    
    // Store the result in Ref_RA_th
    Ref_RA_th.resize(4, length);  
    Ref_RA_th.row(0) = RA_add_1 * deg2rad;
    Ref_RA_th.row(1) = RA_add_2 * deg2rad;
    Ref_RA_th.row(2) = RA_add_3 * deg2rad;
    Ref_RA_th.row(3) = RA_add_4 * deg2rad;
}

void Pick::LA_Trajectory(double LA_th1, double LA_th2, double LA_th3, double LA_th4, int length)
{
    RowVectorXd LA_1(length);
    RowVectorXd LA_2(length);
    RowVectorXd LA_3(length);
    RowVectorXd LA_4(length);
    RowVectorXd LA_add_1(length);
    RowVectorXd LA_add_2(length);
    RowVectorXd LA_add_3(length);
    RowVectorXd LA_add_4(length);

    // Compute the LA values
    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1); 
        LA_1(i) = LA_th1 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LA_2(i) = LA_th2 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LA_3(i) = LA_th3 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LA_4(i) = LA_th4 * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }

    // Compute LA_add
    LA_add_1(0) = LA_1(0); // Initial value
    LA_add_2(0) = LA_2(0); // Initial value
    LA_add_3(0) = LA_3(0); // Initial value
    LA_add_4(0) = LA_4(0); // Initial value

    for (int i = 1; i < length; ++i)
    {
        LA_add_1(i) = LA_1(i) - LA_1(i - 1);
        LA_add_2(i) = LA_2(i) - LA_2(i - 1);
        LA_add_3(i) = LA_3(i) - LA_3(i - 1);
        LA_add_4(i) = LA_4(i) - LA_4(i - 1);
    }

    // Store the result in Ref_LA_th
    Ref_LA_th.resize(4, length);
    Ref_LA_th.row(0) = LA_add_1 * deg2rad;
    Ref_LA_th.row(1) = LA_add_2 * deg2rad;
    Ref_LA_th.row(2) = LA_add_3 * deg2rad;
    Ref_LA_th.row(3) = LA_add_4 * deg2rad;
}

void Pick::NC_Trajectory(double NC_th1, double NC_th2, int length)
{
    RowVectorXd NC_1(length);
    RowVectorXd NC_2(length);
    RowVectorXd NC_add_1(length);
    RowVectorXd NC_add_2(length);

    // Compute the NC values
    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1); 
        NC_1(i) = NC_th1 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        NC_2(i) = NC_th2 * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }

    // Compute NC_add
    NC_add_1(0) = NC_1(0); // Initial value
    NC_add_2(0) = NC_2(0); // Initial value

    for (int i = 1; i < length; ++i)
    {
        NC_add_1(i) = NC_1(i) - NC_1(i - 1);
        NC_add_2(i) = NC_2(i) - NC_2(i - 1);
    }

    // Store the result in Ref_NC_th
    Ref_NC_th.resize(2, length);
    Ref_NC_th.row(0) = NC_add_1 * deg2rad;
    Ref_NC_th.row(1) = NC_add_2 * deg2rad;
}

void Pick::RL_Trajectory(double th1, double th2, double th3, double th4, double th5, double th6, int length)
{
    RowVectorXd RL_1(length), RL_2(length), RL_3(length), RL_4(length), RL_5(length), RL_6(length);
    RowVectorXd RL_add_1(length), RL_add_2(length), RL_add_3(length), RL_add_4(length), RL_add_5(length), RL_add_6(length);

    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1);
        RL_1(i) = th1 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RL_2(i) = th2 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RL_3(i) = th3 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RL_4(i) = th4 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RL_5(i) = th5 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        RL_6(i) = th6 * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }

    RL_add_1(0) = RL_1(0);
    RL_add_2(0) = RL_2(0);
    RL_add_3(0) = RL_3(0);
    RL_add_4(0) = RL_4(0);
    RL_add_5(0) = RL_5(0);
    RL_add_6(0) = RL_6(0);

    for (int i = 1; i < length; ++i)
    {
        RL_add_1(i) = RL_1(i) - RL_1(i - 1);
        RL_add_2(i) = RL_2(i) - RL_2(i - 1);
        RL_add_3(i) = RL_3(i) - RL_3(i - 1);
        RL_add_4(i) = RL_4(i) - RL_4(i - 1);
        RL_add_5(i) = RL_5(i) - RL_5(i - 1);
        RL_add_6(i) = RL_6(i) - RL_6(i - 1);
    }

    Ref_RL_th.resize(6, length);
    Ref_RL_th.row(0) = RL_add_1 * deg2rad;
    Ref_RL_th.row(1) = RL_add_2 * deg2rad;
    Ref_RL_th.row(2) = RL_add_3 * deg2rad;
    Ref_RL_th.row(3) = RL_add_4 * deg2rad;
    Ref_RL_th.row(4) = RL_add_5 * deg2rad;
    Ref_RL_th.row(5) = RL_add_6 * deg2rad;
}

void Pick::LL_Trajectory(double th1, double th2, double th3, double th4, double th5, double th6, int length)
{
    RowVectorXd LL_1(length), LL_2(length), LL_3(length), LL_4(length), LL_5(length), LL_6(length);
    RowVectorXd LL_add_1(length), LL_add_2(length), LL_add_3(length), LL_add_4(length), LL_add_5(length), LL_add_6(length);

    for (int i = 0; i < length; ++i)
    {
        double t = del_t * (i + 1);
        LL_1(i) = th1 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LL_2(i) = th2 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LL_3(i) = th3 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LL_4(i) = th4 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LL_5(i) = th5 * 0.5 * (1 - cos(PI * t / (del_t * length)));
        LL_6(i) = th6 * 0.5 * (1 - cos(PI * t / (del_t * length)));
    }

    LL_add_1(0) = LL_1(0);
    LL_add_2(0) = LL_2(0);
    LL_add_3(0) = LL_3(0);
    LL_add_4(0) = LL_4(0);
    LL_add_5(0) = LL_5(0);
    LL_add_6(0) = LL_6(0);

    for (int i = 1; i < length; ++i)
    {
        LL_add_1(i) = LL_1(i) - LL_1(i - 1);
        LL_add_2(i) = LL_2(i) - LL_2(i - 1);
        LL_add_3(i) = LL_3(i) - LL_3(i - 1);
        LL_add_4(i) = LL_4(i) - LL_4(i - 1);
        LL_add_5(i) = LL_5(i) - LL_5(i - 1);
        LL_add_6(i) = LL_6(i) - LL_6(i - 1);
    }

    Ref_LL_th.resize(6, length);
    Ref_LL_th.row(0) = LL_add_1 * deg2rad;
    Ref_LL_th.row(1) = LL_add_2 * deg2rad;
    Ref_LL_th.row(2) = LL_add_3 * deg2rad;
    Ref_LL_th.row(3) = LL_add_4 * deg2rad;
    Ref_LL_th.row(4) = LL_add_5 * deg2rad;
    Ref_LL_th.row(5) = LL_add_6 * deg2rad;
}


void Pick::UPBD_SET(const MatrixXd& WT, const MatrixXd& RA, const MatrixXd& LA, const MatrixXd& NC, int Index_CNT)
{
    // Ensure Index_CNT is within bounds
    Index_CNT = std::min(Index_CNT, static_cast<int>(RA.cols()) - 1);

    // Function to safely set values from Ref_* arrays
    auto set_values = [Index_CNT](auto& dest, const auto& ref, int size) {
        for (int i = 0; i < size; ++i) {
            dest[i] = ref(i, Index_CNT);
        }
    };


	// Function to safely add values from Ref_* arrays to dest arrays
    auto add_values = [Index_CNT](auto& dest, const auto& ref, int size) {
        for (int i = 0; i < size; ++i) {
            dest[i] += ref(i, Index_CNT); // Perform addition instead of direct assignment
        }
    };

    // // Set values for WT_th, RA_th, LA_th, NC_th
    // set_values(WT_th, Ref_WT_th, 1);
    // set_values(RA_th, Ref_RA_th, 4);
    // set_values(LA_th, Ref_LA_th, 4);
    // set_values(NC_th, Ref_NC_th, 2);
	
	add_values(WT_th, Ref_WT_th, 1);
    add_values(RA_th, Ref_RA_th, 4);
    add_values(LA_th, Ref_LA_th, 4);
    add_values(NC_th, Ref_NC_th, 2);
}

void Pick::UPBD_SET_ALL(const MatrixXd& WT, const MatrixXd& RA, const MatrixXd& LA,
                    const MatrixXd& NC, const MatrixXd& RL, const MatrixXd& LL,
                    int Index_CNT)
{
    // Ensure Index_CNT is within bounds (RA 기준으로 확인)
    Index_CNT = std::min(Index_CNT, static_cast<int>(RA.cols()) - 1);

    // 함수: 값 덮어쓰기
    auto set_values = [Index_CNT](auto& dest, const auto& ref, int size) {
        for (int i = 0; i < size; ++i) {
            dest[i] = ref(i, Index_CNT);
        }
    };

    // 함수: 누적 더하기
    auto add_values = [Index_CNT](auto& dest, const auto& ref, int size) {
        for (int i = 0; i < size; ++i) {
            dest[i] += ref(i, Index_CNT);
        }
    };

    // === 최종 반영 ===
    add_values(WT_th, Ref_WT_th, 1);   // 허리 1자유도
    add_values(RA_th, Ref_RA_th, 4);   // 오른팔 4자유도
    add_values(LA_th, Ref_LA_th, 4);   // 왼팔 4자유도
    add_values(NC_th, Ref_NC_th, 2);   // 목 2자유도
    add_values(RL_th_ALL, Ref_RL_th, 6);   // 오른다리 6자유도
    add_values(LL_th_ALL, Ref_LL_th, 6);   // 왼다리 6자유도
}




void Pick::Fast_Arm_Swing(int indext, int walktime_n, double sim_n)
{

	int fwalktime_n = walktime_n * 0.5;
	double dwalktime = fwalktime_n;
	double check_index = indext % fwalktime_n;


	if (indext > 0.5 * dwalktime && indext < sim_n + dwalktime)
	{
		if (check_index > 0.075 * dwalktime && check_index < 0.15 * dwalktime) // swing
		{
			UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, check_index - 0.075 * dwalktime);

			std::cout << "11111111111" << std::endl;

		}
		else if (check_index > 0.15 * dwalktime && check_index < 0.35 * dwalktime)
		{
			// LL_th[1] = LL_th[1] + LL_Swing_Leg;
			std::cout << "22222222222" << std::endl;
		}
		else if (check_index > 0.35 * dwalktime && check_index < 0.425 * dwalktime)
		{
			RA_Trajectory(-10,-10,-10,-10, 0.75*dwalktime);
		    LA_Trajectory(-10,-10,-10,-10, 0.75*dwalktime);
			UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, check_index - 0.35 * dwalktime);
			std::cout << "3333333333333" << std::endl;
		}
		else if (check_index > 0.575 * dwalktime && check_index < 0.65 * dwalktime) // support
		{
			RA_Trajectory(10,10,10,10, 0.75*dwalktime);
		    LA_Trajectory(10,10,10,10, 0.75*dwalktime);
			UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, check_index - 0.575 * dwalktime);

			std::cout << "44444444444" << std::endl;
		}
		else if (check_index > 0.65 * dwalktime && check_index < 0.85 * dwalktime) // support
		{
			// LL_th[1] = LL_th[1] + LL_Support_Leg;
			// LL_th[3] = LL_th[3] + LL_Support_Knee;
			// LL_th[4] = LL_th[4] + LL_Support_Ankle;
			// LL_th[5] = LL_th[5] + LL_Support_Leg;
			
			std::cout << "555555555555" << std::endl;
		}
		else if (check_index > 0.85 * dwalktime && check_index < 0.925 * dwalktime) // support
		{
			RA_Trajectory(-10,-10,-10,-10, 0.75*dwalktime);
		    LA_Trajectory(-10,-10,-10,-10, 0.75*dwalktime);
			UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, check_index - 0.85 * dwalktime);

			    std::cout << "6666666666" << std::endl;


		}
	}
    // std::cout << Ref_RA_th(1, check_index) << std::endl;
}

void Pick::Picking(const MatrixXd& RFx, int Index_CNT, double& RL, double& LL)
{

	if (Index_CNT > RFx.cols() - 1)
    {
        Index_CNT = RFx.cols();
    }
    // std::cout << RFx.cols() << std::endl;
	

	if( 150 < Index_CNT && Index_CNT < 231)
	{
		WT_Trajectory(-55, 80);
        RA_Trajectory(0,0,0,0,80);
        LA_Trajectory(-30,-10,110,-60,80);
        NC_Trajectory(0,0,80);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-150);

	}
	else if( 300 < Index_CNT && Index_CNT < 331)
	{
		WT_Trajectory(0, 30);
        RA_Trajectory(0,0,0,0,30);
        LA_Trajectory(0,0,0,60,30);
        NC_Trajectory(0,0,30);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-300);

	}
    else if( 620 < Index_CNT && Index_CNT < 751)
	{
		WT_Trajectory(55, 130);
        RA_Trajectory(0,0,0,0,130);
        LA_Trajectory(30,10,-110,0,130);
        NC_Trajectory(0,0,130);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-620);

	}

	if(100 < Index_CNT && Index_CNT < 250 )
	{

		RL += 0.1;
		LL += 0.1;

	}

	else if(350 < Index_CNT && Index_CNT < 500 )
	{

		RL += -0.1;
		LL += -0.1;

	}
	    // std::cout << Index_CNT << std::endl;
	    // std::cout << RL << std::endl;

	    // std::cout << RFx.cols() << std::endl;

		//150mm

}

void Pick::Stand_up(const MatrixXd& RFx, int Index_CNT)
{
	if (Index_CNT > RFx.cols() - 1)
    {
        Index_CNT = RFx.cols();
    }


	if( 0 <= Index_CNT && Index_CNT < 200)
	{
		// WT_Trajectory(0, 200);
        // RA_Trajectory(0,0,0,0,200);
        // LA_Trajectory(0,0,0,0,200);
        // NC_Trajectory(0,0,200);
		// RL_Trajectory(0,0,0,0,55,0,200);
        // LL_Trajectory(0,0,0,0,-55,0,200);
		WT_Trajectory(0, 200);
        RA_Trajectory(90,25,-90,0,200);
        LA_Trajectory(-90,-25,90,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,-80,-100,80,0,200);
        LL_Trajectory(0,0,80,100,-80,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT);


	}
	else if( 200 <= Index_CNT && Index_CNT < 400)
	{
		// WT_Trajectory(0, 200);
        // RA_Trajectory(90,0,-90,0,200);
        // LA_Trajectory(-90,0,90,0,200);
        // NC_Trajectory(0,0,200);
		// RL_Trajectory(0,0,-75,-83,20,0,200);
        // LL_Trajectory(0,0,75,83,-20,0,200);
		WT_Trajectory(0, 200);
        RA_Trajectory(-42,0,0,0,200);
        LA_Trajectory(42,0,0,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,25,0,0,0,200);
        LL_Trajectory(0,0,-25,0,0,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-200);

	}
    else if( 400 <= Index_CNT && Index_CNT < 600)
	{
		// WT_Trajectory(0, 200);
        // RA_Trajectory(-180,0,0,0,200);
        // LA_Trajectory(180,0,0,0,200);
        // NC_Trajectory(0,0,200);
		// RL_Trajectory(0,0,40,-17,-20,0,200);
        // LL_Trajectory(0,0,-40,17,20,0,200);
		// WT_Trajectory(0, 200);
        // RA_Trajectory(-30,25,0,0,200);
        // LA_Trajectory(30,-25,0,0,200);
        // NC_Trajectory(0,0,200);
		// RL_Trajectory(0,0,25,0,0,0,200);
        // LL_Trajectory(0,0,-25,0,0,0,200);
		WT_Trajectory(0, 200);
        RA_Trajectory(-138,-25,90,0,200);
        LA_Trajectory(138,25,-90,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,30,0,0,0,200);
        LL_Trajectory(0,0,-30,0,0,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-400);

	}
	else if( 600 <= Index_CNT && Index_CNT < 800)
	{
		// WT_Trajectory(20, 200);
        // RA_Trajectory(0,0,0,0,200);
        // LA_Trajectory(0,0,0,0,200);
        // NC_Trajectory(0,0,200);
		// RL_Trajectory(0,0,15,-10,0,0,200);
        // LL_Trajectory(0,0,-15,10,0,0,200);
		WT_Trajectory(0, 200);
        RA_Trajectory(90,0,0,0,200);
        LA_Trajectory(-90,0,0,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,0,0,0,0,200);
        LL_Trajectory(0,0,0,0,0,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-600);

	}
	else if( 800 <= Index_CNT && Index_CNT < 1000)
	{
		WT_Trajectory(0, 200);
        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,0,0,0,0,200);
        LL_Trajectory(0,0,0,0,0,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-800);

	}
		else if( 1000 <= Index_CNT && Index_CNT < 1200)
	{
		WT_Trajectory(0, 200);
        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);
        NC_Trajectory(0,0,200);
		RL_Trajectory(0,0,0,0,0,0,200);
        LL_Trajectory(0,0,0,0,0,0,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1000);

	}
}


void Pick::hhhh(const MatrixXd& RFx, int Index_CNT)
{
	if (Index_CNT > RFx.cols() - 1)
    {
        Index_CNT = RFx.cols();
    }


	if( 0 <= Index_CNT && Index_CNT < 200)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,60,60,0,200);

        NC_Trajectory(0,0,200);

		RL_Trajectory(0    , -13,0    ,0    ,0,-13    ,200);
        LL_Trajectory(0    ,-13,0    ,0    ,0,-13    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT);
	}

	if( 200 <= Index_CNT && Index_CNT < 400)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,-10,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,0    ,-30    ,30,0    ,200);
        LL_Trajectory(0    ,0,0    ,0    ,0,-2    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-200);


	}

	if( 400 <= Index_CNT && Index_CNT < 600)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,-80    ,-75    ,0,0    ,200);
        LL_Trajectory(0    ,0,0    ,0    ,0,0    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-400);


	}

	if( 600 <= Index_CNT && Index_CNT < 800)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 8,30    ,110    ,-80,8    ,200);
        LL_Trajectory(0    ,0,0    ,0    ,0,0    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-600);


	}

	if( 800 <= Index_CNT && Index_CNT < 900)
	{
		WT_Trajectory(0, 100);

        RA_Trajectory(0,0,0,0,100);
        LA_Trajectory(0,0,0,0,100);

        NC_Trajectory(0,0,100);
		
		RL_Trajectory(0    , 0,25    ,45    ,-20,0    ,100);
        LL_Trajectory(0    ,0,0    ,0    ,0,0    ,100);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-800);


	}

	if( 900 <= Index_CNT && Index_CNT < 1000)
	{
		WT_Trajectory(0, 100);

        RA_Trajectory(0,-60,0,0,100);
        LA_Trajectory(0,0,-60,0,100);

        NC_Trajectory(0,0,100);

		// RL_Trajectory(0    , 0,0    ,0    ,-5,0    ,100);
        // LL_Trajectory(0    ,0,0    ,25    ,-25,0    ,100);
		RL_Trajectory(0    , 3,0    ,0    ,-5,3    ,100);
        LL_Trajectory(0    ,3,0    ,25    ,-25,3    ,100);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-900);


	}

	if( 1000 <= Index_CNT && Index_CNT < 1200)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,-60,0,0,200);

        NC_Trajectory(0,0,200);
		
		// RL_Trajectory(0    ,5,0    ,-50    ,50,5    ,100);
        // LL_Trajectory(0    ,13,-25    ,-25    ,0,15    ,100);

		RL_Trajectory(0    ,2,0    ,-50    ,50,2    ,200);
        LL_Trajectory(0    ,10,-25    ,-25    ,0,12    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1000);


	}

	if( 1200 <= Index_CNT && Index_CNT < 1400)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,-30,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		// RL_Trajectory(0    ,7,10    ,-40    ,40,10    ,200);
        // LL_Trajectory(0    ,10,-40    ,-30    ,-5,10    ,200);
		RL_Trajectory(0    ,7,5    ,-40    ,40,10    ,200);
        LL_Trajectory(0    ,10,-35    ,-30    ,-5,10    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1200);


	}

	if( 1400 <= Index_CNT && Index_CNT < 1600)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 3,-5    ,0    ,0,3    ,200);
        LL_Trajectory(0    ,3,0    ,0    ,0,3    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1400);
	}

	if( 1600 <= Index_CNT && Index_CNT < 1800)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,-30,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,0    ,0    ,0,3    ,200);
        LL_Trajectory(0    ,0,0    ,30    ,0,0    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1600);
	}

	if( 1800 <= Index_CNT && Index_CNT < 2000)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,0    ,0    ,0,0    ,200);
        LL_Trajectory(0    ,0,0    ,80    ,-50,0    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-1800);
	}

	if( 2000 <= Index_CNT && Index_CNT < 2200)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,0,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,0    ,0    ,0,0    ,200);
        LL_Trajectory(0    ,0,110    ,30    ,0,0    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-2000);
	}

	if( 2200 <= Index_CNT && Index_CNT < 2400)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,60,70,0,200);
        LA_Trajectory(0,60,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , 0,0    ,0    ,0,-6    ,200);
        LL_Trajectory(0    ,-3,-25    ,-70    ,65,-3    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-2200);
	}

	if( 2400 <= Index_CNT && Index_CNT < 2600)
	{
		WT_Trajectory(0, 200);

        RA_Trajectory(0,0,0,0,200);
        LA_Trajectory(0,-60,0,0,200);

        NC_Trajectory(0,0,200);
		
		RL_Trajectory(0    , -10,25    ,40    ,-15,-10    ,200);
        LL_Trajectory(0    ,-10,-25    ,-40    ,15,-10    ,200);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-2400);
	}

}

void Pick::Back_Halfstep(const MatrixXd& RFx, int Index_CNT)
{
	if (Index_CNT > RFx.cols() - 1)
    {
        Index_CNT = RFx.cols();
    }


	if( 0 <= Index_CNT && Index_CNT < 50)
	{
		WT_Trajectory(0, 50);

        RA_Trajectory(0,0,0,0,50);
        LA_Trajectory(0,45,0,0,50);

        NC_Trajectory(0,0,50);

		RL_Trajectory(0    , 0,0    ,0    ,0,0    ,50);
        LL_Trajectory(0    ,0,0    ,0    ,0,0    ,50);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT);
	}

	if( 200 <= Index_CNT && Index_CNT < 269)
	{
		WT_Trajectory(0, 69);

        RA_Trajectory(0,0,0,0,69);
        LA_Trajectory(0,-45,0,0,69);

        NC_Trajectory(0,0,69);

		RL_Trajectory(0    , 0,0    ,0    ,0,0    ,69);
        LL_Trajectory(0    ,0,0    ,0    ,0,0    ,69);
    	UPBD_SET_ALL(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th,Ref_RL_th, Ref_LL_th, Index_CNT-200);
	}
;}

void Pick::Huddle(const MatrixXd& RFx, int Index_CNT,double& RL, double& LL, double& RL_2, double& LL_2,double& HS, double& SR)
{

	if (Index_CNT > RFx.cols() - 1)
    {
        Index_CNT = RFx.cols();
    }
    // std::cout << RFx.cols() << std::endl;
	if(1 < Index_CNT && Index_CNT < 12){

		SR =+ 0.2;
	}
	else if(1750 < Index_CNT && Index_CNT < 1801){
		SR =+ -0.04;
	}
	

    if( 250 < Index_CNT && Index_CNT < 401)
	{
		WT_Trajectory(10, 150);
        RA_Trajectory(10,0,-40,0,150);
        LA_Trajectory(30,0,70,0,150);
        NC_Trajectory(0,0,150);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-250);

	}

    else if( 919 < Index_CNT && Index_CNT < 1020)
	{
		WT_Trajectory(10, 50);
        RA_Trajectory(5,0,-20,0,50);
        LA_Trajectory(15,0,35,0,50);
        NC_Trajectory(0,0,50);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-919);

	}


	else if( 1314 < Index_CNT && Index_CNT < 1365)
	{
		WT_Trajectory(-25, 50);
        RA_Trajectory(-45,0,20,0,50);
        LA_Trajectory(-55,0,-35,0,50);
        NC_Trajectory(0,0,50);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-1314);

	}

	else if(1415 < Index_CNT && Index_CNT < 1516)
	{
		WT_Trajectory(5, 100);
        RA_Trajectory(30,0,40,0,100);
        LA_Trajectory(10,0,-70,0,100);
        NC_Trajectory(0,0,100);
		UPBD_SET(Ref_WT_th, Ref_RA_th, Ref_LA_th, Ref_NC_th, Index_CNT-1415);

	}


	

	if(250 < Index_CNT && Index_CNT < 350 )
	{

		RL += 0.22;
		LL += -0.14;

	}

		
	else if(910 < Index_CNT && Index_CNT < 1011 )
	{

		RL += -0.22;
		LL += 0.14;

	}


    else if(1315 < Index_CNT && Index_CNT < 1366 )
	{

		RL += -0.3;
		LL += 0.3;
		LL_2 += 0.2;


	}

	else if(1635 < Index_CNT && Index_CNT < 1686 )
	{
		RL += 0.2;
	}

	else if(1705 < Index_CNT && Index_CNT < 1786 )
	{
		LL += -0.1875;
		RL += 0.0625;
		// RL += -0.1;

	}


	if(1680 < Index_CNT && Index_CNT < 1781 )
	{
		LL_2 += -0.1;
	}
	

	
	// if(1581 < Index_CNT && Index_CNT < 1607 )
	// {
	// 	RL += -0.2;

	// }
	//HS
	if(900 < Index_CNT && Index_CNT < 950 )
	{
		HS += 0.225;

	}
	else if(967 < Index_CNT && Index_CNT < 1018 )
	{
		HS += -0.375;

	}
	else if(1020 < Index_CNT && Index_CNT < 1071 )
	{
		HS += 0.15;

	}



}