#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
using namespace Eigen;
using namespace std;

namespace BRP_Kinematics{

    void BRP_RL_FK(const Eigen::VectorXd& th, const Eigen::VectorXd& link, Eigen::VectorXd& PR);
    void BRP_LL_FK(const Eigen::VectorXd& th, const Eigen::VectorXd& link, Eigen::VectorXd& PR);

    void BRP_RL_IK(const Eigen::VectorXd& target_PR, const Eigen::VectorXd& theta, const Eigen::VectorXd& link, Eigen::VectorXd& IK_theta);
    void BRP_LL_IK(const Eigen::VectorXd& target_PR, const Eigen::VectorXd& theta, const Eigen::VectorXd& link, Eigen::VectorXd& IK_theta);

}