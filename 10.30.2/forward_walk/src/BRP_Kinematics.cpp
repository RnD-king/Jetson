#include "BRP_Kinematics.hpp"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix4d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace BRP_Kinematics{

// th: [6x1], link: [7x1], PR: [6x1] (x, y, z, roll, pitch, yaw)
void BRP_RL_FK(const VectorXd& th, const VectorXd& link, VectorXd& PR){

    double t1 = th(0), t2 = th(1), t3 = th(2), t4 = th(3), t5 = th(4), t6 = th(5);
    double L0 = link(0), L1 = link(1), L2 = link(2), L3 = link(3), L4 = link(4), L5 = link(5), L6 = link(6);

    double c1 = cos(t1), c2 = cos(t2), c3 = cos(t3), c4 = cos(t4), c5 = cos(t5), c6 = cos(t6);
    double s1 = sin(t1), s2 = sin(t2), s3 = sin(t3), s4 = sin(t4), s5 = sin(t5), s6 = sin(t6);

    // x, y, z position setup
    PR(0) = L5 * s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))- L4 * s4 * (c1 * c3 - s1 * s2 * s3)
          - L5 * c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2))- L3 * c1 * s3 - L4 * c4 * (c1 * s3 + c3 * s1 * s2)- L2 * s1 * s2
          - L6 * c6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3)))
          - L3 * c3 * s1 * s2 - L6 * c2 * s1 * s6;

    PR(1) = L2 * c1 * s2 - L4 * c4 * (s1 * s3 - c1 * c3 * s2) - L4 * s4 * (c3 * s1 + c1 * s2 * s3)
          - L5 * c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2))- L0
          + L5 * s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3)) - L3 * s1 * s3
          - L6 * c6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3)))
          + L3 * c1 * c3 * s2 + L6 * c1 * c2 * s6;

    PR(2) = L6 * s2 * s6 - L2 * c2 - L6 * c6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3))
          - L3 * c2 * c3 - L1 - L5 * c5 * (c2 * c3 * c4 - c2 * s3 * s4) + L5 * s5 * (c2 * c3 * s4 + c2 * c4 * s3) - L4 * c2 * c3 * c4 + L4 * c2 * s3 * s4;

    // Orientation Calculation (n, o, a vector)
    double nx = -c5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3)) - s5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2));
    double ny = -c5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3)) - s5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2));
    double nz = -c5 * (c2 * c3 * s4 + c2 * c4 * s3) - s5 * (c2 * c3 * c4 - c2 * s3 * s4);

    double ox = s6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))) - c2 * c6 * s1;
    double oy = s6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3))) + c1 * c2 * c6;
    double oz = c6 * s2 + s6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3));

    double ax = c6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))) + c2 * s1 * s6;
    double ay = c6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3))) - c1 * c2 * s6;
    double az = c6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3)) - s2 * s6;

    // RPY 오일러 각
    PR(5) = atan2(ny, nx);                                                                     // Roll (z축)
    PR(4) = atan2(-nz, cos(PR(5)) * nx + sin(PR(5)) * ny);                                     // Pitch (y축)
    PR(3) = atan2(sin(PR(5)) * ax - cos(PR(5)) * ay, -sin(PR(5)) * ox + cos(PR(5)) * oy);      // Yaw (x축)
}


void BRP_RL_IK(const VectorXd& target_PR, const VectorXd& init_theta, const VectorXd& link, VectorXd& IK_theta)
{    
    const int dof = 6;
    Eigen::VectorXd th = init_theta;
    Eigen::VectorXd PR(dof), old_PR(dof), F(dof), old_Q(dof);
    Eigen::MatrixXd J(dof, dof), Inv_J(dof, dof), New_PR(dof, dof);
    Eigen::VectorXd New_Q4J(dof);
    double del_Q = 0.0001, ERR = 0.0;
    int iter, i, j, k;
    double sum = 0.0;

    for (iter = 0; iter < 100; ++iter){
        old_Q = th;
        BRP_RL_FK(th, link, PR);
        F = target_PR - PR; // Error_vector
        ERR = F.norm();

        if (ERR < 0.0001){
            IK_theta = th;
            break;
        }
        else if (iter == 99){
            IK_theta = init_theta;
            break;
        }

        old_PR = PR;

        // Numerical Jacobian 
        for (i = 0; i < dof; ++i){
            New_Q4J = old_Q;
            New_Q4J(i) += del_Q;
            BRP_RL_FK(New_Q4J, link, PR);
            New_PR.col(i) = PR;
        }

        for (i = 0; i < dof; ++i)
            for (j = 0; j < dof; ++j)
                J(i, j) = (New_PR(i, j) - old_PR(i)) / del_Q;

        // Inverse Matrix (Using Eigen)
        Inv_J = J.inverse();

        // Joint Angle Update
        for (k = 0; k < dof; ++k){
            double sum = 0.0;
            for (j = 0; j < dof; ++j)
                sum += Inv_J(k, j) * F(j);
            th(k) = old_Q(k) + sum;
        }

        // Knee, Joint3 음수 방지 
        if (th(3) < 0) th(3) = -th(3);
    }
}

void BRP_LL_FK(const VectorXd& th, const VectorXd& link, VectorXd& PR)
{
    double t1 = th(0), t2 = th(1), t3 = th(2), t4 = th(3), t5 = th(4), t6 = th(5);
    double L0 = link(0), L1 = link(1), L2 = link(2), L3 = link(3), L4 = link(4), L5 = link(5), L6 = link(6);

    double c1 = cos(t1), c2 = cos(t2), c3 = cos(t3), c4 = cos(t4), c5 = cos(t5), c6 = cos(t6);
    double s1 = sin(t1), s2 = sin(t2), s3 = sin(t3), s4 = sin(t4), s5 = sin(t5), s6 = sin(t6);

    // x, y, z position setup
    PR(0) = L5 * s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))- L4 * s4 * (c1 * c3 - s1 * s2 * s3)
          - L5 * c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2))- L3 * c1 * s3 - L4 * c4 * (c1 * s3 + c3 * s1 * s2)- L2 * s1 * s2
          - L6 * c6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3)))
          - L3 * c3 * s1 * s2 - L6 * c2 * s1 * s6;

    PR(1) = L0 - L4 * c4 * (s1 * s3 - c1 * c3 * s2) - L4 * s4 * (c3 * s1 + c1 * s2 * s3) - L5 * c5 * (s4 * (c3 * s1 + c1 * s2 * s3)
          + c4 * (s1 * s3 - c1 * c3 * s2)) + L2 * c1 * s2 + L5 * s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3))
          - L3 * s1 * s3 - L6 * c6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2)
          - c4 * (c3 * s1 + c1 * s2 * s3))) + L3 * c1 * c3 * s2 + L6 * c1 * c2 * s6;

    PR(2) = L6 * s2 * s6 - L2 * c2 - L6 * c6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3))
          - L3 * c2 * c3 - L1 - L5 * c5 * (c2 * c3 * c4 - c2 * s3 * s4) + L5 * s5 * (c2 * c3 * s4 + c2 * c4 * s3) - L4 * c2 * c3 * c4 + L4 * c2 * s3 * s4;

    // Orientation Calculation (n, o, a vector)
    double nx = -c5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3)) - s5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2));
    double ny = -c5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3)) - s5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2));
    double nz = -c5 * (c2 * c3 * s4 + c2 * c4 * s3) - s5 * (c2 * c3 * c4 - c2 * s3 * s4);

    double ox = s6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))) - c2 * c6 * s1;
    double oy = s6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3))) + c1 * c2 * c6;
    double oz = c6 * s2 + s6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3));

    double ax = c6 * (c5 * (s4 * (c1 * c3 - s1 * s2 * s3) + c4 * (c1 * s3 + c3 * s1 * s2)) - s5 * (s4 * (c1 * s3 + c3 * s1 * s2) - c4 * (c1 * c3 - s1 * s2 * s3))) + c2 * s1 * s6;
    double ay = c6 * (c5 * (s4 * (c3 * s1 + c1 * s2 * s3) + c4 * (s1 * s3 - c1 * c3 * s2)) - s5 * (s4 * (s1 * s3 - c1 * c3 * s2) - c4 * (c3 * s1 + c1 * s2 * s3))) - c1 * c2 * s6;
    double az = c6 * (c5 * (c2 * c3 * c4 - c2 * s3 * s4) - s5 * (c2 * c3 * s4 + c2 * c4 * s3)) - s2 * s6;

    // RPY 오일러 각
    PR(5) = atan2(ny, nx);                                                                 // Roll (z축)
    PR(4) = atan2(-nz, cos(PR(5)) * nx + sin(PR(5)) * ny);                                 // Pitch (y축)
    PR(3) = atan2(sin(PR(5)) * ax - cos(PR(5)) * ay, -sin(PR(5)) * ox + cos(PR(5)) * oy);  // Yaw (x축)
}


void BRP_LL_IK(const VectorXd& target_PR, const VectorXd& init_theta, const VectorXd& link, VectorXd& IK_theta)
{    
    const int dof = 6;
    Eigen::VectorXd th = init_theta;
    Eigen::VectorXd PR(dof), old_PR(dof), F(dof), old_Q(dof);
    Eigen::MatrixXd J(dof, dof), Inv_J(dof, dof), New_PR(dof, dof);
    Eigen::VectorXd New_Q4J(dof);
    double del_Q = 0.0001, ERR = 0.0;
    int iter, i, j, k;
    double sum = 0.0;

    for (iter = 0; iter < 100; ++iter){
        old_Q = th;
        BRP_LL_FK(th, link, PR);
        F = target_PR - PR; // Error_vector
        ERR = F.norm();

        if (ERR < 0.0001){
            IK_theta = th;
            break;
        }
        else if (iter == 99){
            IK_theta = init_theta;
            break;
        }

        old_PR = PR;

        // Numerical Jacobian 
        for (i = 0; i < dof; ++i){
            New_Q4J = old_Q;
            New_Q4J(i) += del_Q;
            BRP_LL_FK(New_Q4J, link, PR);
            New_PR.col(i) = PR;
        }

        for (i = 0; i < dof; ++i)
            for (j = 0; j < dof; ++j)
                J(i, j) = (New_PR(i, j) - old_PR(i)) / del_Q;

        // Inverse Matrix (Using Eigen)
        Inv_J = J.inverse();

        // Joint Angle Update
        for (k = 0; k < dof; ++k){
            double sum = 0.0;
            for (j = 0; j < dof; ++j)
                sum += Inv_J(k, j) * F(j);
            th(k) = old_Q(k) + sum;
        }

        // Knee, Joint3 음수 방지 
        if (th(3) < 0) th(3) = -th(3);
    }
}

}