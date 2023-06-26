
#include <iostream>
#include <string>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/csv.h"

int main(int argc, char* argv[]) {

//    std::string res_path = "/home/qjy/Initialization/pro_pro/";
//    std::string res_path = "/home/qjy/Initialization/MH_01_analysis/";
    std::string res_path = "/home/qjy/Research/build-VISLAM-unknown-Debug/";
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> proposed_result  = csv::read<double>(res_path+"testing_ours.csv");
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> iterative_result = csv::read<double>(res_path+"testing_iterative.csv");
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> BSpline_result   = csv::read<double>(res_path+"testing_BSpline.csv");
    std::cout << proposed_result.rows() << std::endl;
    std::cout << iterative_result.rows() << std::endl;
    std::cout << BSpline_result.rows() << std::endl;
    Eigen::Vector4d proposed_error;
    Eigen::Vector4d iterative_error;
    Eigen::Vector4d BSpline_error;
    proposed_error.setZero();
    iterative_error.setZero();
    BSpline_error.setZero();
    double tmp;
    double proposed_sovle_time = 0;
    double iterative_sovle_time = 0;
    double BSpline_sovle_time = 0;
    for(int i = 0; i<proposed_result.rows(); i++)
    {
        proposed_error(0) += proposed_result(i,3);// s
        proposed_error(1) += proposed_result(i,4);// bg
        proposed_error(2) += proposed_result(i,5);// ba
        proposed_error(3) += proposed_result(i,6);// g
        tmp = proposed_result(i,0)/10e6;// ms
        proposed_sovle_time += tmp;
    }
    proposed_error = proposed_error/proposed_result.rows();
    proposed_sovle_time /= proposed_result.rows();

    for(int i = 0; i<iterative_result.rows(); i++)
    {
        iterative_error(0) += iterative_result(i,3);// s
        iterative_error(1) += iterative_result(i,4);// bg
        iterative_error(2) += iterative_result(i,5);// ba
        iterative_error(3) += iterative_result(i,6);// g
        tmp = iterative_result(i,0)/10e6;// ms
        iterative_sovle_time += tmp;
    }
    iterative_error = iterative_error/iterative_result.rows();
    iterative_sovle_time /= iterative_result.rows();

    for(int i = 0; i<BSpline_result.rows(); i++)
    {
        BSpline_error(0) += BSpline_result(i,3);// s
        BSpline_error(1) += BSpline_result(i,4);// bg
        BSpline_error(2) += BSpline_result(i,5);// ba
        BSpline_error(3) += BSpline_result(i,6);// g
        tmp = BSpline_result(i,0)/10e6;// ms
        BSpline_sovle_time += tmp;
    }
    BSpline_error = BSpline_error/BSpline_result.rows();
    BSpline_sovle_time /= BSpline_result.rows();

    std::cout << "proposed_solve_time  : "<< proposed_sovle_time << std::endl;
    std::cout << "proposed_error :"<< std::endl << proposed_error.transpose() << std::endl;
    std::cout << "iterative_solve_time : "<< iterative_sovle_time << std::endl;
    std::cout << "iterative_error :"<< std::endl << iterative_error.transpose() << std::endl;
    std::cout << "BSpline_solve_time : "<< BSpline_sovle_time << std::endl;
    std::cout << "BSpline_error :"<< std::endl << BSpline_error.transpose() << std::endl;
    return 0;
}
