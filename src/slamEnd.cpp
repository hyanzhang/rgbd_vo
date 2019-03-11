/******************************************************************
    >File name:slamEnd.cpp
    >Author:Hyan Zhang
    >Mail:hyzhang1210@stu.xjtu.edu.cn
    >Created Time:2018-12-04 14:43:45
*******************************************************************/

#include "slamBase.h"
#include <sstream>
#include <chrono>
#include <memory>
using namespace std;

#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

//test performance
#include <chrono>

FRAME readFrame(int index, ParameterReader &pd);

double normOfTransform(cv::Mat rvec, cv::Mat tvec);

CAMERA_INTRINSIC_PARAMETERS getCamera(ParameterReader &pd);

int main(int argc, char *argv[])
{
    ParameterReader pd;
    int startIndex = std::atoi(pd.getValue("start_index").c_str());
    int endIndex = std::atoi(pd.getValue("end_index").c_str());

    cout << "Initinalizing ..." << endl;
    int currIndex = startIndex;
    FRAME lastFrame = readFrame(currIndex, pd);
    CAMERA_INTRINSIC_PARAMETERS camera = getCamera(pd);
    computeKeyPointsAndDescriptor(lastFrame);
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.color, lastFrame.depth, camera);
    // pcl::visualization::CloudViewer viewer("viewer");
    int minInliers = std::atoi(pd.getValue("min_inliers").c_str());
    double maxNorm = std::atof(pd.getValue("max_norm").c_str());
    //    std::cout << "maxNorm" << maxNorm << std::endl;
    // chrono::steady_clock::time_point startTime;
    // chrono::duration<double> timeUsed;

    typedef g2o::BlockSolver_6_3 SlamBlockSolver;
    typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);

    SlamBlockSolver *blockSolver = new SlamBlockSolver(unique_ptr<SlamBlockSolver::LinearSolverType>(linearSolver)); //parameter has to be converted to unique_ptr for new construtor

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<SlamBlockSolver>(blockSolver)); //parameter has to be converted to unique_prt for new constructor
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);
    //add first node(fixed)
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed(true);
    optimizer.addVertex(v);
    int lastIndex = currIndex;
    //Optimize:declare variable out of loop
    FRAME *currFrame = new FRAME;
    RESULT_OF_PNP *result = new RESULT_OF_PNP;
    double norm;
    Eigen::Isometry3d T;
    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++)
    {
        // startTime = chrono::steady_clock::now();
        cout << "Handling " << lastIndex << " frame." << endl;
        *currFrame = readFrame(currIndex, pd);
        computeKeyPointsAndDescriptor(*currFrame);
        *result = estimateMotion(lastFrame, *currFrame, camera);
        if (result->inliers < minInliers)
        {
            continue;
        }
        norm = normOfTransform(result->rvec, result->rvec);
        if (norm > maxNorm)
        {
            continue;
        }
        T = cvMat2Eigen(result->rvec, result->tvec);
        //add rest of verteces(non_fixed)
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(currIndex);
        v->setEstimate(Eigen::Isometry3d::Identity());
        optimizer.addVertex(v);

        //add edges
        g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
        edge->vertices()[0] = optimizer.vertex(lastIndex);
        edge->vertices()[1] = optimizer.vertex(currIndex);
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(1, 1) = information(2, 2) = 100;
        information(3, 3) = information(4, 4) = information(5, 5) = 100;
        edge->setInformation(information);
        edge->setMeasurement(T);
        optimizer.addEdge(edge);
        // cloud = jointPointCloud(cloud, *currFrame, T, camera);
        // viewer.showCloud(cloud);
        lastFrame = *currFrame;
        lastIndex = currIndex;
        // timeUsed = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - startTime);
        // std::cout << "Timeused per frame: " << timeUsed.count() << std::endl;
    }
    delete currFrame;
    delete result;
    optimizer.save("../data/result_before.g2o");
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    optimizer.save("../data/result_after.g2o");
    // pcl::io::savePCDFile("../data/rgbd_slam.pcd", *cloud);
    std::cout << "Saving point cloud." << std::endl;
    return 0;
}

FRAME readFrame(int index, ParameterReader &pd)
{
    FRAME f;
    string rgbPath = pd.getValue("rgb_dir");
    string depthPath = pd.getValue("depth_dir");
    string rgbExtesion = pd.getValue("rgb_extension");
    string depthExtesion = pd.getValue("depth_extension");
    std::stringstream ss;
    ss << rgbPath << index << rgbExtesion;
    string fileName;
    ss >> fileName;
    f.color = cv::imread(fileName);

    ss.clear();

    ss << depthPath << index << depthExtesion;
    ss >> fileName;
    f.depth = cv::imread(fileName, -1);
    return f;
}

double normOfTransform(cv::Mat rvec, cv::Mat tvec)
{
    //    std::cout << "Norm:" << fabs(std::min(2 * CV_PI - cv::norm(rvec), cv::norm(rvec))) + fabs(cv::norm(tvec))<< std::endl;
    return fabs(std::min(2 * CV_PI - cv::norm(rvec), cv::norm(rvec))) + fabs(cv::norm(tvec));
}
