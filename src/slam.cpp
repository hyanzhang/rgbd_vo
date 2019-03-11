/******************************************************************
    >File name:slam.cpp
    >Author:Hyan Zhang
    >Mail:hyzhang1210@stu.xjtu.edu.cn
    >Created Time:2018-12-04 21:22:18
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

/*read color image,depth image and it's ID */
FRAME readFrame(int index, ParameterReader &pd);

/*metric of transformation between frames*/
double normOfTransform(cv::Mat rvec, cv::Mat tvec);

enum CHECK_RERSULT
{
    NOT_MATCHED = 0,
    TOO_FAR_AWAY,
    TOO_CLOSE,
    KEYFRAME
};

/*determine a frame is keyFrame or not*/
CHECK_RERSULT checkKeyFrame(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops = false);

/*determine keyFrames nearby current frame are closure loop or not*/
void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);

/*determine random chosen keyFrames are closure loop of current KeyFrame or not*/
void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti);

int main(int argc, char *argv[])
{
    ParameterReader pd; //Parameter handle

    int startIndex = atoi(pd.getValue("start_index").c_str());
    int endIndex = atoi(pd.getValue("end_index").c_str());

    vector<FRAME> keyFrames; //keyFrame list

    cout << "Initinalizing ..." << endl;
    int currIndex = startIndex;
    FRAME currFrame = readFrame(currIndex, pd);
    CAMERA_INTRINSIC_PARAMETERS camera = getCamera(pd);
    double keyframe_threshold = atof(pd.getValue("keyframe_threshold").c_str());
    /******************************************************************
    *Block:construct grough optimization solver
    ******************************************************************/
    typedef g2o::BlockSolver_6_3 SlamBlockSolver;
    typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);

    SlamBlockSolver *blockSolver = new SlamBlockSolver(unique_ptr<SlamBlockSolver::LinearSolverType>(
        linearSolver)); //parameter has to be converted to unique_ptr for new construtor

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
        unique_ptr<SlamBlockSolver>(blockSolver)); //parameter has to be converted to unique_prt for new constructor
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    //add first node(fixed)
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); //initial state
    v->setFixed(true);
    optimizer.addVertex(v);

    //first frame must be the keyFrame
    computeKeyPointsAndDescriptor(currFrame);
    keyFrames.push_back(currFrame);

    CHECK_RERSULT result;
    /******************************************************************
    *Block: loop for gathering keyFrames
    ******************************************************************/
    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++)
    {
        // startTime = chrono::steady_clock::now();
        cout << "Handling " << currIndex << " frame." << endl;
        currFrame = readFrame(currIndex, pd);
        computeKeyPointsAndDescriptor(currFrame);

        result = checkKeyFrame(keyFrames.back(), currFrame, optimizer); //add new vertex and edge connecting currFrame and last frame(all frames)
        switch (result)
        {
        case NOT_MATCHED:
            cout << "Not Enough inliers." << endl;
            break;
        case TOO_FAR_AWAY:
            cout << "Too far away,may be an error." << endl;
            break;
        case TOO_CLOSE:
            cout << "Too close,not a keyframe." << endl;
            break;
        case KEYFRAME:
            cout << "This is a key frame." << endl;
            checkNearbyLoops(keyFrames, currFrame, optimizer); //add edge connecting currFrame(be specifed as keyFrame) and keyFrames which passed loopclosure check(nearby) with currFrame
            checkRandomLoops(keyFrames, currFrame, optimizer); //add edge connecting currFrame(be specifed as keyFrame) and keyFrames which passed loopclosure check(random) with currFrame
            keyFrames.push_back(currFrame);                    //just for loopclosure and cloud points
            break;
        default:
            break;
        }
    }

    /******************************************************************
    *Block:optimization only for keyFrames
    ******************************************************************/
    cout << "Optimizing pose graph..." << optimizer.vertices().size() << endl;
    optimizer.save("../data/result_before.g2o"); //grough without optimizing

    optimizer.initializeOptimization();
    optimizer.optimize(100);
    optimizer.save("../data/result_after.g2o"); //grough with optimizing
    cout << "Saving optimization result cloud." << endl;

    /******************************************************************
    *Block:merge point cloud
    ******************************************************************/
    cout << "Saving point cloud map..." << endl;
    PointCloud::Ptr output(new PointCloud());
    PointCloud::Ptr tmp(new PointCloud());

    pcl::VoxelGrid<PointT> voxel;
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0); //limits come from the precision of rgb-dcamera
    double gridSize = atof(pd.getValue("voxel_grid").c_str());
    voxel.setLeafSize(gridSize, gridSize, gridSize);

    for (auto key_frame : keyFrames)
    {
        g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(key_frame.frameID));
        Eigen::Isometry3d pose = vertex->estimate(); //return pose which had been optimized
        PointCloud::Ptr newCloud = image2PointCloud(key_frame.color, key_frame.depth, camera);
        //filter point cloud respect to z cordinate(depth)
        voxel.setInputCloud(newCloud);
        voxel.filter(*tmp);
        pass.setInputCloud(tmp);
        pass.filter(*newCloud);
        //merge point cloud
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;
        tmp->clear();
        newCloud->clear();
    }
    voxel.setInputCloud(output);
    voxel.filter(*tmp);
    pcl::io::savePCDFile("../data/result.pcd", *tmp);
    cout << "Point cloud result has been saved." << endl;
    return 0;
}

FRAME readFrame(int index, ParameterReader &pd)
{
    FRAME f;
    string rgbPath = pd.getValue("rgb_dir");
    string depthPath = pd.getValue("depth_dir");
    string rgbExtesion = pd.getValue("rgb_extension");
    string depthExtesion = pd.getValue("depth_extension");
    stringstream ss;
    ss << rgbPath << index << rgbExtesion;
    string fileName;
    ss >> fileName; //filename's format "rgbpath[index]rgbExtension" e.g. /home/data/1.png
    f.color = cv::imread(fileName);

    ss.clear(); //clear the buffer
    ss << depthPath << index << depthExtesion;
    ss >> fileName;
    f.depth = cv::imread(fileName, -1);
    f.frameID = index;
    return f;
}

double normOfTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(std::min(2 * CV_PI - cv::norm(rvec), cv::norm(rvec))) + fabs(cv::norm(tvec));
}

CHECK_RERSULT checkKeyFrame(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti, bool is_loops)
{
    static ParameterReader pd;
    static int minInliers = atoi(pd.getValue("min_inliers").c_str());
    static double maxNorm = atof(pd.getValue("max_norm").c_str());
    static double keyframe_threshold = atof(pd.getValue("keyframe_threshold").c_str());
    static double maxNorm_lp = atof(pd.getValue("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getCamera(pd);
    static g2o::RobustKernel *robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
    RESULT_OF_PNP pnpRes = estimateMotion(f1, f2, camera);
    if (pnpRes.inliers < minInliers)
        return NOT_MATCHED;
    double norm = normOfTransform(pnpRes.rvec, pnpRes.tvec);
    if (is_loops == false)
    {
        if (norm >= maxNorm)
            return TOO_FAR_AWAY;
    }
    else
    {
        if (norm >= maxNorm_lp)
            return TOO_FAR_AWAY;
    }
    if (norm <= keyframe_threshold)
        return TOO_CLOSE;
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(f2.frameID); //if call this function for closure loop checking,do not add it to the graph(has been added when check for keyFrame)
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
    edge->vertices()[0] = opti.vertex(f1.frameID);
    edge->vertices()[1] = opti.vertex(f2.frameID);
    edge->setRobustKernel(robustKernel);
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(1, 1) = information(2, 2) = 100;
    information(3, 3) = information(4, 4) = information(5, 5) = 100;
    edge->setInformation(information);
    Eigen::Isometry3d T = cvMat2Eigen(pnpRes.rvec, pnpRes.tvec);
    edge->setMeasurement(T.inverse());
    opti.addEdge(edge);
    return KEYFRAME;
}

void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int nearbyLoops = atoi(pd.getValue("nearby_loops").c_str());
    if (frames.size() <= nearbyLoops)
    {
        for (int i = 0; i < frames.size(); i++)
            checkKeyFrame(frames[i], currFrame, opti, true);
    }
    else
    {
        for (int i = (frames.size() - nearbyLoops); i < frames.size(); i++)
            checkKeyFrame(frames[i], currFrame, opti, true);
    }
}

void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame, g2o::SparseOptimizer &opti)
{
    static ParameterReader pd;
    static int randomLoops = atoi(pd.getValue("random_loops").c_str());
    srand((unsigned int)time(nullptr));
    if (frames.size() <= randomLoops)
    {
        for (int i = 0; i < frames.size(); i++)
            checkKeyFrame(frames[i], currFrame, opti, true);
    }
    else
    {
        for (int i = 0; i < randomLoops; i++)
        {
            int index = rand() % frames.size();
            checkKeyFrame(frames[index], currFrame, opti, true);
        }
    }
}