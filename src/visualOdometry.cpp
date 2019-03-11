/******************************************************************
    >File name:visualOdometry.cpp
    >Author:Hyan Zhang
    >Mail:hyzhang1210@stu.xjtu.edu.cn
    >Created Time:2018-12-01 23:48:13
*******************************************************************/
#include "slamBase.h"
#include <sstream>

using std::string;

FRAME readFrame(int index, ParameterReader &pd);

double normOfTransform(cv::Mat rvec, cv::Mat tvec);

CAMERA_INTRINSIC_PARAMETERS getCamera();

int main(int argc, char *argv[]) {
    ParameterReader pd;
    int startIndex = std::atoi(pd.getValue("start_index").c_str());
    int endIndex = std::atoi(pd.getValue("end_index").c_str());

    cout << "Initinalizing ..." << endl;
    int currIndex = startIndex;
    FRAME lastFrame = readFrame(currIndex, pd);
    CAMERA_INTRINSIC_PARAMETERS camera = getCamera(pd);
    computeKeyPointsAndDescriptor(lastFrame);
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.color, lastFrame.depth, camera);
    pcl::visualization::CloudViewer viewer("viewer");
    int minInliers = std::atoi(pd.getValue("min_inliers").c_str());
    double maxNorm = std::atof(pd.getValue("max_norm").c_str());
//    std::cout << "maxNorm" << maxNorm << std::endl;
    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++) {
        FRAME currFrame = readFrame(currIndex, pd);
        computeKeyPointsAndDescriptor(currFrame);
        RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);
        if (result.inliers < minInliers) {
            continue;
        }
        double norm = normOfTransform(result.rvec, result.rvec);
        if (norm > maxNorm) {
            continue;
        }
        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
        cloud = jointPointCloud(cloud, currFrame, T, camera);
        viewer.showCloud(cloud);
        lastFrame = currFrame;
    }
    pcl::io::savePCDFile("../data/rgbd_slam.pcd", *cloud);
    std::cout << "Saving point cloud." << std::endl;
    return 0;
}

FRAME readFrame(int index, ParameterReader &pd) {
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

double normOfTransform(cv::Mat rvec, cv::Mat tvec) {
//    std::cout << "Norm:" << fabs(std::min(2 * CV_PI - cv::norm(rvec), cv::norm(rvec))) + fabs(cv::norm(tvec))<< std::endl;
    return fabs(std::min(2 * CV_PI - cv::norm(rvec), cv::norm(rvec))) + fabs(cv::norm(tvec));
}

