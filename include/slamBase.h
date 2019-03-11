
/******************************************************************
    >File name:slamBase.h
    >Author:Hyan Zhang
    >Mail:hyzhang1210@xjtu.edu.cn
    >Created Time:2018-12-01 21:20:37
*******************************************************************/
#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

/*Bugs: Eigen/Core header must be places above opencv2/core/eigen.hpp header,
* or there are errors in eigen.hpp file.
*/

#include <opencv2/opencv.hpp>     //for all opencv modules
#include <opencv2/core/eigen.hpp> //for convertion between opencv and eigen

#include <pcl/io/pcd_io.h>         //for file saving
#include <pcl/point_types.h>       //for pointType
#include <pcl/common/transforms.h> //for pointcloud merge
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx;
    double cy;
    double fx;
    double fy;
    double scale;
};

struct FRAME
{
    int frameID;
    cv::Mat color, depth;
    cv::Mat descriptor;
    std::vector<cv::KeyPoint> keyPoints;
};

struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};

class ParameterReader
{
  public:
    std::map<std::string, std::string> data;

    ParameterReader(std::string filePath = "../data/parameters.txt")
    {
        std::ifstream fin(filePath.c_str());
        if (!fin)
        {
            std::cerr << "parameter file does not exit." << std::endl;
            return;
        }

        std::string strTemp;
        int posTemp;
        std::string key;
        std::string value;
        while (!fin.eof())
        {
            getline(fin, strTemp);
            if (strTemp[0] == '#')
                continue;
            posTemp = strTemp.find('=');
            if (posTemp == -1)
                continue;
            key = strTemp.substr(0, posTemp);
            value = strTemp.substr(posTemp + 1, strTemp.length());
            data[key] = value;
            if (!fin.good())
                break;
        }
    }

    std::string getValue(std::string key)
    {
        auto iter = data.find(key);
        if (iter == data.end())
        {
            std::cerr << "Paraneter name " << key << " not found." << std::endl;
            return "NOT_FOUND";
        }
        return iter->second;
    }
};

PointCloud::Ptr image2PointCloud(cv::Mat &color, cv::Mat &depth, CAMERA_INTRINSIC_PARAMETERS &camera);

cv::Point3f point2dTo3d(cv::Point2f &index, ushort depth, CAMERA_INTRINSIC_PARAMETERS &camera);

void computeKeyPointsAndDescriptor(FRAME &frame); //now we just compute ORB feature

RESULT_OF_PNP estimateMotion(FRAME &frame1, FRAME &frame2, CAMERA_INTRINSIC_PARAMETERS &camera);

//cvMat -> eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat rvec, cv::Mat tvec);

// //jointPointCloud
PointCloud::Ptr jointPointCloud(PointCloud::Ptr originalCloud, FRAME &newFrame, Eigen::Isometry3d T,
                                CAMERA_INTRINSIC_PARAMETERS &camera);

CAMERA_INTRINSIC_PARAMETERS getCamera(ParameterReader &pd);
