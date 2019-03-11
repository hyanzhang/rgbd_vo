#include "slamBase.h"

using namespace cv;
int main(int argc, char const *argv[])
{
    Mat color1 = imread("../data/color1.png");
    Mat depth1 = imread("../data/depth1.png", -1);
    Mat color2 = imread("../data/color2.png");
    Mat depth2 = imread("../data/depth2.png", -1);
    CAMERA_INTRINSIC_PARAMETERS camera = {325.5, 253.5, 518.0, 519.0, 1000.0};
    FRAME frame1, frame2;
    frame1.color = color1;
    frame1.depth = depth1;
    frame2.color = color2;
    frame2.depth = depth2;
    computeKeyPointsAndDescriptor(frame1);
    computeKeyPointsAndDescriptor(frame2);
    RESULT_OF_PNP result = estimateMotion(frame1, frame2, camera);
    //use my own slambasse lib


    Mat rMat;
    Rodrigues(result.rvec, rMat); //rotate vector -> rotate mattrix
    Eigen::Matrix3d R;
    cv2eigen(rMat, R);

    Eigen::Vector3d t;
    cv2eigen(result.tvec, t);
    Eigen::AngleAxisd r_vec(R);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    T.rotate(r_vec);
    T.pretranslate(t);
    cout << "T: " << T.matrix()<< endl;
    PointCloud::Ptr cloud1 = image2PointCloud(color1, depth1, camera);
    PointCloud::Ptr cloud2 = image2PointCloud(color2, depth2, camera);
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*cloud1, *output, T.matrix());
    *output += *cloud2;
    pcl::io::savePCDFile("../data/cloud.pcd", *output);
    PointCloud::Ptr cloud=jointPointCloud(cloud1,frame2,T,camera);
    pcl::io::savePCDFile("../data/cloud1.pcd",*cloud);
    cout << "Point cloud saved." << endl;
    return 0;
}