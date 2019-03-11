#include "slamBase.h"
#include <chrono>

using namespace cv;
using std::cout;
using std::endl;
using std::vector;
int main(int argc, char const *argv[])
{
    //import object's color image and depth image
    Mat color1 = imread("../data/color1.png");
    Mat depth1 = imread("../data/depth1.png", -1);
    Mat color2 = imread("../data/color2.png");
    CAMERA_INTRINSIC_PARAMETERS camera = {325.5, 253.5, 518.0, 519.0, 1000.0};
    FRAME frame1, frame2;
    frame1.color = color1;
    frame1.depth = depth1;
    frame2.color = color2;
    computeKeyPointsAndDescriptor(frame1);
    computeKeyPointsAndDescriptor(frame2);

    RESULT_OF_PNP result = estimateMotion(frame1, frame2, camera);
    //construct SIFT feature detector and descriptor
    // Ptr<ORB> _detector = ORB::create();
    // //detect keypoints
    // vector<KeyPoint> kp1, kp2;
    // _detector->detect(color1, kp1);
    // _detector->detect(color2, kp2);
    // //show keypoints
    // Mat imgTemp;
    // drawKeypoints(color2, kp2, imgTemp, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // imshow("keypoints", imgTemp);
    // //compute descriptor
    // Mat descriptor1, descriptor2;
    // _detector->compute(color1, kp1, descriptor1);
    // _detector->compute(color2, kp2, descriptor2);
    // //match descriptors
    // vector<DMatch> matches;
    // // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);//match time: 0.012s
    // // BFMatcher matcher(NORM_HAMMING);
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING); //match time:0.0056s
    // std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    // matcher->match(descriptor1, descriptor2, matches);
    // std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> timeForMatch = std::chrono::duration_cast<std::chrono::duration<double>>(time2 - time1);
    // std::cout << "Match time: " << timeForMatch.count() << std::endl;
    // Mat imgTempMatches;
    // drawMatches(color1, kp1, color2, kp2, matches, imgTempMatches);
    // imshow("matches", imgTempMatches);
    // //filt matches based on distance of matched keypoint
    // vector<DMatch> goodMatches;
    // double minDist = 9999;
    // for (auto match : matches)
    // {
    //     if (match.distance < minDist)
    //         minDist = match.distance;
    // }

    // for (auto match : matches)
    // {
    //     if (match.distance < 10 * minDist)
    //         goodMatches.push_back(match);
    // }
    // Mat imgTempGoodMatches;
    // drawMatches(color1, kp1, color2, kp2, goodMatches, imgTempGoodMatches);
    // imshow("good matches", imgTempGoodMatches);
    // //estimate camera's translation and orientation
    // /*1.get the correspondence of 2D image points and 3D object points*/
    // vector<Point3f> objPts;
    // vector<Point2f> imgPts;
    // CAMERA_INTRINSIC_PARAMETERS camera = {325.5, 253.5, 518.0, 519.0, 1000.0};
    // Point2f objPtTemp;
    // for (auto match : goodMatches)
    // {
    //     objPtTemp = kp1[match.queryIdx].pt;
    //     ushort d = depth1.ptr<ushort>((int)objPtTemp.y)[(int)objPtTemp.x];
    //     //ATTENTION:depth of pixel is queried by colum and row.
    //     if (d == 0)
    //         continue;
    //     objPts.push_back(point2dTo3d(objPtTemp, d, camera));
    //     imgPts.push_back(kp2[match.trainIdx].pt);
    // }
    // cout << "Points num: " << objPts.size() << " , " << imgPts.size() << endl;
    // /*2.solve PnP with opencv*/
    // double m[3][3] = {
    //     {camera.fx, 0, camera.cx},
    //     {0, camera.fy, camera.cy},
    //     {0, 0, 1}};
    // Mat cameraMat(3, 3, CV_64F, m); //
    // Mat rvec, tvec, inliers;
    // // bool ok=solvePnPRansac(objPts, imgPts, cameraMat, Mat(), rvec, tvec,false,100,10,0.95,inliers,SOLVEPNP_EPNP);
    // bool ok = solvePnP(objPts, imgPts, cameraMat, Mat(), rvec, tvec, false, SOLVEPNP_EPNP);
    cout << "R= " << result.rvec << endl;
    cout << "t= " << result.tvec << endl;
    // cout << "PnP is OK? " << ok << endl;
    // waitKey(0);
    // destroyAllWindows();

    return 0;
}
