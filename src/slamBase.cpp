#include "slamBase.h"

using std::cout;
using std::endl;

PointCloud::Ptr image2PointCloud(cv::Mat &color, cv::Mat &depth, CAMERA_INTRINSIC_PARAMETERS &camera)
{
    PointCloud::Ptr cloud(new PointCloud);
    int ncols = depth.cols;
    int nrows = depth.rows;
    PointT point;
    // uchar *pd;
    // uchar *pc;
    int i, j;
    // for (i = 0; i < nrows; i++)
    // {
    // 	pd = depth.ptr<uchar>(i);
    // 	pc = rgb.ptr<uchar>(i);
    // 	for (j = 0; j < ncols; j++)
    // 	{
    // 		if(*pd++ == 0)
    // 			continue;
    // 		point.z = (double)*pd++ / cameraFactor;
    // 		point.x = (j - cameraCx) * point.z / cameraFx;
    // 		point.y = (i - cameraCy) * point.z / cameraFy;
    // 		point.b = *pc++;
    // 		point.g = *pc++;
    // 		point.r = *pc++;
    // 		cloud->points.push_back(point);
    // 	}
    // }
    /*time for traverse:41.6673ms*/
    ushort d;
    for (i = 0; i < nrows; i += 2)
    {
        for (j = 0; j < ncols; j += 2)
        {
            d = depth.ptr<ushort>(i)[j];
            if (d == 0)
                continue;
            point.z = double(d) / camera.scale;
            point.x = (j - camera.cx) * point.z / camera.fx;
            point.y = -(i - camera.cy) * point.z / camera.fy;
            point.b = color.ptr<uchar>(i)[3 * j];
            point.g = color.ptr<uchar>(i)[3 * j + 1];
            point.r = color.ptr<uchar>(i)[3 * j + 2];
            cloud->points.push_back(point);
        }
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    return cloud;
}

cv::Point3f point2dTo3d(cv::Point2f &index, ushort depth, CAMERA_INTRINSIC_PARAMETERS &camera)
{
    cv::Point3f p;
    p.z = double(depth) / camera.scale;
    p.x = (index.x - camera.cx) * p.z / camera.fx;
    p.y = (index.y - camera.cy) * p.z / camera.fy;
    return p;
}

void computeKeyPointsAndDescriptor(FRAME &frame)
{
    static ParameterReader pd;
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detectAndCompute(frame.color, cv::Mat(), frame.keyPoints, frame.descriptor);
}

RESULT_OF_PNP estimateMotion(FRAME &frame1, FRAME &frame2, CAMERA_INTRINSIC_PARAMETERS &camera)
{
    static ParameterReader pd;
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match(frame1.descriptor, frame2.descriptor, matches);
    //    cout << "Match number: " << matches.size() << endl;
    std::vector<cv::DMatch> goodMatches;
    double thresh = atof(pd.getValue("threshold").c_str());
    //    cout << "thresh: " << thresh << endl;
    float minDist = 9999;
    for (auto match : matches)
    {
        if (match.distance < minDist)
            minDist = match.distance;
    }
    //    std::cout << "minDist:" << minDist << std::endl;

    if (minDist < 10)
    {
        minDist = 10;
    }
    for (auto match : matches)
    {
        if (match.distance < minDist * thresh)
            goodMatches.push_back(match);
    }
    RESULT_OF_PNP result;
    if (goodMatches.size() <= 5)
    {
        result.inliers = -1;
        return result;
    }

    //    cout << "Good match number: " << goodMatches.size() << endl;
    std::vector<cv::Point3f> Pts3d;
    std::vector<cv::Point2f> Pts2d;
    cv::Point2f indexTemp;
    ushort d;
    cv::Point3f ptsTemp;
    for (auto goodMatch : goodMatches)
    {
        indexTemp = frame1.keyPoints[goodMatch.queryIdx].pt;
        d = frame1.depth.ptr<ushort>((int)indexTemp.y)[(int)indexTemp.x];
        if (d == 0)
            continue;
        ptsTemp = point2dTo3d(indexTemp, d, camera);
        Pts3d.push_back(ptsTemp);
        Pts2d.push_back(cv::Point2f(frame2.keyPoints[goodMatch.trainIdx].pt));
    }
    if (Pts3d.size() < 4 || Pts2d.size() < 4)
    {
        result.inliers = -1;
        return result;
    }
    double c[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}};
    cv::Mat cameraMat(3, 3, CV_64F, c);
    cv::Mat rvec, tvec, inliners;
    cv::solvePnPRansac(Pts3d, Pts2d, cameraMat, cv::Mat(), rvec, tvec, false, 100, 1.0f, 0.99, inliners,
                       cv::SOLVEPNP_EPNP);
    result.inliers = inliners.rows;
    result.rvec = rvec;
    result.tvec = tvec;
    //    std::cout << "Number of inliers:" << inliners.rows << endl;
    return result;
}

Eigen::Isometry3d cvMat2Eigen(cv::Mat rvec, cv::Mat tvec)
{
    cv::Mat cvMat;
    cv::Rodrigues(rvec, cvMat);
    Eigen::Matrix3d rMat;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            rMat(i, j) = cvMat.at<double>(i, j);
        }
    }
    //    cv::cv2eigen(cvMat, rMat);
    //    Eigen::Vector3d t;
    //    cv::cv2eigen(tvec, t);
    Eigen::AngleAxisd angle(rMat);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T = rMat;
    T(0, 3) = tvec.at<double>(0, 0);
    T(1, 3) = tvec.at<double>(1, 0);
    T(2, 3) = tvec.at<double>(2, 0);
    //    std::cout << "T matrix" << T.matrix() << std::endl;
    return T;
}

PointCloud::Ptr jointPointCloud(PointCloud::Ptr originalCloud, FRAME &newFrame, Eigen::Isometry3d T,
                                CAMERA_INTRINSIC_PARAMETERS &camera)
{
    //merge point cloud
    PointCloud::Ptr newCloud = image2PointCloud(newFrame.color, newFrame.depth, camera);
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*originalCloud, *output, T.matrix());
    *newCloud += *output;
    // voxle grid
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridSize = atof(pd.getValue("voxel_grid").c_str());

    voxel.setLeafSize(gridSize, gridSize, gridSize);
    voxel.setInputCloud(newCloud);
    PointCloud::Ptr temp(new PointCloud());
    voxel.filter(*temp);
    //    return newCloud;
    return temp;
}

CAMERA_INTRINSIC_PARAMETERS getCamera(ParameterReader &pd)
{
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx = std::atof(pd.getValue("cx").c_str());
    camera.cy = std::atof(pd.getValue("cy").c_str());
    camera.fx = std::atof(pd.getValue("fx").c_str());
    camera.fy = std::atof(pd.getValue("fy").c_str());
    camera.scale = std::atof(pd.getValue("scale").c_str());
    return camera;
}
