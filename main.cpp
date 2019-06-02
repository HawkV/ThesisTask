
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
 
#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <cfloat>
#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <PoissonRecon.h>

using namespace cv;
using std::vector;

struct dataType { Point3d point; int red; int green; int blue; };
typedef dataType SpacePoint;
vector<SpacePoint> pointCloud;

function writeToPLY(vector<SpacePoint> pointCloud) 
{
    ofstream outfile("pointcloud.ply");
  
    outfile << "ply\n" << "format ascii 1.0\n" << "comment VTK generated PLY File\n";
    outfile << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << pointCloud.size() << "\n";
    outfile << "property float x\n" << "property float y\n" << "property float z\n" << "element face 0\n";
    outfile << "property list uchar int vertex_indices\n" << "end_header\n";
  
    for (int i = 0; i < pointCloud.size(); i++)
    {
        Point3d point = pointCloud.at(i).point;
        outfile << point.x << " ";
        outfile << point.y << " ";
        outfile << point.z << " ";
        outfile << "\n";
    }
    outfile.close();
}	

const float nn_match_ratio = 0.7f;     
const float keypoint_diameter = 15.0f;

static void drawEpipolarLines(const std::string& title,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<KeyPoint> points1,
                const std::vector<KeyPoint> points2,
                const float inlierDistance = -1)
{
  Mat F = cv::findFundamentalMat(points1,points2, FM_RANSAC);
 
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<KeyPoint,3> > epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        continue;
      }
    }
    /*
     * Эпиполярные прямые первого изображения рисуются на втором и наоборот
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
  }
  cv::imshow(title, outImg);
  cv::waitKey(1);
 
 
}
 
template <typename T>
static float distancePointLine(const KeyPoint point, const cv::Vec<KeyPoint,3>& line)
{
  //Прямая выглядит как a*x + b*y + c = 0
  return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

class FundamentalMatrixEightPointEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix3d M_t;
  static const int kMinNumSamples = 8;

  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                   const std::vector<Y_t>& points2);
};

std::vector<FundamentalMatrixEightPointEstimator::M_t>
FundamentalMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  std::vector<X_t> normed_points1;
  std::vector<Y_t> normed_points2;
  Eigen::Matrix3d points1_norm_matrix;
  Eigen::Matrix3d points2_norm_matrix;
  CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
  CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

  Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
  for (size_t i = 0; i < points1.size(); ++i) {
    cmatrix.block<1, 3>(i, 0) = normed_points1[i].homogeneous();
    cmatrix.block<1, 3>(i, 0) *= normed_points2[i].x();
    cmatrix.block<1, 3>(i, 3) = normed_points1[i].homogeneous();
    cmatrix.block<1, 3>(i, 3) *= normed_points2[i].y();
    cmatrix.block<1, 3>(i, 6) = normed_points1[i].homogeneous();
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(
      cmatrix, Eigen::ComputeFullV);
  const Eigen::VectorXd cmatrix_nullspace = cmatrix_svd.matrixV().col(8);
  const Eigen::Map<const Eigen::Matrix3d> ematrix_t(cmatrix_nullspace.data());

  Eigen::JacobiSVD<Eigen::Matrix3d> fmatrix_svd(
      ematrix_t.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = fmatrix_svd.singularValues();
  singular_values(2) = 0.0;
  const Eigen::Matrix3d F = fmatrix_svd.matrixU() *
                            singular_values.asDiagonal() *
                            fmatrix_svd.matrixV().transpose();

  const std::vector<M_t> models = {points2_norm_matrix.transpose() * F *
                                   points1_norm_matrix};
  return models;
}


int main(int argc, char ** argv){
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
 
    vector<KeyPoint> keypoints1, keypoints2;
 
    // Добавляем каждый пиксель к списку особых точек обоих изображений
    for (float xx = keypoint_diameter; xx < img1.size().width - keypoint_diameter; xx++) {
        for (float yy = keypoint_diameter; yy < img1.size().height - keypoint_diameter; yy++) {
            keypoints1.push_back(KeyPoint(xx, yy, keypoint_diameter));
            keypoints2.push_back(KeyPoint(xx, yy, keypoint_diameter));
        }
    }
 
    Mat desc1, desc2;
 
    Ptr<cv::xfeatures2d::DAISY> descriptor_extractor = cv::xfeatures2d::DAISY::create();
 
    // дескрипторы DAISY 
    descriptor_extractor->compute(img1, keypoints1, desc1);
    descriptor_extractor->compute(img2, keypoints2, desc2);
 
    vector <vector<DMatch> > matches;
 
   if ( desc1.empty() )
      cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);    
 
   if ( desc2.empty() )
      cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
   
    std::cout << "test2";
	
    FlannBasedMatcher flannmatcher;
    flannmatcher.add(desc1);
    flannmatcher.train();
    flannmatcher.knnMatch(desc2, matches, 2);
 
    // Фильтрация ложных соответствий по критерию Lowe
    int                 num_good = 0;
    vector<KeyPoint>    matched1, matched2;
    vector<DMatch>      good_matches;
 
    for (int i = 0; i < matches.size(); i++) {
        DMatch first  = matches[i][0];
        DMatch second = matches[i][1];
 
        if (first.distance < nn_match_ratio * second.distance) {
            matched1.push_back(keypoints1[first.trainIdx]);
            matched2.push_back(keypoints2[first.queryIdx]);
            good_matches.push_back(DMatch(num_good, num_good, 0));
            num_good++;
        }
    }
	
  float f  = atof(argv[2]),
        cx = atof(argv[3]), cy = atof(argv[4]);
	
  Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
	
  bool is_projective = true;
 
  vector<Mat> Rs_est, ts_est, points3d_estimated;
  reconstruct(std::vector<String> {argv[1], argv[2]}, Rs_est, ts_est, K, points3d_estimated, is_projective);

  viz::Viz3d window("Coordinate Frame");
             window.setWindowSize(Size(500,500));
             window.setWindowPosition(Point(150,150));
             window.setBackgroundColor(); 
	
  cout << "Recovering points  ... ";
  vector<Vec3f> point_cloud_est;
	
  for (int i = 0; i < points3d_estimated.size(); ++i)
    point_cloud_est.push_back(Vec3f(points3d_estimated[i]));
	
  cout << "[DONE]" << endl;
  cout << "Recovering cameras ... ";
  vector<Affine3d> path;
	
  for (size_t i = 0; i < Rs_est.size(); ++i)
    path.push_back(Affine3d(Rs_est[i],ts_est[i]));
	
  cout << "[DONE]" << endl;
	
  if ( point_cloud_est.size() > 0 )
  {
    cout << "Rendering points   ... ";
    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);
    cout << "[DONE]" << endl;
  }
  else
  {
    cout << "Cannot render points: Empty pointcloud" << endl;
  }
  if ( path.size() > 0 )
  {
    cout << "Rendering Cameras  ... ";
    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));
    window.setViewerPose(path[0]);
    cout << "[DONE]" << endl;
  }
  else
  {
    cout << "Cannot render the cameras: Empty path" << endl;
  }
	
  writeToPLY(point_cloud_est);	
  window.spin();
  return 0;
}
