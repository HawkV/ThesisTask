#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <random>
#include <fstream>
#include <memory>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

using std::vector;
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;
using Eigen::MatrixXd;

const float nn_match_ratio = 0.8 f;

void performanceTest(Mat img1, Mat img2) {
    int max_keypoints = 10000;

    vector < KeyPoint > keypoints1, keypoints2;

    auto start1 = std::chrono::steady_clock::now();

    Ptr < FeatureDetector > detector1 = ORB::create(max_keypoints);

    detector1 - > detect(img1, keypoints1);
    detector1 - > detect(img2, keypoints2);

    Mat desc1, desc2;

    Ptr < cv::xfeatures2d::DAISY > descriptor_extractor = cv::xfeatures2d::DAISY::create();

    descriptor_extractor - > compute(img1, keypoints1, desc1);
    descriptor_extractor - > compute(img2, keypoints2, desc2);

    auto end1 = std::chrono::steady_clock::now();

    auto diff1 = std::chrono::duration_cast < std::chrono::seconds > (end1 - start1).count();

    std::vector < vector < DMatch >> matches;

    FlannBasedMatcher flannmatcher;
    flannmatcher.add(desc1);
    flannmatcher.train();
    flannmatcher.knnMatch(desc2, matches, 2);

    int num_good = 0;
    std::vector < KeyPoint > matched1, matched2;
    std::vector < DMatch > good_matches;

    for (int i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];
        DMatch second = matches[i][1];

        if (first.distance < nn_match_ratio * second.distance) {
            matched1.push_back(keypoints1[first.trainIdx]);
            matched2.push_back(keypoints2[first.queryIdx]);
            good_matches.push_back(DMatch(num_good, num_good, 0));
            num_good++;
        }
    }

    auto precision1 = matches.size() / good_matches.size();

    auto start2 = std::chrono::steady_clock::now();

    Ptr < SIFT > detector2 = SIFT::create();
    Ptr < SIFT > extractor = SIFT::create();

    detector2 - > detect(img1, keypoints1);
    detector2 - > detect(img2, keypoints2);

    extractor - > compute(img1, keypoints1, desc1);
    extractor - > compute(img2, keypoints2, desc2);

    auto end2 = std::chrono::steady_clock::now();

    auto diff2 = std::chrono::duration_cast < std::chrono::seconds > (end2 - start2).count();

    std::vector < vector < DMatch >> matches2;

    FlannBasedMatcher flannmatcher2;
    flannmatcher2.add(desc1);
    flannmatcher2.train();
    flannmatcher2.knnMatch(desc2, matches2, 2);

    num_good = 0;

    std::vector < KeyPoint > matched21, matched22;
    std::vector < DMatch > good_matches2;

    for (int i = 0; i < matches.size(); i++) {
        DMatch first = matches2[i][0];
        DMatch second = matches2[i][1];

        if (first.distance < nn_match_ratio * second.distance) {
            matched21.push_back(keypoints1[first.trainIdx]);
            matched22.push_back(keypoints2[first.queryIdx]);
            good_matches2.push_back(DMatch(num_good, num_good, 0));
            num_good++;
        }
    }

    auto precision2 = matches2.size() / good_matches2.size();

    std::cout << "Время, сек.: " << diff1 << "Точность: " << precision1 << std::endl;
    std::cout << "Время, сек.: " << diff2 << "Точность: " << precision2 << std::endl;
}

void writePly(const char * path,
    const Eigen::MatrixXd & points,
        const Eigen::MatrixXd & colors) {
    std::ofstream ofs(path);
    ofs
        <<
        "ply" << std::endl <<
        "format ascii 1.0" << std::endl <<
        "element vertex " << points.rows() * points.cols() / 3 << std::endl <<
        "property float x" << std::endl <<
        "property float y" << std::endl <<
        "property float z" << std::endl <<
        "property uchar red" << std::endl <<
        "property uchar green" << std::endl <<
        "property uchar blue" << std::endl <<
        "end_header" << std::endl;

    for (Eigen::DenseIndex i = 0; i < points.cols(); ++i) {
        Eigen::Vector3d x = points.col(i);
        Eigen::Vector3d c = colors.col(i);

        ofs << x(0) << " " << x(1) << " " << x(2) << " " << (int) c(0) << " " << (int) c(1) << " " << (int) c(2) << std::endl;
    }

    ofs.close();
}

namespace eight {
    Eigen::Matrix3d essentialMatrix(const Eigen::Matrix3d & k,
        const Eigen::Matrix3d & f);

    Eigen::Matrix < double, 3, 4 > pose(const Eigen::Matrix3d & e,
        const Eigen::Matrix3d & k, Eigen::Ref <
            const Eigen::MatrixXd > a, Eigen::Ref <
                const Eigen::MatrixXd > b);

    Eigen::Matrix3d fundamentalMatrixUnnormalized(Eigen::Ref <
        const Eigen::MatrixXd > a, Eigen::Ref <
            const Eigen::MatrixXd > b);

    Eigen::Matrix3d fundamentalMatrix(Eigen::Ref <
        const Eigen::MatrixXd > a, Eigen::Ref <
            const Eigen::MatrixXd > b);

    Eigen::Matrix3d fundamentalMatrixRobust(
        Eigen::Ref <
        const Eigen::MatrixXd > a,
            Eigen::Ref <
            const Eigen::MatrixXd > b,
                std::vector < Eigen::DenseIndex > & inliers,
                double d = 3.0,
                double e = 0.2,
                double p = 0.99);

    Eigen::Affine2d findIsotropicNormalizingTransform(Eigen::Ref <
        const Eigen::MatrixXd > a);

    Eigen::Matrix < double, 3, 4 > cameraPose(const Eigen::Matrix < double, 3, 3 > & r,
        const Eigen::Vector3d & t);

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 3 > & r,
            const Eigen::Vector3d & t);

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::AffineCompact3d & pose);

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 4 > & pose);

    Eigen::Matrix < double, 3, Eigen::Dynamic > perspectiveProject(Eigen::Ref <
        const Eigen::MatrixXd > points,
            const Eigen::Matrix < double, 3, 4 > & p);

    template < class IndexIterator >
        Eigen::MatrixXd
    selectColumnsByIndex(const Eigen::Ref <
        const Eigen::MatrixXd > & m, IndexIterator begin, IndexIterator end) {

        Eigen::DenseIndex count = (Eigen::DenseIndex) std::distance(begin, end);

        Eigen::MatrixXd r(m.rows(), count);

        Eigen::DenseIndex i = 0;
        while (begin != end) {
            r.col(i++) = m.col( * begin++);
        }

        return r;
    }

    Eigen::Matrix < double, 3, Eigen::Dynamic > structureFromTwoViews(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 4 > & pose1,
            Eigen::Ref <
            const Eigen::MatrixXd > u0,
                Eigen::Ref <
                const Eigen::MatrixXd > u1);

    inline Eigen::Vector3d triangulate(const Eigen::Matrix < double, 3, 4 > & cam0,
        const Eigen::Matrix < double, 3, 4 > & cam1,
            const Eigen::Vector2d & u0,
                const Eigen::Vector2d & u1) {
        Eigen::Matrix < double, 4, 4 > X;
        X.row(0) = u0(0) * cam0.row(2) - cam0.row(0);
        X.row(1) = u0(1) * cam0.row(2) - cam0.row(1);
        X.row(2) = u1(0) * cam1.row(2) - cam1.row(0);
        X.row(3) = u1(1) * cam1.row(2) - cam1.row(1);

        return X.block < 4, 3 > (0, 0).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-X.col(3));
    }

    inline Eigen::Matrix < double, 3, Eigen::Dynamic > triangulateMany(const Eigen::Matrix < double, 3, 4 > & cam0,
        const Eigen::Matrix < double, 3, 4 > & cam1,
            Eigen::Ref <
            const Eigen::MatrixXd > u0,
                Eigen::Ref <
                const Eigen::MatrixXd > u1) {
        Eigen::Matrix < double, 3, Eigen::Dynamic > vecs(3, u0.cols());

        Eigen::Matrix < double, 4, 4 > X;
        for (Eigen::DenseIndex i = 0; i < u0.cols(); ++i) {
            X.row(0) = u0(0, i) * cam0.row(2) - cam0.row(0);
            X.row(1) = u0(1, i) * cam0.row(2) - cam0.row(1);
            X.row(2) = u1(0, i) * cam1.row(2) - cam1.row(0);
            X.row(3) = u1(1, i) * cam1.row(2) - cam1.row(1);
            vecs.col(i) = X.block < 4, 3 > (0, 0).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-X.col(3));
        }

        return vecs;
    }

    class SampsonDistanceSquared {
        public:
            double operator()(const Eigen::Matrix3d & f, Eigen::Ref <
                const Eigen::Vector2d > a, Eigen::Ref <
                    const Eigen::Vector2d > b) const;
    };

    template < class Functor >
        Eigen::VectorXd distances(const Eigen::Matrix3d & f, Eigen::Ref <
            const Eigen::MatrixXd > a, Eigen::Ref <
                const Eigen::MatrixXd > b, Functor err) {
            Eigen::VectorXd errs(a.cols());
            for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
                errs(i) = err(f, a.col(i), b.col(i));
            }
            return errs;
        }

    double SampsonDistanceSquared::operator()(const Eigen::Matrix3d & f, Eigen::Ref <
        const Eigen::Vector2d > a, Eigen::Ref <
            const Eigen::Vector2d > b) const {
        Eigen::Vector3d fa = f.transpose() * a.homogeneous();
        Eigen::Vector3d fb = f * b.homogeneous();

        double bfa = a.homogeneous().transpose() * fb;

        return (bfa * bfa) / (fa.topRows(2).squaredNorm() + fb.topRows(2).squaredNorm());
    }

    Eigen::Matrix3d essentialMatrix(const Eigen::Matrix3d & k,
        const Eigen::Matrix3d & f) {
        return k.transpose() * f * k;
    }

    Eigen::Matrix < double, 3, 4 > pose(const Eigen::Matrix3d & e,
        const Eigen::Matrix3d & k, Eigen::Ref <
            const Eigen::MatrixXd > a, Eigen::Ref <
                const Eigen::MatrixXd > b) {
        Eigen::JacobiSVD < Eigen::Matrix3d > svd(e, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3d u = svd.matrixU();
        Eigen::Matrix3d v = svd.matrixV();

        if (u.determinant() < 0.0)
            u *= -1.0;
        if (v.determinant() < 0.0)
            v *= -1.0;

        Eigen::Matrix3d w;
        w <<
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;

        Eigen::Matrix3d r0 = u * w * v.transpose();
        Eigen::Matrix3d r1 = u * w.transpose() * v.transpose();
        Eigen::Vector3d t = u.col(2);

        Eigen::Matrix < double, 3, 4 > camFirst = cameraMatrix(k, Eigen::Matrix < double, 3, 4 > ::Identity());
        Eigen::Matrix < double, 3, 4 > camPosesSecond[4] = {
            cameraPose(r0, t),
            cameraPose(r0, -t),
            cameraPose(r1, t),
            cameraPose(r1, -t)
        };

        int bestId = 0;
        int bestCount = 0;
        for (int i = 0; i < 4; ++i) {

            Eigen::Matrix < double, 3, 4 > camSecond = cameraMatrix(k, camPosesSecond[i]);

            Eigen::Matrix < double, 3, Eigen::Dynamic > p = triangulateMany(camFirst, camSecond, a, b);
            Eigen::Matrix < double, 3, Eigen::Dynamic > pSecond = camSecond * p.colwise().homogeneous();

            int count = 0;
            for (Eigen::DenseIndex j = 0; j < p.cols(); ++j) {
                if (p(2, j) >= 0.0 && pSecond(2, j) >= 0.0) {
                    ++count;
                }
            }

            if (count > bestCount) {
                bestCount = count;
                bestId = i;
            }
        }

        return camPosesSecond[bestId];
    }

    Eigen::Matrix3d fundamentalMatrixUnnormalized(Eigen::Ref <
        const Eigen::MatrixXd > a, Eigen::Ref <
            const Eigen::MatrixXd > b) {

        eigen_assert(a.cols() == b.cols());
        eigen_assert(a.rows() == b.rows());
        eigen_assert(a.cols() >= 8);

        Eigen::Matrix < double, Eigen::Dynamic, 9 > A(a.cols(), 9);

        for (Eigen::DenseIndex i = 0; i < a.cols(); ++i) {
            const auto & ca = a.col(i);
            const auto & cb = b.col(i);

            auto r = A.row(i);

            r(0) = cb.x() * ca.x();
            r(1) = cb.x() * ca.y();
            r(2) = cb.x();
            r(3) = cb.y() * ca.x();
            r(4) = cb.y() * ca.y();
            r(5) = cb.y();
            r(6) = ca.x();
            r(7) = ca.y();
            r(8) = 1.0;
        }

        Eigen::SelfAdjointEigenSolver < Eigen::Matrix < double, Eigen::Dynamic, 9 > > e;
        e.compute((A.transpose() * A));
        eigen_assert(e.info() == Eigen::Success);

        Eigen::Matrix < double, 1, 9 > f = e.eigenvectors().col(0);

        Eigen::Matrix3d F;
        F <<
            f(0), f(3), f(6),
            f(1), f(4), f(7),
            f(2), f(5), f(8);

        Eigen::JacobiSVD < Eigen::Matrix3d > svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::DiagonalMatrix < double, 3 > dPrime(svd.singularValues()(0), svd.singularValues()(1), 0.0);
        Eigen::Matrix3d FPrime = svd.matrixU() * dPrime * svd.matrixV().transpose();

        return FPrime;

    }

    Eigen::Matrix3d fundamentalMatrix(Eigen::Ref <
        const Eigen::MatrixXd > a, Eigen::Ref <
            const Eigen::MatrixXd > b) {
        std::cout << "fund";

        Eigen::Transform < double, 2, Eigen::Affine > t0 = findIsotropicNormalizingTransform(a);
        Eigen::Transform < double, 2, Eigen::Affine > t1 = findIsotropicNormalizingTransform(b);

        Eigen::Matrix < double, 2, Eigen::Dynamic > na = (t0.matrix() * a.colwise().homogeneous()).colwise().hnormalized();
        Eigen::Matrix < double, 2, Eigen::Dynamic > nb = (t1.matrix() * b.colwise().homogeneous()).colwise().hnormalized();

        Eigen::Matrix3d Fn = eight::fundamentalMatrixUnnormalized(na, nb);
        Eigen::Matrix3d F = (t0.matrix().transpose() * Fn * t1.matrix());
        return F;
    }

    template < class ForwardIterator, class T >
        void fillIncremental(ForwardIterator first, ForwardIterator last, T val) {
            while (first != last) {
                * first = val;
                ++first;
                ++val;
            }
        }

    std::vector < Eigen::DenseIndex > samplePointIndices(Eigen::DenseIndex setSize, Eigen::DenseIndex sampleSize) {
        std::vector < Eigen::DenseIndex > ids(setSize);
        fillIncremental(ids.begin(), ids.end(), 0);

        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector < Eigen::DenseIndex > result(sampleSize);
        for (Eigen::DenseIndex i = 0; i < sampleSize; i++) {
            std::uniform_int_distribution < Eigen::DenseIndex > dis(i, setSize - 1);
            std::swap(ids[i], ids[dis(gen)]);
            result[i] = ids[i];
        }

        return result;
    }

    Eigen::Affine2d findIsotropicNormalizingTransform(Eigen::Ref <
        const Eigen::MatrixXd > a) {
        std::cout << "isotropic";

        Eigen::Vector2d mean = a.rowwise().mean();
        Eigen::Vector2d stddev = (a.colwise() - mean).array().square().rowwise().mean().sqrt();

        Eigen::Affine2d t;
        t = Eigen::Scaling(1.0 / stddev.norm()) * Eigen::Translation2d(-mean);
        return t;
    }

    Eigen::Matrix < double, 3, 4 > cameraPose(const Eigen::Matrix < double, 3, 3 > & r,
        const Eigen::Vector3d & t) {
        Eigen::AffineCompact3d iso;
        iso.linear() = r;
        iso.translation() = t;

        return iso.matrix();
    }

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 3 > & r,
            const Eigen::Vector3d & t) {
        return cameraMatrix(k, cameraPose(r, t));
    }

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::AffineCompact3d & pose) {
        return k * pose.inverse(Eigen::Isometry).matrix();
    }

    Eigen::Matrix < double, 3, 4 > cameraMatrix(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 4 > & pose) {
        Eigen::AffineCompact3d iso(pose);
        return k * iso.inverse(Eigen::Isometry).matrix();
    }

    Eigen::Matrix < double, 3, Eigen::Dynamic > perspectiveProject(Eigen::Ref <
        const Eigen::MatrixXd > points,
            const Eigen::Matrix < double, 3, 4 > & p) {
        return p * points.colwise().homogeneous();
    }

    inline Eigen::Matrix < double, 3, 3 > hat(Eigen::Ref <
        const Eigen::Vector3d > u) {

        Eigen::Matrix < double, 3, 3 > h(3, 3);
        h <<
            0.0, -u.z(), u.y(),
            u.z(), 0.0, -u.x(),
            -u.y(), u.x(), 0.0;

        return h;
    }

    Eigen::Matrix < double, 3, Eigen::Dynamic > structureFromTwoViews(const Eigen::Matrix < double, 3, 3 > & k,
        const Eigen::Matrix < double, 3, 4 > & pose1,
            Eigen::Ref <
            const Eigen::MatrixXd > u0,
                Eigen::Ref <
                const Eigen::MatrixXd > u1) {
        eigen_assert(u0.cols() == u1.cols());
        eigen_assert(u0.cols() > 0);

        Eigen::MatrixXd points(3, u0.cols());

        Eigen::MatrixXd m = Eigen::MatrixXd::Zero(3 * u0.cols(), u0.cols() + 1);

        auto u0h = u0.colwise().homogeneous();
        auto u1h = u1.colwise().homogeneous();

        Eigen::Matrix < double, 3, 3 > kinv = k.inverse();

        Eigen::Matrix < double, 3, 4 > poseinv = Eigen::AffineCompact3d(pose1).inverse(Eigen::Isometry).matrix();
        Eigen::Matrix < double, 3, 3 > r = poseinv.block < 3, 3 > (0, 0);
        Eigen::Vector3d t = poseinv.block < 3, 1 > (0, 3);

        for (Eigen::DenseIndex i = 0; i < u0.cols(); ++i) {
            Eigen::Matrix < double, 3, 3 > u1hat = hat(kinv * u1h.col(i));

            m.block < 3, 1 > (i * 3, i) = u1hat * r * kinv * u0h.col(i);
            m.block < 3, 1 > (i * 3, m.cols() - 1) = u1hat * t;
        }

        std::cout << m.cols() * m.rows();

        Eigen::SelfAdjointEigenSolver < Eigen::MatrixXd > e;
        e.compute(m.transpose() * m);
        Eigen::VectorXd x = e.eigenvectors().col(0);

        for (Eigen::DenseIndex i = 0; i < u0.cols(); ++i) {
            points.col(i) = x(i) * kinv * u0h.col(i);
        }

        return points;
    }
}

int main(int argc, char ** argv) {
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);

    vector < KeyPoint > keypoints1, keypoints2;

    Ptr < FeatureDetector > detector = ORB::create(10000);

    detector - > detect(img1, keypoints1);
    detector - > detect(img2, keypoints2);

    Mat desc1, desc2;

    Ptr < cv::xfeatures2d::DAISY > descriptor_extractor = cv::xfeatures2d::DAISY::create();

    descriptor_extractor - > compute(img1, keypoints1, desc1);
    descriptor_extractor - > compute(img2, keypoints2, desc2);

    std::vector < vector < DMatch >> matches;

    FlannBasedMatcher flannmatcher;
    flannmatcher.add(desc1);
    flannmatcher.train();
    flannmatcher.knnMatch(desc2, matches, 2);

    int num_good = 0;
    std::vector < KeyPoint > matched1, matched2;
    std::vector < DMatch > good_matches;

    for (int i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];
        DMatch second = matches[i][1];

        if (first.distance < nn_match_ratio * second.distance) {
            matched1.push_back(keypoints1[first.trainIdx]);
            matched2.push_back(keypoints2[first.queryIdx]);
            good_matches.push_back(DMatch(num_good, num_good, 0));
            num_good++;
        }
    }

    const double focusDistance = 1.0;
    const int width = img1.cols;
    const int height = img1.rows;

    Eigen::Matrix3d k;

    k << focusDistance, 0.0, 0.5 * (width - 1),
        0.0, focusDistance, 0.5 * (height - 1),
        0.0, 0.0, 1.0;

    int pointCount = good_matches.size();

    Eigen::MatrixXd a(2, pointCount), b(2, pointCount);
    Eigen::MatrixXd colors(3, pointCount);

    for (size_t i = 0; i < pointCount; i++) {
        Point2f point1 = matched1[good_matches[i].queryIdx].pt;
        Point2f point2 = matched2[good_matches[i].trainIdx].pt;

        a(0, i) = point1.x;
        a(1, i) = point1.y;

        b(0, i) = point2.x;
        b(1, i) = point2.y;

        Scalar color1 = img1.at < uchar > (point1);
        Scalar color2 = img2.at < uchar > (point2);

        for (size_t j = 0; j < 3; j++) {
            colors(j, i) = (color1.val[j] + color2.val[j]) / 2;
        }
    }

    std::cout << "Reconstruction start";

    auto F = eight::fundamentalMatrix(a, b);

    std::cout << "fundamentalMatrix";

    Eigen::Matrix3d E = eight::essentialMatrix(k, F);

    std::cout << "essentialMatrix";

    Eigen::Matrix < double, 3, 4 > pose = eight::pose(E, k, a, b);

    std::cout << "pose";

    Eigen::Matrix < double, 3, Eigen::Dynamic > pointsTriangulated = eight::structureFromTwoViews(k, pose, a, b);

    std::cout << "points";

    writePly("points.ply", pointsTriangulated, colors);

    std::cout << "triangulated";

    return 0;
}
