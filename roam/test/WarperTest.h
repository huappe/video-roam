#include "DynamicProgramming.h"
#include "ClosedContour.h"
#include "EnergyTerms.h"
#include "Node.h"
#include "RotatedRect.h"
#include "ContourWarper.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


TEST(WarperTest, WarperTest1)
{

    std::vector<cv::Point> pts_a = {cv::Point(0,0), cv::Point(1,1), cv::Point(1,0), cv::Point(0,1), cv::Point(2,2), cv::Point(3,3), cv::Point(2,3), cv::Point(3,2)};
    std::vector<cv::Point> pts_b = {cv::Point(1,1), cv::Point(2,2), cv::Point(2,0), cv::Point(0,2), cv::Point(3,3), cv::Point(4,4), cv::Point(3,4), cv::Point(4,3)};


    std::shared_ptr<roam::RigidTransform_ContourWarper> warper1 = std::make_shared<roam::RigidTransform_ContourWarper>(roam::RigidTransform_ContourWarper::Params(roam::RigidTransform_ContourWarper::TRANSLATION));
    std::shared_ptr<roam::RigidTransform_ContourWarper> warper2 = std::make_shared<roam::RigidTransform_ContourWarper>(r