/*
This is a source code of "ROAM: a Rich Object Appearance Model with Application to Rotoscoping" implemented by Ondrej Miksik and Juan-Manuel Perez-Rua. 

@inproceedings{miksik2017roam,
  author = {Ondrej Miksik and Juan-Manuel Perez-Rua and Philip H.S. Torr and Patrick Perez},
  title = {ROAM: a Rich Object Appearance Model with Application to Rotoscoping},
  booktitle = {CVPR},
  year = {2017}
}
*/

#pragma once

#include <vector>
#include <set>
#include <list>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Configuration.h"
#include "../tools/om_utils/include/roam/utils/timer.h"

namespace ROAM
{

// TODO: Reparametrization was quickly implemented during a short layover at the Munich Airport. 
//    The code is very sub-optimal and should be re-implemented.

// TODO: no ``reparametrization'' class ??

// -----------------------------------------------------------------------------------
// TODO: This should be refactored
// also, do we use it globally or just as a container within Reparametrization.cpp?
struct ProposalsBlobs
// -----------------------------------------------------------------------------------
{
    std::vector<std::vector<cv::Point> > blobs_g_c;
    std::vector<std::vector<cv::Point> > blobs_c_g;
    std::vector<std::vector<cv::Point> > blobs_holes;

    std::vector<std::vector<cv::Point> > contour_g_c;
    std::vector<std::vector<cv::Point> > contour_c_g;
    std::vector<std::vector<cv::Point> > contour_holes;

    cv::Mat dst_g_c; //_g_c
    cv::Mat dst_c_g; //_c_g
};

// -----------------------------------------------------------------------------------
struct ProposalsBox
// -----------------------------------------------------------------------------------
{
    std::set<size_t> remove_nodes;
    std::vector<cv::Point2i> add_nodes;
    std::pair<size_t, size_t> min_max_ids;
    size_t mass;
};

std::vector<cv::Point> ContourFromMask(const cv::Mat &mask, const int type_simplification=CV_CHAIN_APPROX_NONE);

