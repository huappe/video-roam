#include "Reparametrization.h"

namespace ROAM
{

// TODO: Reparametrization was quickly implemented during a short layover at the Munich Airport. 
//    The code is very sub-optimal and should be re-implemented.

// -----------------------------------------------------------------------------------
std::vector<cv::Point> ContourFromMask(const cv::Mat &mask, const int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> output_contour;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, CV_RETR_EXTERNAL, type_simplification);

    // Choose largest contour
    std::vector<double> sizes;
    sizes.resize(contours.size());

    #pragma omp parallel for
    for(auto i = 0; i < contours.size(); ++i)
        sizes[i] = cv::contourArea(contours[i]);

    const auto maxIndex = std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end()));

    if (type_simplification == CV_CHAIN_APPROX_NONE && contours.size()>0 )
        for ( size_t i=0; i<contours[maxIndex].size(); i=i+2 ) // TODO: not sure why ``i+2'', isn't this hardcoded quantization?
            output_contour.push_back(contours[maxIndex][i]);
    else
        output_contour = contours[maxIndex];

    return output_contour;
}

// -----------------------------------------------------------------------------------
std::vector< std::vector<cv::Point> > ComponentsFromMask(const cv::Mat &mask, const int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( mask, contours, CV_RETR_EXTERNAL, type_simplification );
    return contours;
}

// -----------------------------------------------------------------------------------
static
size_t findClosestNode(const std::vector<cv::Point> &node_pts, const cv::Point &pt,
                       const std::vector<cv::Point> &banned,
                       const FLOAT_TYPE banned_dst = 1.0)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> distances = std::vector<FLOAT_TYPE>(node_pts.size(), std::numeric_limits<FLOAT_TYPE>::infinity());

    #pragma omp parallel for
    for(auto i = 0; i < node_pts.size(); ++i)
    {
        for(auto j = 0; j < banned.size(); ++j)
            if(cv::norm(banned[j] - node_pts[i]) < banned_dst)
                continue;

        distances[i] = static_cast<FLOAT_TYPE>(cv::norm(node_pts[i] - pt));
    }

    const size_t min_id = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    return min_id;
}

// -----------------------------------------------------------------------------------
static
size_t findClosestNode(const std::vector<cv::Point> &node_pts, const cv::Point &pt)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> distances;
    distances.resize(node_pts.size());

    #pragma omp parallel for
    for(auto i = 0; i < node_pts.size(); ++i)
        distances[i] = static_cast<FLOAT_TYPE>(cv::norm(node_pts[i] - pt));

    const size_t min_id = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    return min_id;
}

// -----------------------------------------------------------------------------------
static
void findPoint(const std::vector<cv::Point2i> &pts, const cv::Point2i &pt,
               std::set<size_t> &ids)
// -----------------------------------------------------------------------------------
{
    for(size_t i = 0; i < pts.size(); ++i)
        if(pts[i] == pt)
            ids.insert(i);
}

// -----------------------------------------------------------------------------------
static
FLOAT_TYPE computePixelDistance(const std::vector<cv::Point> &proposed_contour,
                                const cv::Point &a, const cv::Point &b, const bool pos,
                                std::vector<size_t> &path)
// -----------------------------------------------------------------------------------
{
    std::set<size_t> id_a;
    std::set<size_t> id_b;

    findPoint(proposed_contour, a, id_a);
    findPoint(proposed_contour, b, id_b);

    if(id_a.empty() || id_b.empty())
        return -1;

    FLOAT_TYPE distance = std::numeric_limits<FLOAT_TYPE>::infinity();

    for(std::set<size_t>::const_iterator it1 = id_a.begin(); it1 != id_a.end(); ++it1)
        for(std::set<size_t>::const_iterator it2 = id_b.begin(); it2 != id_b.end(); ++it2)
        {
            // TODO: change
            const size_t idx_a = *it1;
            const size_t idx_b = *it2;

            const int min_id = static_cast<int>(std::min(idx_a, idx_b));
            const int max_id = static_cast<int>(std::max(idx_a, idx_b));

            const FLOAT_TYPE current_dst = pos ? static_cast<float>(std::abs(max_id - min_id)) : static_cast<float>(proposed_contour.size() - std::abs(max_id - min_id));

            if(current_dst < distance)
            {
                distance = current_dst;

                // TODO: reimplement
                std::set<size_t> added;

                if(pos)
                {
              