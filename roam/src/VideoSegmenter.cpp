#include "VideoSegmenter.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
VideoSegmenter::VideoSegmenter(const VideoSegmenter::Params& params)
// -----------------------------------------------------------------------------------
{
    this->params = params;
    this->contour_init = false;
    this->frame_counter = 0;
    this->write_masks = false;
	this->current_contour_cost = std::numeric_limits<FLOAT_TYPE>::max();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetNextImageAuto(const cv::Mat &next_image)
// -----------------------------------------------------------------------------------
{
    if (!this->next_image.empty())
        this->prev_image = this->next_image.clone();
    else
        this->prev_image = next_image.clone();

    this->next_image = next_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetNextImage(const cv::Mat &next_image)
// -----------------------------------------------------------------------------------
{
    this->next_image = next_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetPrevImage(const cv::Mat &prev_image)
// -----------------------------------------------------------------------------------
{
    this->prev_image = prev_image.clone();
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetParameters(const Params &params)
// -----------------------------------------------------------------------------------
{
    this->params = params;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::Write(cv::FileStorage& fs) const
// -----------------------------------------------------------------------------------
{
    this->params.Write(fs);
}

// -----------------------------------------------------------------------------------
static std::vector<FLOAT_TYPE> compDiffs(const std::shared_ptr<ClosedContour>& cont)
// -----------------------------------------------------------------------------------
{
    std::vector<FLOAT_TYPE> diffs;

    FLOAT_TYPE avg = 0.0f;
    for(auto itc = cont->contour_nodes.begin(); itc != --cont->contour_nodes.end(); ++itc)
    {
        auto nitc = std::next(itc, 1);
        const FLOAT_TYPE dis = static_cast<FLOAT_TYPE>(cv::norm(itc->GetCoordinates()-nitc->GetCoordinates()));
        avg += dis;
        diffs.push_back(dis);
    }
    avg /= static_cast<FLOAT_TYPE>(cont->contour_nodes.size());

    diffs.push_back( avg );

    return diffs;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> VideoSegmenter::findDiffsContourMove(const std::vector<cv::Point> &move_to_point) const
// -----------------------------------------------------------------------------------
{
    //move_to point is the intermediate_contour
    std::vector<cv::Point> diff;
    diff.reserve(move_to_point.size());

    size_t ind = 0;
    for(auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it, ++ind)
        diff.push_back(move_to_point[ind] - it->GetCoordinates());

    // outputs the vector that goes from contour coordinates to the new position
    return diff;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::performIntermediateContourMove(const std::vector<cv::Point> &move_to_point) const
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> diff;
    diff.reserve(move_to_point.size());

    size_t ind = 0;
    for(auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it)
        it->SetCoordinates(move_to_point[ind++]);
}

// -----------------------------------------------------------------------------------
static void
contourToMask(const std::vector<cv::Point> &contour_pts, cv::Mat &mask, cv::Size size)
// -----------------------------------------------------------------------------------
{
    // Mask from contours
    std::vector<std::vector<cv::Point>> array_conts = {contour_pts};
    mask = cv::Mat::zeros(size, CV_8UC1);
    cv::fillPoly(mask, array_conts, cv::Scalar(255));
}

// -----------------------------------------------------------------------------------
template<typename T> static
size_t addNodeToList(std::list<T>& added_list, const T &element, const int prev_node)
// -----------------------------------------------------------------------------------
{
    if(prev_node < 0)
    {
        added_list.push_back(element);
        return added_list.size();
    }
    else
    {