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
        added_list.insert(std::next(added_list.begin(), prev_node + 1), element);
        return prev_node + 1;
    }
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::reinitializeNode(Node &node, const Node &next_node, int prev_node, int node_idx, ContourElementsHelper &ce) const
// -----------------------------------------------------------------------------------
{
    if(this->contour_init)
    {
        node = Node(node.GetCoordinates(), Node::Params(params.label_space_side));

        if (params.use_gradients_unary)
            node.AddUnaryTerm(ce.contour_gradient_unary);

        if (params.use_norm_pairwise)
            node.AddPairwiseTerm(ce.contour_norm_pairwise);

        if(params.use_temp_norm_pairwise)
        {
            const cv::Ptr<ROAM::TempAnglePairwise> tempAnglePairwise = ROAM::TempAnglePairwise::createPairwiseTerm(ROAM::TempAnglePairwise::Params(this->params.temp_angle_weight));
            const cv::Ptr<ROAM::TempNormPairwise> tempNormPairwise = ROAM::TempNormPairwise::createPairwiseTerm(ROAM::TempNormPairwise::Params(this->params.temp_norm_weight));

            tempAnglePairwise->Init(cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1) << -5.f * CV_PI), cv::Mat());
            tempNormPairwise->Init(cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1) << -1.f), cv::Mat());

            addNodeToList(ce.tempangle_pairwises_per_contour_node, tempAnglePairwise, prev_node);
            addNodeToList(ce.tempnorm_pairwises_per_contour_node, tempNormPairwise, prev_node);

            node.AddPairwiseTerm(tempAnglePairwise);
            node.AddPairwiseTerm(tempNormPairwise);

            const cv::Point a = node.GetCoordinates();
            const cv::Point b = next_node.GetCoordinates();

            addNodeToList(ce.prev_angles, FLOAT_TYPE(std::atan2(b.y - a.y, b.x - a.x)), prev_node);
            addNodeToList(ce.prev_norms, FLOAT_TYPE(cv::norm(b - a)), prev_node);
        }

        if(params.use_landmarks)
        {
            const cv::Ptr<ROAM::DistanceUnary> distanceUnary = ROAM::DistanceUnary::createUnaryTerm(
                                ROAM::DistanceUnary::Params(this->params.landmark_to_node_weight));
            distanceUnary->Init(cv::Mat());
            distanceUnary->SetPastDistance(-1.f /*when a negative number is passed, the cost becomes 0: only for intiialization*/);

            addNodeToList(ce.distance_unaries_per_contour_node, distanceUnary, prev_node);
            node.AddUnaryTerm(distanceUnary);
        }

        if(params.use_green_theorem_term)
        {
            const cv::Ptr<ROAM::GreenTheoremPairwise> gtPairwise =
                    ROAM::GreenTheoremPairwise::createPairwiseTerm(
                    ROAM::GreenTheoremPairwise::Params(this->params.green_theorem_weight,
                                                           this->contour->IsCounterClockWise(),
                                                           prev_node, node_idx));

            gtPairwise->Init(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);

            addNodeToList(ce.green_theorem_pairwises, gtPairwise, prev_node);
            node.AddPairwiseTerm(gtPairwise);
        }

        if (params.use_snapcut_pairwise)
        {
            ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                         params.snapcut_sigma_color, params.snapcut_region_height,
                                                         params.label_space_side, params.snapcut_number_clusters, true);

            const cv::Ptr<ROAM::SnapcutPairwise> snapcutPairwise = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
            snapcutPairwise->Init(this->next_image, this->frame_mask/*intermediate_mask*/);

            addNodeToList(ce.snapcut_pairwises_per_contour_node, snapcutPairwise, prev_node);

            // I guesss I don't need to check this since frame_mask was already updated:
            // if (!params.use_landmarks || !this->landmarks_tree->DPTableIsBuilt())
            snapcutPairwise->InitializeEdge(node.GetCoordinates(),
                                            next_node.GetCoordinates(),
                                            this->next_image, this->frame_mask);

            node.AddPairwiseTerm(snapcutPairwise);
        }
    }

}

// -----------------------------------------------------------------------------------
template<typename T> static std::vector<T*>
createVectorOfPointersFromList(std::list<T>& the_list)
// -----------------------------------------------------------------------------------
{
    std::vector<T*> ptrs;
    for(auto it = the_list.begin(); it != the_list.end(); ++it)
	    ptrs.push_back(&(*it));

    return ptrs;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::SetContours(const std::vector<cv::Point> &contour_pts_)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> contour_pts;
    if(contour_pts_.empty() && !contour->contour_nodes.empty())
    {
        for(auto itc=contour->contour_nodes.begin(); itc!=this->contour->contour_nodes.end(); ++itc)
            contour_pts.push_back(itc->GetCoordinates());
    }
    else
    {
        contour_pts = contour_pts_;
    }

    chrono_timer_per_frame.Start();

    if (!this->contour_init)
    {
        this->params.Print();

        this->frame_mask = cv::Mat::zeros(this->next_image.size(), CV_8UC1);
        this->contour = std::make_shared<ClosedContour>(ClosedContour::Params(this->params.label_space_side * this->params.label_space_side));

        contourToMask(contour_pts, this->frame_mask, this->next_image.size());

        // Initializing position of nodes with contour_pts
        this->contour->contour_nodes.clear();
        for(size_t i=0; i<contour_pts.size(); ++i)
            this->contour->contour_nodes.push_back(ROAM::Node(contour_pts[i], Node::Params(params.label_space_side)));

        frame_counter=0;
        if(write_masks)
        {
            // Writing the first frame output
            cv::Mat draw = this->contour->DrawContour(this->prev_image, this->frame_mask);
            std::string filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".png");
            cv::imwrite(filename, draw);

            filename = namefolder + std::string("/") + std::to_string(frame_counter) + std::string(".png");
            cv::imwrite(filename, frame_mask);
            ++frame_counter;
        }

        // Initialize Landmarks
        if(this->params.use_landmarks)
        {
            this->landmarks_tree = std::make_shared<StarGraph>(
                    StarGraph::Params(this->params.landmarks_searchspace_side,
                    this->params.landmark_min_response, this->params.landmark_max_area_overlap,
                    this->params.landmark_min_area, this->params.max_number_landmarks,
                    this->params.landmark_pairwise_weight));

            switch (this->params.warper_type)
            {
            case WARP_TRANSLATION:
                this->contour_warper = std::make_shared<RigidTransform_ContourWarper>(RigidTransform_ContourWarper::Params(RigidTransform_ContourWarper::TRANSLATION));
                break;

            default:
            case WARP_SIMILARITY:
                this->contour_warper = std::make_shared<RigidTransform_ContourWarper>(RigidTransform_ContourWarper::Params(RigidTransform_ContourWarper::SIMILARITY));
                break;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////// Processing Landmark Nodes ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create / Check for new  landmarks
    if(this->params.use_landmarks)
    {
        this->landmarks_tree->UpdateLandmarks(this->prev_image, this->frame_mask, true);
        this->landmarks_tree->TrackLandmarks(this->next_image); //landmarks only get moved when DP solved

        if(this->landmarks_tree->graph_nodes.size()>0)
        {
            this->landm