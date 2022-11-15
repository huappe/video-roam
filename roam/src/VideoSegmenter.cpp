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
            this->landmarks_tree->BuildDPTable();

            // Run DP and apply moves
            FLOAT_TYPE min_cost_landmarks = 0.f;
            if(this->landmarks_tree->GetDPTable()->pairwise_costs.size() > 0)
            {
                min_cost_landmarks = this->landmarks_tree->RunDPInference();
                this->landmarks_tree->ApplyMoves();
            }
            LOG_INFO("VideoSegmenter::SetContours() - Landmarks min_cost: " << min_cost_landmarks);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////// Processing Intermediate Mask ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<Node*> contour_nodes_ptrs = createVectorOfPointersFromList(contour->contour_nodes);
    cv::Mat intermediate_mask;
    std::vector<cv::Point> intermediate_contour;
    std::vector<cv::Point> intermediate_motion_diff;
    if (this->params.use_landmarks && this->landmarks_tree->DPTableIsBuilt())
    {
        const std::shared_ptr<RigidTransform_ContourWarper> &warper = std::dynamic_pointer_cast<RigidTransform_ContourWarper>(this->contour_warper);
        warper->Init(this->landmarks_tree->correspondences_a, this->landmarks_tree->correspondences_b,this->frame_mask);
        intermediate_contour = warper->Warp(this->contour->contour_nodes);

        intermediate_motion_diff = findDiffsContourMove(intermediate_contour);
        contourToMask(intermediate_contour, intermediate_mask, this->next_image.size());
    }
    else
        intermediate_mask = this->frame_mask;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////// Processing Contour Nodes /////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Check if Green theorem is needed
    if (params.use_green_theorem_term)
    {
        if (!contour_init)
            fb_model = GlobalModel(GlobalModel::Params(3,50.f,50.f,1000.f));

        this->updateIntegralNegRatioForGreenTheorem(intermediate_mask);
    }

    // Simple terms if/else
    if (!contour_init)
    {
        if (params.use_gradients_unary)
        {
            contour_elements.contour_gradient_unary = ROAM::GradientUnary::createUnaryTerm(
                            ROAM::GradientUnary::Params(params.grad_type, params.grad_kernel_size, params.gradients_weight,
                                                        params.gaussian_smoothing_factor));
            contour_elements.contour_gradient_unary->Init(this->next_image);
        }

        if (params.use_norm_pairwise)
        {
            contour_elements.contour_norm_pairwise = ROAM::NormPairwise::createPairwiseTerm(ROAM::NormPairwise::Params(params.norm_type, params.norm_weight));
            contour_elements.contour_norm_pairwise->Init(cv::Mat(), cv::Mat());
        }

        if (params.use_gradient_pairwise && params.use_gradients_unary)
        {
            contour_elements.contour_generic_pairwise = ROAM::GenericPairwise::createPairwiseTerm(ROAM::GenericPairwise::Params(params.gradient_pairwise_weight));
            contour_elements.contour_generic_pairwise->Init(contour_elements.contour_gradient_unary->GetUnaries(), cv::Mat());
        }

    }
    else
    {

        if (params.use_gradients_unary)
            contour_elements.contour_gradient_unary->Update(this->next_image);

        if (params.use_gradient_pairwise && params.use_gradients_unary)
            contour_elements.contour_generic_pairwise->Update(contour_elements.contour_gradient_unary->GetUnaries(), cv::Mat());

        if (params.use_norm_pairwise)
            contour_elements.contour_norm_pairwise->Update(cv::Mat(), cv::Mat());
    }

    // Resize lists of pairwises
    if (!contour_init)
    {
        if (params.use_snapcut_pairwise)
        {
            contour_elements.snapcut_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::SnapcutPairwise>>(contour_nodes_ptrs.size());
        }

        if (params.use_temp_norm_pairwise)
        {
            contour_elements.tempangle_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::TempAnglePairwise>>(contour_nodes_ptrs.size());
            contour_elements.tempnorm_pairwises_per_contour_node = std::list<cv::Ptr<ROAM::TempNormPairwise>>(contour_nodes_ptrs.size());
            contour_elements.prev_norms = std::list<FLOAT_TYPE>(contour_nodes_ptrs.size(), -1.f);
            contour_elements.prev_angles = std::list<FLOAT_TYPE>(contour_nodes_ptrs.size(), static_cast<FLOAT_TYPE>(-5 * CV_PI));
        }

        if (params.use_landmarks)
            contour_elements.distance_unaries_per_contour_node = std::list<cv::Ptr<ROAM::DistanceUnary>>(contour_nodes_ptrs.size());

        if (params.use_green_theorem_term)
            contour_elements.green_theorem_pairwises = std::list<cv::Ptr<ROAM::GreenTheoremPairwise>>(contour_nodes_ptrs.size());

    }

    // For simplicity: Let us first process non-cuda terms
    if (!contour_init)
    {
        #pragma omp parallel for
        for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
        {
            if (params.use_gradients_unary)
                contour_nodes_ptrs[ii]->AddUnaryTerm(contour_elements.contour_gradient_unary);

            if (params.use_norm_pairwise)
                contour_nodes_ptrs[ii]->AddPairwiseTerm(contour_elements.contour_norm_pairwise);

            if (params.use_temp_norm_pairwise)
            {
                auto tempnorm_pairwises_per_contour_node_it = std::next(contour_elements.tempnorm_pairwises_per_contour_node.begin(), ii);
                auto tempangle_pairwises_per_contour_node_it = std::next(contour_elements.tempangle_pairwises_per_contour_node.begin(), ii);
                auto prev_norms_it = std::next(contour_elements.prev_norms.begin(), ii);
                auto prev_angles_it = std::next(contour_elements.prev_angles.begin(), ii);

                const cv::Ptr<ROAM::TempAnglePairwise> tempAnglePairwise = ROAM::TempAnglePairwise::createPairwiseTerm(ROAM::TempAnglePairwise::Params(this->params.temp_angle_weight));
                const cv::Ptr<ROAM::TempNormPairwise> tempNormPairwise = ROAM::TempNormPairwise::createPairwiseTerm(ROAM::TempNormPairwise::Params(this->params.temp_norm_weight));

                tempAnglePairwise->Init( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_angles_it), cv::Mat() );
                tempNormPairwise->Init( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_norms_it), cv::Mat() );

                *tempangle_pairwises_per_contour_node_it = tempAnglePairwise;
                *tempnorm_pairwises_per_contour_node_it = tempNormPairwise;

                contour_nodes_ptrs[ii]->AddPairwiseTerm(tempAnglePairwise);
                contour_nodes_ptrs[ii]->AddPairwiseTerm(tempNormPairwise);

                if (ii == contour_nodes_ptrs.size()-1)
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[0]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
                else
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[ii+1]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
            }

            if (params.use_landmarks)
            {
                auto distance_unaries_per_contour_node_it = std::next(contour_elements.distance_unaries_per_contour_node.begin(), ii);
                cv::Ptr<ROAM::DistanceUnary> distanceUnary = ROAM::DistanceUnary::createUnaryTerm(
                            ROAM::DistanceUnary::Params(this->params.landmark_to_node_weight) );
                distanceUnary->Init( this->landmarks_tree->VectorOfClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                     this->params.landmark_to_node_radius) );

                distanceUnary->SetPastDistance(-1.f /*when a negative number is passed, the cost becomes 0: only for intiialization*/);
                *distance_unaries_per_contour_node_it = distanceUnary;
                contour_nodes_ptrs[ii]->AddUnaryTerm(distanceUnary);
            }

            if (params.use_green_theorem_term)
            {
                auto green_pairwises_per_contour_node_it = std::next(contour_elements.green_theorem_pairwises.begin(), ii);
                const cv::Ptr<ROAM::GreenTheoremPairwise> gtPairwise =
                        ROAM::GreenTheoremPairwise::createPairwiseTerm(
                            ROAM::GreenTheoremPairwise::Params(this->params.green_theorem_weight,
                                                               this->contour->IsCounterClockWise(),
                                                               ii, (ii+1==contour_nodes_ptrs.size())?0:ii+1));
                //std::cout<<integral_negative_ratio_foreground_background_likelihood<<std::endl;
                gtPairwise->Init(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);

                *green_pairwises_per_contour_node_it = gtPairwise;
                contour_nodes_ptrs[ii]->AddPairwiseTerm(gtPairwise);
            }

#ifndef WITH_CUDA
            // Fill vector of snapcut terms (non cuda)
            if (params.use_snapcut_pairwise)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);
                ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                             params.snapcut_sigma_color, params.snapcut_region_height,
                                                             params.label_space_side,
                                                             params.snapcut_number_clusters, false);

                cv::Ptr<ROAM::SnapcutPairwise> snapcutPairwise = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
                snapcutPairwise->Init(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                 prev_image, frame_mask);
                    *(snapcut_pairwises_per_contour_node_it) = snapcutPairwise;
                    contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[ii+1]);
                    *snapcut_pairwises_per_contour_node_it = snapcutPairwise;
                    contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
                }
            }
#endif
        }
    }
    else // The contour was already initialized, so we update
    {
        if (this->params.use_landmarks)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto distance_unaries_per_contour_node_it = std::next(contour_elements.distance_unaries_per_contour_node.begin(), ii);

                const cv::Ptr<ROAM::DistanceUnary> &distanceUnary = *distance_unaries_per_contour_node_it;
                distanceUnary->Update( this->landmarks_tree->VectorOfClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                                           this->params.landmark_to_node_radius) );
                if (!landmarks_tree->DPTableIsBuilt())
                    distanceUnary->SetPastDistance( -1.f );
                else
                    distanceUnary->SetPastDistance( this->landmarks_tree->AverageDistanceToClosestLandmarkPoints(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                                                                 this->params.landmark_to_node_radius) );
                    //distanceUnary->SetPastDistance( -1.f );
            }
        }

        if (params.use_temp_norm_pairwise)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto tempnorm_pairwises_per_contour_node_it = std::next(contour_elements.tempnorm_pairwises_per_contour_node.begin(), ii);
                auto tempangle_pairwises_per_contour_node_it = std::next(contour_elements.tempangle_pairwises_per_contour_node.begin(), ii);
                auto prev_norms_it = std::next(contour_elements.prev_norms.begin(), ii);
                auto prev_angles_it = std::next(contour_elements.prev_angles.begin(), ii);

                const cv::Ptr<ROAM::TempAnglePairwise> &tempAnglePairwise = *tempangle_pairwises_per_contour_node_it;
                const cv::Ptr<ROAM::TempNormPairwise> &tempNormPairwise = *tempnorm_pairwises_per_contour_node_it;

                tempAnglePairwise->Update( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_angles_it), cv::Mat() );
                tempNormPairwise->Update( cv::Mat(cv::Mat_<FLOAT_TYPE>(1,1)<<*prev_norms_it), cv::Mat() );

                if (ii == contour_nodes_ptrs.size()-1)
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[0]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
                else
                {
                    const cv::Point a = contour_nodes_ptrs[ii]->GetCoordinates();
                    const cv::Point b = contour_nodes_ptrs[ii + 1]->GetCoordinates();
                    *prev_angles_it = static_cast<FLOAT_TYPE>(std::atan2(b.y - a.y, b.x - a.x));
                    *prev_norms_it = static_cast<FLOAT_TYPE>(cv::norm(b - a));
                }
            }
        }

        if (params.use_green_theorem_term)
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto green_pairwises_per_contour_node_it = std::next(contour_elements.green_theorem_pairwises.begin(), ii);
                const cv::Ptr<ROAM::GreenTheoremPairwise> &gtPairwise = *green_pairwises_per_contour_node_it;

                gtPairwise->Update(this->next_image, this->integral_negative_ratio_foreground_background_likelihood);
            }
        }

#ifndef WITH_CUDA
        #pragma omp parallel for
        for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
        {
            // Fill vector of snapcut terms (non cuda)
            if (params.use_snapcut_pairwise)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);
                cv::Ptr<ROAM::SnapcutPairwise>& snapcutPairwise = *(snapcut_pairwises_per_contour_node_it);
                snapcutPairwise->Update(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[0]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(), contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                prev_image, frame_mask,
                                intermediate_motion_diff[ii], intermediate_motion_diff[ii+1]);
                }
            }
        }
#endif
    }

#ifdef WITH_CUDA
    contour->BuildDPTable();
    if (params.use_snapcut_pairwise)
    {
        if (!contour_init)
        {
            ROAM::SnapcutPairwise::Params snapcut_params(params.snapcut_weight,
                                                         params.snapcut_sigma_color, params.snapcut_region_height,
                                                         params.label_space_side, params.snapcut_number_clusters, true);

            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);

                *snapcut_pairwises_per_contour_node_it = ROAM::SnapcutPairwise::createPairwiseTerm(snapcut_params);
                (*snapcut_pairwises_per_contour_node_it)->Init(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {

                    if (ii == contour_nodes_ptrs.size()-1)
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[0]->GetCoordinates(),
                                                                               prev_image, frame_mask);
                    else
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                                               prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[0]->GetCoordinates(),
                                                                               prev_image, frame_mask,
                                                                               intermediate_motion_diff[ii],
                                                                               intermediate_motion_diff[0]);
                    else
                        (*snapcut_pairwises_per_contour_node_it)->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                                               contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                                               prev_image, frame_mask,
                                                                               intermediate_motion_diff[ii],
                                                                               intermediate_motion_diff[ii+1]);
                }

                contour_nodes_ptrs[ii]->AddPairwiseTerm(*snapcut_pairwises_per_contour_node_it);
            }
        }
        else
        {
            #pragma omp parallel for
            for(auto ii = 0; ii<contour_nodes_ptrs.size(); ++ii)
            {
                auto snapcut_pairwises_per_contour_node_it = std::next(contour_elements.snapcut_pairwises_per_contour_node.begin(), ii);

                cv::Ptr<ROAM::SnapcutPairwise>& snapcutPairwise = *snapcut_pairwises_per_contour_node_it;

                snapcutPairwise->Update(this->next_image, intermediate_mask);

                if (!params.use_landmarks || !landmarks_tree->DPTableIsBuilt())
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[0]->GetCoordinates(), prev_image, frame_mask);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[ii+1]->GetCoordinates(), prev_image, frame_mask);
                }
                else
                {
                    if (ii == contour_nodes_ptrs.size()-1)
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[0]->GetCoordinates(),
                                                        prev_image, frame_mask,
                                                        intermediate_motion_diff[ii],
                                                        intermediate_motion_diff[0]);
                    else
                        snapcutPairwise->InitializeEdge(contour_nodes_ptrs[ii]->GetCoordinates(),
                                                        contour_nodes_ptrs[ii+1]->GetCoordinates(),
                                                        prev_image, frame_mask,
                                                        intermediate_motion_diff[ii],
                                                        intermediate_motion_diff[ii+1]);
                }

                *snapcut_pairwises_per_contour_node_it = snapcutPairwise;
            }
        }
    }

    if (params.use_landmarks && landmarks_tree->DPTableIsBuilt())
        performIntermediateContourMove(intermediate_contour);

    if (params.use_snapcut_pairwise)
    {
        contour->ExecuteCudaPairwises(params.snapcut_region_height, params.snapcut_sigma_color,
                                      params.snapcut_weight, next_image.rows, next_image.cols);
    }

    contour->BuildDPTable();
#else
    if (params.use_landmarks && landmarks_tree->DPTableIsBuilt())
        performIntermediateContourMove(intermediate_contour);

    contour->BuildDPTable();
#endif

    this->contour_init = true;
}

// -----------------------------------------------------------------------------------
bool VideoSegmenter::IsInit() const
// -----------------------------------------------------------------------------------
{
    return this->contour_init;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::WriteOutput(const std::string &foldername)
// -----------------------------------------------------------------------------------
{
    this->namefolder = foldername;
    this->write_masks = true;
}

// -----------------------------------------------------------------------------------
std::vector<cv::Point> VideoSegmenter::ProcessFrame()
// -----------------------------------------------------------------------------------
{
    assert(this->contour_init);

    this->current_contour_cost = this->contour->RunDPInference();

    this->contour->ApplyMoves();

    std::vector<cv::Point> output_cont_pts;
    for (auto itc=contour->contour_nodes.begin(); itc!=this->contour->contour_nodes.end(); ++itc)
        output_cont_pts.push_back(itc->GetCoordinates());

    contourToMask(output_cont_pts, this->frame_mask, this->next_image.size());

    if (this->params.use_graphcut_term)
        automaticReparametrization();


    const double t = chrono_timer_per_frame.Stop();
    LOG_INFO("VideoSegmenter::ProcessFrame() - TOTAL per-frame exec: " << t);
    LOG_INFO("VideoSegmenter::ProcessFrame() - min cost: " << this->current_contour_cost );

    costs_per_frame.push_back(this->current_contour_cost);

    return output_cont_pts;
}

// -----------------------------------------------------------------------------------
cv::Mat VideoSegmenter::WritingOperations()
// -----------------------------------------------------------------------------------
{
    if (write_masks)
    {
        const cv::Mat draw = this->contour->DrawContour(this->next_image, this->frame_mask);
        std::string filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".png");
        cv::imwrite(filename, draw);

        filename = namefolder + std::string("/") + std::to_string(frame_counter) + std::string(".png");
        cv::imwrite(filename, frame_mask);

        filename = namefolder + std::string("/cont_") + std::to_string(frame_counter) + std::string(".txt");
        this->WriteTxt(filename);

        ++frame_counter;
    }
    return frame_mask;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::WriteTxt(const std::string &filename) const
// -----------------------------------------------------------------------------------
{
	std::ofstream f_out(filename);

	const auto &nodes = this->contour->contour_nodes;
	f_out << nodes.size() << " # number of nodes (for closed contour)" <<std::endl;
	f_out << "x y" << std::endl;

	for (auto it = nodes.begin(); it != nodes.end(); ++it)
		f_out << it->GetCoordinates().x << " " << it->GetCoordinates().y << std::endl;


	if(this->params.use_landmarks)
	{
		f_out << "------------------------------------------------" << std::endl;

		const auto &landmarks = this->landmarks_tree->graph_nodes;
		
		f_out << std::endl << landmarks.size() << " # number of landmarks (pictorial structure)" << std::endl;
		f_out << "x y" << std::endl;

		for (auto it = landmarks.begin(); it != landmarks.end(); ++it)
		{
			const cv::Point n_c = GET_NODE_FROM_TUPLE(*it)->GetCoordinates();
			f_out << n_c.x << " " << n_c.y << std::endl;
		}

	}

}

// -----------------------------------------------------------------------------------
VideoSegmenter::Params VideoSegmenter::getParams() const
// -----------------------------------------------------------------------------------
{
    return params;
}

// -----------------------------------------------------------------------------------
void VideoSegmenter::setParams(const VideoSegmenter::Params &value)
// -----------------------------------------------------------------------------------
{
    params = value;
}

// -----------------------------------------------------------------------------------
static
std::vector<cv::Point> ContourToVectorPoints(const std::shared_ptr<ClosedContour> &contour)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> pts;

    for (auto it = contour->contour_nodes.begin(); it != contour->contour_nodes.end(); ++it)
        pts.push_back(it->GetCoordinates());

    return pts;
}

// -----------------------------------------------------------------------------------
static
cv::Rect FindContourRange(const std::vector<cv::Point> &contour, int slack=30)
// -----------------------------------------------------------------------------------
{
    cv::Rect range;

    int min_x=100000;
    int min_y=100000;
    int max_x=0;
    int max_y=0;
    for (int p_i = 0; p_i < contour.size(); ++p_i)
    {
        const cv::Point& p = contour[p_i];


        min_x = std::min(p.x,min_x);
        max_x = std::max(p.x,max_x);
        min_y = std::min(p.y,min_y);
        max_y = std::max(p.y,max_y);
    }

    range = cv::Rect(min_x-slack/2, min_y-slack/2, max_x-min_x+slack, max_y-min_y+slack);
    return range;
}


// -----------------------------------------------------------------------------------
void VideoSegmenter::automaticReparametrization()
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> next_contour = ContourToVectorPoints(this->contour);

    LOG_INFO("VideoSegmenter::AutomaticReparametrization() - Reparametrizing ");

    cv::Mat gc_segmented;

    bool 