#include "StarGraph.h"

using namespace ROAM;

//-----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------- StarGraph --------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
StarGraph::StarGraph(const StarGraph::Params &parameters)
// -----------------------------------------------------------------------------------
{
    params = parameters;
    dp_table_built = false;
}

// -----------------------------------------------------------------------------------
void StarGraph::BuildDPTable()
// -----------------------------------------------------------------------------------
{
    const uint number_nodes = static_cast<uint>(graph_nodes.size());

    dp_table = std::make_shared<StarDPTable>(params.node_side_length*params.node_side_length, number_nodes);
    dp_table->Initialize();

    if(dp_table->pairwise_costs.size() < 1 || dp_table->unary_costs.size() < 1)
    {
        dp_table_built = false;
        return;
    }

    // openMP does not support std::list
    std::vector<Node*> elements;
    for(auto it = graph_nodes.begin(); it != graph_nodes.end(); ++it)
        elements.push_back(GET_NODE_FROM_TUPLE(*it).get());

    // fill in unary terms
    #pragma omp parallel for
    for(auto n = 0; n < elements.size(); ++n)
        for(auto l = 0; l < static_cast<int>(elements[n]->GetLabelSpaceSize()); ++l)
            dp_table->unary_costs[n][l] = (n == 0) ? 0.f : elements[n]->GetTotalUnaryCost(l);

    #pragma omp parallel for
    for(auto n = 1; n < elements.size(); ++n)
        for(auto l1 = 0; l1 < static_cast<int>(elements[n]->GetLabelSpaceSize()); ++l1)
            for(auto l2 = 0; l2 < static_cast<int>(elements[0]->GetLabelSpaceSize()); ++l2)
                dp_table->pairwise_costs[n-1][l1][l2] = elements[n]->GetTotalPairwiseCost(l1, l2, *elements[0]);

    // Table was built
    dp_table_built = true;
}

// -----------------------------------------------------------------------------------
bool StarGraph::DPTableIsBuilt() const
// -----------------------------------------------------------------------------------
{
    return this->dp_table_built;
}

// -----------------------------------------------------------------------------------
FLOAT_TYPE StarGraph::RunDPInference()
// -----------------------------------------------------------------------------------
{
    assert(dp_table_built);

    FLOAT_TYPE min_cost = std::numeric_limits<FLOAT_TYPE>::max();
    current_solution = dp_solver.Minimize(this->dp_table, min_cost);

    return min_cost;
}

// -----------------------------------------------------------------------------------
std::vector<label> StarGraph::GetCurrentSolution() const
// -----------------------------------------------------------------------------------
{
    return this->current_solution;
}

// TODO: double check
// -----------------------------------------------------------------------------------
void StarGraph::ApplyMoves()
// -----------------------------------------------------------------------------------
{
    assert(graph_nodes.size() == current_solution.size());

    const size_t n_nodes = graph_nodes.size();
    correspondences_a.resize(n_nodes);
    correspondences_b.resize(n_nodes);

    // TODO: make parallel (despite it's std::list)
    size_t ind_sol = 0;
    for (auto it1 = graph_nodes.begin(); it1 != graph_nodes.end(); ++it1, ++ind_sol)
    {
        auto node_it1 = GET_NODE_FROM_TUPLE(*it1);
        cv::Rect &rect_it1 = GET_RECT_FROM_TUPLE(*it1);
        const cv::Point pt_before_move = node_it1->GetCoordinates();

        // move node (center point)
        node_it1->SetCoordinates(current_solution[ind_sol]);
        const cv::Point pt_after_move = node_it1->GetCoordinates();

        // updates position of rectangle
        rect_it1.x = pt_after_move.x - rect_it1.width / 2;
        rect_it1.y = pt_after_move.y - rect_it1.height / 2;

        correspondences_a[ind_sol] = pt_before_move;
        correspondences_b[ind_sol] = pt_after_move;
    }
}

// -----------------------------------------------------------------------------------
std::shared_ptr<DPTable> StarGraph::GetDPTable() const
// -----------------------------------------------------------------------------------
{
    return dp_table;
}

// -----------------------------------------------------------------------------------
static cv::Point2f centerNode(const std::list<Landmark>& graph_nodes,
                            bool count_first = true)
// -----------------------------------------------------------------------------------
{
    cv::Point2f center(0,0);

    // TODO: make parallel with reduction (despite it's std::list)
    if(count_first)
    {
        for(auto it = graph_nodes.begin(); it != graph_nodes.end(); ++it)
        {
            center.x += static_cast<FLOAT_TYPE>(GET_NODE_FROM_TUPLE(*it)->GetCoordinates().x);
            center.y += static_cast<FLOAT_TYPE>(GET_NODE_FROM_TUPLE(*it)->GetCoordinates().y);
        }

        center.x /= static_cast<FLOAT_TYPE>(graph_nodes.size());
        center.y /= static_cast<FLOAT_TYPE>(graph_nodes.size());
    }
    else
    {
        for(auto it = ++graph_nodes.begin(); it != graph_nodes.end(); ++it)
        {
            center.x += static_cast<FLOAT_TYPE>(GET_NODE_FROM_TUPLE(*it)->GetCoordinates().x);
            center.y += static_cast<FLOAT_TYPE>(GET_NODE_FROM_TUPLE(*it)->GetCoordinates().y);
        }

        center.x /= static_cast<FLOAT_TYPE>(graph_nodes.size()) - 1;
        center.y /= static_cast<FLOAT_TYPE>(graph_nodes.size()) - 1;
    }

    return center;
}

// -----------------------------------------------------------------------------------
void StarGraph::UpdateLandmarks(const cv::Mat &image, const cv::Mat &mask, bool update_root)
// -----------------------------------------------------------------------------------
{
    const size_t sz_before_adding = this->graph_nodes.size();

    this->removeOutsiderLandmarks(mask, image);
    this->addLandmarks(image, mask);

    // Add virtual central node
    if(sz_before_adding == 0 && this->graph_nodes.size() > 0)
    {
        const cv::Point2f center_pt = centerNode(this->graph_nodes);
        auto node = std::make_shared<Node>(center_pt, Node::Params(this->params.node_side_length));

        this->generic_unaries_per_landmark.push_front(GenericUnary::createUnaryTerm(GenericUnary::Params(0.f)));
        this->tempnorm_pairwises_per_landmark.push_front(TempNormPairwise::createPairwiseTerm(TempNormPairwise::Params(0.f)));

        node->AddUnaryTerm(this->generic_unaries_per_landmark.front());
        node->AddPairwiseTerm(this->tempnorm_pairwises_per_landmark.front());

        const Landmark lm = std::make_tuple(node, cv::Rect(), std::make_shared<KCF>());
        this->graph_nodes.push_front(lm);
    }

    // we'll update also the root node
    if(sz_before_adding > 0 && update_root)
    {
        const cv::Point center = centerNode(this->graph_nodes, false);
        auto root_node = this->graph_nodes.begin();
        auto root = GET_NODE_FROM_TUPLE(*root_node);
        root->SetCoordinates(center);
    }
}

// -----------------------------------------------------------------------------------
void StarGraph::addLandmarks(const cv::Mat &image, const cv::Mat &mask)
// -----------------------------------------------------------------------------------
{
    const std::vector<cv::Rect> rects = this->getMSERBoxes(image, mask);

    for(size_t r = 0; r < rects.size(); ++r)
        if(this->params.max_number_landmarks < 0 || this->graph_nodes.size() < this->params.max_number_landmarks)
        {
            KCF::Output response;
            auto kcf = std::make_shared<KCF>();
            try
            {
                kcf->Evaluate(image, response, rects[r]); //!> Init // TODO: Use a rect on image so to make it more efficient
                kcf->Evaluate(image, response);		      //!> Evaluate response
            }
            catch(cv::Exception& ) // TODO: ooooh, this has to be fixed properly !!!
            {
                LOG_ERROR("Something went wrong");
                continue;
            }

            double min, max;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(response.img_regression, &min, &max, &min_loc, &max_loc);

            // Better use PSR?
            if(max < this->params.min_response_new_landmarks)
                continue;

            auto node = std::make_shared<Node>(max_loc, Node::Params(this->params.node_side_length));

            cv::Ptr<GenericUnary> generic_unary = GenericUnary::createUnaryTerm(GenericUnary::Params(1.f, ROAM::GenericUnary::NO_POOL));
            cv::Ptr<TempNormPairwise> temp_norm_pairwise = TempNormPairwise::createPairwiseTerm(TempNormPairwise::Params(this->params.weight_pairwises));
            generic_unaries_per_landmark.push_back(generic_unary);
            tempnorm_pairwises_per_landmark.push_back(temp_norm_pairwise);

            node->AddUnaryTerm(generic_unary);
            node->AddPairwiseTerm(temp_norm_pairwise);

            const Landmark lm = std::make_tuple(node, rects[r], kcf);
            graph_nodes.push_back(lm);
        }
}

// -----------------------------------------------------------------------------------
void StarGraph::removeOutsiderLandmarks(const cv::Mat &mask, const cv::Mat &image)
// -------------------------------------------------