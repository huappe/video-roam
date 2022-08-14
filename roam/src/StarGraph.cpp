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
        center.y /= static_cast<FLOAT_TYPE>(graph_n