#include "GC_Energy.h"



using namespace ROAM;

// ---------------------------------------------------------------------------
GC_Energy::GC_Energy(const GC_Energy::Params &parameters)
// ---------------------------------------------------------------------------
{
    this->params = parameters;
    model = GlobalModel(GlobalModel::Params(this->params.iterations, this->params.betha_nd, this->params.gamma_gc, this->params.lambda_gc));
    this->is_initialized = false;
}

// ---------------------------------------------------------------------------
bool GC_Energy::Initialized() const
// ---------------------------------------------------------------------------
{
    return is_initialized;
}

// ---------------------------------------------------------------------------
bool GC_Energy::initialize(const cv::Mat &data, const cv::Mat &labels)
// ---------------------------------------------------------------------------
{
    return this->model.Initialize(data, labels);
}

// ---------------------------------------------------------------------------
bool GC_Energy::update(const cv::Mat &data, const cv::Mat &labels)
// ---------------------------------------------------------------------------
{
    return this->model.Update(data, labels);
}

// TODO: why raw pointer and undeclared in header?
// ---------------------------------------------------------------------------
static void
construct8Graph(const cv::Mat& img, const cv::Mat& mask,
                const GlobalModel& model, double lambda,
                const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
                std::shared_ptr<GraphType> graph, const cv::Mat &precomputed_contour_likelihood = cv::Mat())
// ---------------------------------------------------------------------------
{
    cv::Mat buffer_er = cv::Mat::zeros( img.size(), CV_32FC1 );

    for(auto y = 0; y < img.rows; ++y)
        for(auto x = 0; x < img.cols; ++x)
        {
            // add node
            const int vtxIdx = graph->add_node();
            const cv::Vec3b &color = img.at<cv::Vec3b>(y, x);

            cv::Vec2f pre_likelihood(0,0);
            if(!precomputed_contour_likelihood.empty())
                pre_likelihood = precomputed_contour_likelihood.at<cv::Vec2f>(y, x);

            // set t-weights
            float fromSource, toSink;
            if (mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_PR_BGD || mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_PR_FGD)
            {
                const cv::Vec2f lik_vec = model.ComputeLikelihood(color) + pre_likelihood;
                fromSource = -std::log( lik_vec[1] );
                toSink = -std::log( lik_vec[0] );

                buffer_er.at<float>(y, x) = static_cast<float>(toSink) / (fromSource + std::numeric_limits<float>::epsilon());
            }
            else if(mask.at<uchar>(y, x) == GC_Energy::GCClasses::GC_BGD)
            {
                fromSource = 0;
                toSink = static_cast<float>(lambda);

                buffer_er.at<float>(y, x) = 100;
            }
            else // GC_FGD
            {
                fromSource = static_cast<float>(lambda);
                toSink = 0;

                buffer_er.at<float>(y, x) = 0;
            }
            graph->add_tweights( vtxIdx, fromSource, toSink );

            // set n-weights
            if(x > 0)
            {
                const float &w = static_cast<float>(leftW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-1, w, w );
            }
            if(x > 0 && y > 0)
            {
                const float &w = static_cast<float>(upleftW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if(y > 0)
            {
                const float &w = static_cast<float>(upW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if(x < img.cols - 1 && y > 0)
            {
                const float &w = static_cast<float>(uprightW.at<double>(y, x));
                graph->add_edge( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
}


// TODO: why raw pointer and undeclared in header?
// ----------------------------------------------