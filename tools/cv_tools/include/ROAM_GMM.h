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

#include <opencv2/opencv.hpp>
#include "../../../roam/include/Configuration.h"

#ifndef M_PI
    #define M_PI       3.14159265358979323846
#endif

namespace ROAM
{

/*!
* \brief GMM: a simple gmm model with online adaptation
*/
// -----------------------------------------------------------------------------------
struct GMMModel
// -----------------------------------------------------------------------------------
{
    // -----------------------------------------------------------------------------------
    struct MixtureComponent
    // -----------------------------------------------------------------------------------
    {
        MixtureComponent() : weight(0.0), mass(0) {}
        cv::Vec3f mean;			/// mean vector
        cv::Mat covariance;		/// covariance matrix
        cv::Mat iCovariance;	/// pre-computed inverted covariance matrix (save computation during Mahalanobis)
        float weight;			/// weight of a mixture
        size_t mass;			/// number of pixels from which the cluster is constructed
    };

    // -----------------------------------------------------------------------------------
    struct Parameters
    // -----------------------------------------------------------------------------------
    {
        // -----------------------------------------------------------------------------------
        explicit Parameters(const size_t k_ = 3)
        : attempts(5), k(k_), min_fraction(0.1f), use_binary_segment(true), update_k(3),
        update_min_fraction(0.1f), d_similar_merge(1.0f), d_similar_skip(1.5f), awu(10000), decay_factor(0.97f),
        discard_threshold(20), noise_factor(0.00005f)
        // -----------------------------------------------------------------------------------
        {
        }

        // -----------------------------------------------------------------------------------
        Parameters(const size_t n_frames1, const size_t n_frames2, const size_t patch_size)
            : Parameters(3)
        // -----------------------------------------------------------------------------------
        {
            const size_t n_pixels = (2 * patch_size) * (2 * patch_size);
            const size_t expected_per_cluster = static_cast<size_t>(n_pixels / static_cast<FLOAT_TYPE>(2 * k));

            awu = expected_per_cluster * n_frames1;
            decay_factor = static_cast<FLOAT_TYPE>(std::pow(awu, -1.0 / static_cast<FLOAT_TYPE>(n_frames2)));
            discard_threshold = std::max(static_cast<int>(expected_per_cluster * 0.01), 10);
        }

        size_t attempts;
        size_t k;
        FLOAT_TYPE min_fraction;

        bool use_binary_segment;

        size_t update_k;
        FLOAT_TYPE update_min_fraction;

        FLOAT_TYPE d_similar_merge;
        FLOAT_TYPE d_similar_skip;

        size_t awu;
        FLOAT_TYPE decay_factor;
        size_t discard_threshold;

        FLOAT_TYPE noise_factor;
    };

    // -----------------------------------------------------------------------------------
    explicit GMMModel(const Parameters &parameters = Parameters())
        : n_components(0), params(parameters), initialized(false)
    // -----------------------------------------------------------------------------------
    {
    }

    // -----------------------------------------------------------------------------------
    int initializeMixture(const cv::Mat &patch, const cv::Mat &mask)
    // -----------------------------------------------------------------------------------
    {
        gmm = getMixture(patch, mask, params.k, params.min_fraction);

        if(gmm.size() == 0)
            return -1;

        n_components = gmm.size();
        initialized = true;

        return 0;
    }

    // -----------------------------------------------------------------------------------
    int initia