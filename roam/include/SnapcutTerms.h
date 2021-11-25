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

#include "Configuration.h"
#include "EnergyTerms.h"
#include "RotatedRect.h"

#include "../../tools/cv_tools/include/ROAM_GMM.h"
#include "../../tools/om_utils/include/roam/utils/timer.h"

#ifdef WITH_CUDA
    #include "../cuda/roam_cuda.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ROAM
{

/*!
 * \brief The SnapcutPairwise class is an implementation of PariwiseTerm
 * It is based on local temporally consistent GMMs. It is implemented in
 * both CPU and GPU, as it is a very expensive term.
 */
// -----------------------------------------------------------------------------------
class SnapcutPairwise : public PairwiseTerm
// -----------------------------------------------------------------------------------
{
public:
    // -------------------------------------------------------------------------------
    struct Params
    // -------------------------------------------------------------------------------
    {
        // ---------------------------------------------------------------------------
        explicit Params(FLOAT_TYPE weight_term=0.5f,
                        FLOAT_TYPE sigma_color=0.1f,
                        FLOAT_TYPE region_height=10.0f,
                        FLOAT_TYPE node_side_length=11.f,
                        int number_clusters=3, bool block_get_cost = false)
        // ----------------------------------------------------------