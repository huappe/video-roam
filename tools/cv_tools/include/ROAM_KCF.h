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
#include <vector>
#include <math.h>

#include "../../../roam/include/Configuration.h"

#ifndef M_PI
    #define M_PI           3.14159265358979323846  /* pi */
#endif

namespace ROAM
{

/*!
* \brief KCF: An object tracking class based on Kernelized Correlation Filters for object tracking.
*/
// variables with "_f" suffix are in fourier domain
// -----------------------------------------------------------------------------------
class KCF
// -----------------------------------------------------------------------------------
{
public:

    // -----------------------------------------------------------------------------------
    enum Kernels
    // -----------------------------------------------------------------------------------
    {
        KRR_LINEAR,
        KRR_GAUSSIAN,
        KRR_POLYNOMIAL
    };

    // -----------------------------------------------------------------------------------
    enum Features
    // -----------------------------------------------------------------------------------
    {
        FT_GRAY,
        FT_COLOUR
    };

    // -----------------------------------------------------------------------------------
    struct Parameters
    // -----------------------------------------------------------------------------------
    {
        Parameters()
        {
            kernel = KRR_LINEAR;
            feature_type = FT_COLOUR; // Features::FT_GRAY;

            interp_patch = 0.075f;
            interp_alpha = 0.075f;

            lambda = 1e-4f;
            target_sigma = 0.1f;//0.1;

            target_padding = 1.5f;
            detection_padding = 1.5f;
            min_displacement = cv::Size(30, 30);

            train_confidence = 0.0f;
        }

        Kernels kernel;
        Features feature_type