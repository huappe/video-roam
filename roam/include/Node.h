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

#include <vector>

#include <opencv2/core.hpp>

#include "EnergyTerms.h"
#include "GreenTheorem.h"

namespace ROAM
{

typedef unsigned short label;

/*!
 * \brief The LabelSpace struct
 */
 // -----------------------------------------------------------------------------------
struct LabelSpace
// -----------------------------------------------------------------------------------
{
    explicit LabelSpace(const uint side_length = 3);

    void Rebuild(const uint side_length);

    cv::Point GetDisplacementsFromLabel(const ROAM::label l) const;
    uint GetNumLabels() const;

protected:
    void fillCoordinates(const uint side_length);

    std::vector<cv::Point> indexed_coordinates;
};


/*!
 * \brief The Node class
 */
// -----------------------------------------------------------------------------------
class Node
// -----------------------------------------------------------------------------------
{
public:
    /*!
     * \brief The Params struct
     */
    struct Params
    {
        // ---------------------------------------------------------------------------
        explicit Params(const unsigned int label_space_side = 5)
        // ---------------------------------------------------------------------------
        {
            this->label_space_side = label_space_side;
        }

        unsigned int label_space_side;
    };

    explicit Node(const cv::Point coo