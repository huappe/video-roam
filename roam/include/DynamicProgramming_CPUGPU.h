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
#include <memory>
#include <limits>
#include <algorithm>
#include <stdint.h>

#include "Configuration.h"
#include "DynamicProgramming.h"

#ifdef WITH_CUDA
#include "../cuda/dp_cuda.h"
#endif

namespace ROAM
{

/*!
 * \brief The DPTable base struct
          Assumes [node][label] and [node][l1][l2]
 */
// -----------------------------------------------------------------------------------
struct DPTable
// -----------------------------------------------------------------------------------
{
    virtual ~DPTable() {}

    DPTableUnaries unary_costs;
    DPTablePairwises pairwise_costs;

    virtual void Initialize() = 0;

   uint16_t max_number_labels;
   uint16_t number_nodes;
};


/*!
 * \brief The DynamicProgramming class
 */
// -----------------------------------------------------------------------------------
cl