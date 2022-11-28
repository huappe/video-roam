
#include <iostream>
#include <exception>

#include <opencv2/core.hpp>

#include "../tools/om_utils/include/roam/utils/timer.h"
#include "../tools/om_utils/include/roam/utils/confusion_matrix.h"
#include "../roam/include/VideoSegmenter.h"
#include "../roam/include/Configuration.h"
#include "../roam/include/ClosedContour.h"

#include "main_utils.h"

#define USE_CUDA

// Command line options
// # ./eval_cli -gt_files="./gt/gt.txt" -res_files="./res/res.txt" -out="./out.yaml"
const cv::String keys =
    "{help h usage ?       |                     | print this message             }"
    "{gt_files gt          |./gt.txt             | list of files gt               }"
    "{res_files res        |./res.txt            | list of result files   