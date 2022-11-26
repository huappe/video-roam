#include <iostream>
#include <exception>

#include "../tools/om_utils/include/roam/utils/timer.h"
#include "../roam/include/VideoSegmenter.h"
#include "../roam/include/Configuration.h"
#include "../roam/include/ClosedContour.h"

#include "main_utils.h"


// Command line options
const cv::String keys =
    "{help h usage ?       |                     | print this message             }"
    "{path_to_sequence seq |../Toy/Toy.txt       | path to sequence file          }"
    "{path_to_masks    msk |../Toy/ToyMasks.txt  | path to masks file             }"
    "{path_to_output   out |./output/Toy         | path to output folder          }"
    "{contour_width    wid |9                    | path to output folder          }"
    ;

int main(int argc, char *argv[])
{

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////// Parsing stuff //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Draw contours v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    cv::String path_to_sequence     = parser.get<cv::String>("path_to_sequence");
    cv::String path_to_masks        = parser.get<cv::String>("path_to_masks");
    cv::String path_to_