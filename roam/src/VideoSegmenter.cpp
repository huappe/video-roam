#include "VideoSegmenter.h"

using namespace ROAM;

// -----------------------------------------------------------------------------------
VideoSegmenter::VideoSegmenter(const VideoSegmenter::Params& params)
// -----------------------------------------------------------------------------------
{
    this->params = params;
    this->contour_init = false;
    this->frame_counter = 0;
    this->write_masks = false;
	this->current_contour_cost = std