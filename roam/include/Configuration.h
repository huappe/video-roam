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

// -----------------------------------------------------------------------------------
// Assert in release mode (we can disable for timing breakdowns)
// -----------------------------------------------------------------------------------
#undef NDEBUG
#include <assert.h>
#include <iostream>

// -----------------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------------
#ifndef DOUBLE_PRECISION
    typedef float FLOAT_TYPE;
#else
    typedef double FLOAT_TYPE;
#endif

// -----------------------------------------------------------------------------------
// LOG
// -----------------------------------------------------------------------------------
//#define USE_GLOG					/// do we want to use glog?
#define USE_CONSOLE					/// do we want to use console?

/// you can define as many macros as you want here (example from some other project)
#define PRINT_PROGRESS				/// do we want to print progress?
//#define PRINT_DETAILS				/// do we want details for diversity?

// -----------------------------------------------------------------------------------
// Visualization
// -----------------------------------------------------------------------------------
#define PLOT_DEBUG_IMGS				/// plot various figures
#define CV_WAIT_PERIOD 1			/// 0 = wait for keypress, otherwise use 1, used only for debug i