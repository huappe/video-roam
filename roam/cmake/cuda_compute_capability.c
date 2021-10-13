/*
* Copyright (C) 2011 Florian Rathgeber, florian.rathgeber@gmail.com
*
* This code is licensed under the MIT License.  See the FindCUDA.cmake script
* for the text of the license.
*
* Based on code by Christopher Bruns published on Stack Overflow (CC-BY):
* http://stackoverflow.com/questions/2285185
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount, device, major = 9999, minor = 9999;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
       