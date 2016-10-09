#ifndef CAFFE_UTIL_RENDER_FUNCTIONS_H_
#define CAFFE_UTIL_RENDER_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
 
#define RENDER_MAX_PEOPLE 64

namespace caffe {

void render_mpi_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize,
                    float* centers, float* poses, vector<int> num_people, int part);
void render_coco_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize,
                    float* centers, float* poses, vector<int> num_people, int part, bool googly_eyes=0);
void render_coco_aff(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses,
                    vector<int> num_people, int part, int num_parts_accum);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
