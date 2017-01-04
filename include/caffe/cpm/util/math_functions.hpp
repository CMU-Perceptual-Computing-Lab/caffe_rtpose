#ifndef CAFFE_CPM_UTIL_MATH_FUNCTIONS_HPP
#define CAFFE_CPM_UTIL_MATH_FUNCTIONS_HPP

#include "caffe/common.hpp"

namespace caffe {

int updiv(const int a, const int b);
void fill_pose_net(const float* image, int width, int height,
                   float* dst, int boxsize,
                   const float* peak_pointer_gpu, std::vector<int> num_people, int limit);

}  // namespace caffe

#endif  // CAFFE_CPM_UTIL_MATH_FUNCTIONS_HPP
