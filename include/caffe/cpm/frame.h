#ifndef CAFFE_CPM_FRAME_HPP_
#define CAFFE_CPM_FRAME_HPP_

#include "caffe/common.hpp"

struct Frame {
  float* data;
  float* data_for_mat;
  unsigned char* data_for_wrap;
  double commit_time;
  double preprocessed_time;
  double gpu_fetched_time;
  double gpu_computed_time;
  double postprocesse_begin_time;
  double postprocesse_end_time;
  double buffer_start_time;
  double buffer_end_time;
  int index;  // coco's id
  int numPeople;
  int video_frame_number;

  //only used for coco
  int counter;
  float scale;
  int ori_width;
  int ori_height;
  int scaled_width;
  int scaled_height;
  int padded_width; //should be 8x for net
  int padded_height; //should be 8x for net

  float* net_output; // raw heatmap in size padded_width/8 * padded_height/8
  caffe::shared_ptr<float[]> joints;
};

#endif // CAFFE_CPM_FRAME_HPP_
