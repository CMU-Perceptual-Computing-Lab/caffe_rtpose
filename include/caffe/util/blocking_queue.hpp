#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>

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
};


namespace caffe {

template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue();

  void push(const T& t);

  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
  T peek();

  size_t size() const;

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
  class sync;

  std::queue<T> queue_;
  shared_ptr<sync> sync_;

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe

#endif
