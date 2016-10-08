#ifndef CAFFE_NMS_LAYER_HPP_
#define CAFFE_NMS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class NmsLayer : public Layer<Dtype> {
 public:
  explicit NmsLayer(const LayerParameter& param)
      : Layer<Dtype>(param), num_parts_(15), max_peaks_(20) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Nms"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline int GetMaxPeaks() const { return max_peaks_; }
  virtual inline int GetNumParts() const { return num_parts_; }
  virtual inline float GetThreshold() const { return threshold_; }
  virtual inline void SetThreshold(float threshold) { threshold_ = threshold; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<int> workspace; //only used by gpu
  //thrust::device_vector<Dtype> workspace;
  Dtype threshold_;
  int num_parts_;
  int max_peaks_;
};

}

#endif
