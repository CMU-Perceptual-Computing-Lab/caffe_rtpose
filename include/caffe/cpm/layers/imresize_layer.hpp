#ifndef CAFFE_IMRESIZE_LAYER_HPP_
#define CAFFE_IMRESIZE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ImResizeLayer : public Layer<Dtype> {
 public:
  explicit ImResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImResize"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  void setTargetDimenions(int nw, int nh);

  void SetStartScale(float astart_scale) { start_scale = astart_scale; }
  void SetScaleGap(float ascale_gap) { scale_gap = ascale_gap; }
  float GetStartScale() { return start_scale; }
  float GetScaleGap() { return scale_gap; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int targetSpatialWidth;
  int targetSpatialHeight;
  float factor;
  float start_scale;
  float scale_gap;
};

}

#endif
