#include <vector>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_gpu_data();
  //const bool kForward = true;
  const Dtype* bottom_data = bottom[switch_select_]->gpu_data();
  caffe_gpu_memcpy(switch_input_size_*sizeof(Dtype), bottom_data, top_data);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  //const bool kForward = false;
  if (propagate_down[switch_select_]) {
    Dtype* bottom_diff = bottom[switch_select_]->mutable_gpu_diff();
    caffe_gpu_memcpy(switch_input_size_*sizeof(Dtype), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
