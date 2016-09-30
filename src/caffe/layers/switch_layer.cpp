#include <vector>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  const SwitchParameter& switch_param = this->layer_param_.switch_param();
  switch_select_ = bottom.size()-1;
}

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const SwitchParameter& switch_param = this->layer_param_.switch_param();
  int num_axes = bottom[0]->num_axes();
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  switch_input_size_ = bottom[0]->count();
  input_count_ = bottom.size();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
  }
  top[0]->Reshape(top_shape);
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* bottom_data = bottom[switch_select_]->cpu_data();
  caffe_copy(switch_input_size_, bottom_data, top_data);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();

  if (propagate_down[switch_select_]) {
      Dtype* bottom_diff = bottom[switch_select_]->mutable_cpu_diff();
      caffe_copy(switch_input_size_, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
REGISTER_LAYER_CLASS(Switch);

}  // namespace caffe
