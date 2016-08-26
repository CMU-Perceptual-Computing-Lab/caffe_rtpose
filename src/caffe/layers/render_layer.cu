#include "caffe/layers/render_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void RenderLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//empty
}

template <typename Dtype>
void RenderLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(RenderLayer);
}