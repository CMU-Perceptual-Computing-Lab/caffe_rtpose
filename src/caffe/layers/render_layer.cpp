#include <vector>
#include "caffe/layers/render_layer.hpp"

using namespace std;

namespace caffe {


template <typename Dtype>
void RenderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	RenderParameter render_param = this->layer_param_.render_param();

	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(bottom_shape);

	top_shape[1] = render_param.num_output();
	
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RenderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//empty
}

template <typename Dtype>
void RenderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(RenderLayer);
#endif

INSTANTIATE_CLASS(RenderLayer);
REGISTER_LAYER_CLASS(Render);

} // namespace caffe