#include "caffe/cpm/layers/nms_layer.hpp"
#include <vector>

using namespace std;

namespace caffe {

template <typename Dtype>
void NmsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NmsParameter nms_param = this->layer_param_.nms_param();
	threshold_ = nms_param.threshold();
	num_parts_ = nms_param.num_parts();
	max_peaks_ = nms_param.max_peaks();
}

template <typename Dtype>
void NmsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	std::vector<int> bottom_shape = bottom[0]->shape();
	std::vector<int> top_shape(bottom_shape);

	top_shape[3] = 3;  // X, Y, score
	top_shape[2] = max_peaks_+1; // 10 people + 1
	top_shape[1] = num_parts_;

	top[0]->Reshape(top_shape);
	workspace.Reshape(bottom_shape);

	//std::cout << "cpu here!" << std::endl;
}

template <typename Dtype>
void NmsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int num = bottom[0]->shape(0);
	//const int channel = bottom[0]->shape(1);
	//std::cout << "cpu here!" << std::endl;
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	Dtype* dst_pointer = top[0]->mutable_cpu_data();
	const Dtype* const src_pointer = bottom[0]->cpu_data();
	const int offset2 = oriSpatialHeight * oriSpatialWidth;
	const int offset2_dst = (max_peaks_+1)*2;

	//stupid method
	for(int n = 0; n < num; n++){
		//assume only one channel
		int peakCount = 0;

		for (int y = 0; y < oriSpatialHeight; y++){
			for (int x = 0; x < oriSpatialWidth; x++){
			    const Dtype value = src_pointer[n * offset2 + y*oriSpatialWidth + x];
			    if(value < threshold_) continue;
			    const Dtype top = (y == 0) ? 0 : src_pointer[n * offset2 + (y-1)*oriSpatialWidth + x];
			    const Dtype bottom = (y == oriSpatialHeight - 1) ? 0 : src_pointer[n * offset2 + (y+1)*oriSpatialWidth + x];
			    const Dtype left = (x == 0) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x-1)];
			    const Dtype right = (x == oriSpatialWidth - 1) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x+1)];
			    if(value > top && value > bottom && value > left && value > right){
			    	dst_pointer[n*offset2_dst + (peakCount + 1) * 2] = x;
			    	dst_pointer[n*offset2_dst + (peakCount + 1) * 2 + 1] = y;
			    	peakCount++;
			    }
			}
		}
		dst_pointer[n*offset2_dst] = peakCount;
	}
}

template <typename Dtype>
void NmsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(NmsLayer);
#endif

INSTANTIATE_CLASS(NmsLayer);
REGISTER_LAYER_CLASS(Nms);

} // namespace caffe
