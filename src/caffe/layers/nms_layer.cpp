#include <vector>
#include "caffe/layers/nms_layer.hpp"

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
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(bottom_shape);

	top_shape[3] = 3;  // X, Y, score
	top_shape[2] = max_peaks_+1; // 10 people + 1
	top_shape[1] = num_parts_;

	top[0]->Reshape(top_shape);
	workspace.Reshape(bottom_shape);

	//std::cout << "cpu here!" << std::endl;
}

template <typename Dtype>
void NmsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int num = bottom[0]->shape(0);
	//int channel = bottom[0]->shape(1);
	//std::cout << "cpu here!" << std::endl;
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);

	Dtype* dst_pointer = top[0]->mutable_cpu_data();
	Dtype* src_pointer = bottom[0]->mutable_cpu_data();
	int offset2 = oriSpatialHeight * oriSpatialWidth;
	int offset2_dst = (max_peaks_+1)*2;

	//stupid method
	for(int n = 0; n < num; n++){
		//assume only one channel
		int peakCount = 0;

		for (int y = 0; y < oriSpatialHeight; y++){
			for (int x = 0; x < oriSpatialWidth; x++){
			    Dtype value = src_pointer[n * offset2 + y*oriSpatialWidth + x];
			    if(value < threshold_) continue;
			    Dtype top = (y == 0) ? 0 : src_pointer[n * offset2 + (y-1)*oriSpatialWidth + x];
			    Dtype bottom = (y == oriSpatialHeight - 1) ? 0 : src_pointer[n * offset2 + (y+1)*oriSpatialWidth + x];
			    Dtype left = (x == 0) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x-1)];
			    Dtype right = (x == oriSpatialWidth - 1) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x+1)];
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
