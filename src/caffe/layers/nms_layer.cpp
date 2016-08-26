#include <vector>
#include "caffe/layers/nms_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void NmsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NmsParameter nms_param = this->layer_param_.nms_param();
	threshold = nms_param.threshold();
}

template <typename Dtype>
void NmsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(bottom_shape);

	int num_peak = this->layer_param_.nms_param().num_peak();
	top_shape[3] = 3;  // X, Y, score
	top_shape[2] = num_peak + 1; // 30 people + 1
	top_shape[1] = 1;
	
	top[0]->Reshape(top_shape);
	workspace.Reshape(bottom_shape);
}

template <typename Dtype>
void NmsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int num_peak = this->layer_param_.nms_param().num_peak();
	int num = bottom[0]->shape(0);
	//int channel = bottom[0]->shape(1);
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);

	Dtype* dst_pointer = top[0]->mutable_cpu_data();
	Dtype* src_pointer = bottom[0]->mutable_cpu_data();
	int offset2 = oriSpatialHeight * oriSpatialWidth;
	int offset2_dst = 3 * (num_peak + 1);

	//stupid method
	for(int n = 0; n < num; n++){
		//assume only one channel
		int peakCount = 0;

		for (int y = 0; y < oriSpatialHeight; y++){
			for (int x = 0; x < oriSpatialWidth; x++){	
			    Dtype value = src_pointer[n * offset2 + y*oriSpatialWidth + x];
			    if(value < threshold) continue;
			    Dtype top = (y == 0) ? 0 : src_pointer[n * offset2 + (y-1)*oriSpatialWidth + x];
			    Dtype bottom = (y == oriSpatialHeight - 1) ? 0 : src_pointer[n * offset2 + (y+1)*oriSpatialWidth + x];
			    Dtype left = (x == 0) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x-1)];
			    Dtype right = (x == oriSpatialWidth - 1) ? 0 : src_pointer[n * offset2 + y*oriSpatialWidth + (x+1)];
			    if(value > top && value > bottom && value > left && value > right){
			    	dst_pointer[n*offset2_dst + (peakCount + 1) * 3] = x;
			    	dst_pointer[n*offset2_dst + (peakCount + 1) * 3 + 1] = y;
			    	dst_pointer[n*offset2_dst + (peakCount + 1) * 3 + 2] = value;
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