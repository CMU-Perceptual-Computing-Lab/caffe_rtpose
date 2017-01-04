#include "caffe/cpm/layers/imresize_layer.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void ImResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	ImResizeParameter imresize_param = this->layer_param_.imresize_param();
	targetSpatialWidth = imresize_param.target_spatial_width(); //temporarily
	targetSpatialHeight = imresize_param.target_spatial_height();
	start_scale = imresize_param.start_scale();
	scale_gap = imresize_param.scale_gap();
}

template <typename Dtype>
void ImResizeLayer<Dtype>::setTargetDimenions(int nw, int nh){
	targetSpatialWidth = nw;
	targetSpatialHeight = nh;
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	std::vector<int> bottom_shape = bottom[0]->shape();
	std::vector<int> top_shape(bottom_shape);

	ImResizeParameter imresize_param = this->layer_param_.imresize_param();

	if(imresize_param.factor() != 0){
		top_shape[3] = top_shape[3] * imresize_param.factor();
		top_shape[2] = top_shape[2] * imresize_param.factor();
		setTargetDimenions(top_shape[3], top_shape[2]);
	}
	else {
		top_shape[3] = targetSpatialWidth;
		top_shape[2] = targetSpatialHeight;
	}
	top_shape[0] = 1;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int num = bottom[0]->shape(0);
	const int channel = bottom[0]->shape(1);
	const int oriSpatialHeight = bottom[0]->shape(2);
	const int oriSpatialWidth = bottom[0]->shape(3);

	//stupid method
	for(int n = 0; n < num; n++){
		for(int c = 0; c < channel; c++){
			//fill src
			cv::Mat src(oriSpatialWidth, oriSpatialHeight, CV_32FC1);
			const int src_offset2 = oriSpatialHeight * oriSpatialWidth;
			const int src_offset3 = src_offset2 * channel;
			const Dtype* const src_pointer = bottom[0]->cpu_data();
			for (int x = 0; x < oriSpatialWidth; x++){
				for (int y = 0; y < oriSpatialHeight; y++){
				    src.at<Dtype>(x,y) = src_pointer[n*src_offset3 + c*src_offset2 + y*oriSpatialWidth + x];
				}
			}
			//resize
			cv::Mat dst(targetSpatialWidth, targetSpatialHeight, CV_32FC1);
			cv::resize(src, dst, dst.size(), 0, 0, CV_INTER_CUBIC);
			//fill top
			const int dst_offset2 = targetSpatialHeight * targetSpatialWidth;
			const int dst_offset3 = dst_offset2 * channel;
			Dtype* dst_pointer = top[0]->mutable_cpu_data();
			for (int x = 0; x < targetSpatialWidth; x++){
				for (int y = 0; y < targetSpatialHeight; y++){
				    dst_pointer[n*dst_offset3 + c*dst_offset2 + y*targetSpatialWidth + x] = dst.at<Dtype>(x,y);
				}
			}
		}
	}
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ImResizeLayer);
#endif

INSTANTIATE_CLASS(ImResizeLayer);
REGISTER_LAYER_CLASS(ImResize);

}  // namespace caffe
