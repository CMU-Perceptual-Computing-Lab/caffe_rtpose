#include "caffe/layers/imresize_layer.hpp"

#define numThreadsPerBlock_1d 16
#define numThreadsPerBlock 256

namespace caffe {

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype &out, Dtype &v0, Dtype &v1, Dtype &v2, Dtype &v3, float dx) {
    // Dtype a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // Dtype b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // Dtype c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}


template <typename Dtype>
__global__ void imresize_cubic_kernel(Dtype* src_pointer, Dtype* dst_pointer, 
	                                  int ow, int oh, int tw, int th){
	// get pixel location (x,y)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int tid = threadIdx.y * blockDim.x + threadIdx.x;

	//get (min_x,max_x) (min_y,max_y)
	// int min_x = (blockIdx.x * blockDim.x);
	// int max_x = (blockIdx.x * blockDim.x) + blockDim.x - 1;
	// int min_y = (blockIdx.y * blockDim.y);
	// int max_y = (blockIdx.y * blockDim.y) + blockDim.y - 1;

	// int min_x_ori = min_x * (float(ow) / tw);
	// int max_x_ori = max_x * (float(ow) / tw);
	// int min_y_ori = min_y * (float(oh) / th);
	// int max_y_ori = max_y * (float(oh) / th);

	// min_x_ori = (min_x_ori - 1 < 0) ? min_x_ori : (min_x_ori - 1);
	// max_x_ori = (max_x_ori + 2 >= ow) ? (max_x_ori + 1 >= ow ? max_x_ori : max_x_ori+1) : (max_x_ori + 2);
	// min_y_ori = (min_y_ori - 1 < 0) ? min_y_ori : (min_y_ori - 1);
	// max_y_ori = (max_y_ori + 2 >= oh) ? (max_y_ori + 1 >= oh ? max_y_ori : max_y_ori+1) : (max_y_ori + 2);

	// // load into shared memory: fixed for 7x7
	// __shared__ Dtype shared[7][7];
	// if(threadIdx.x < 7 && threadIdx.y < 7 && min_x_ori + threadIdx.x < ow && min_y_ori + threadIdx.y < oh) {
	// 	int x_ref = min_x_ori + threadIdx.x;
	// 	int y_ref = min_y_ori + threadIdx.y;
	// 	shared[threadIdx.x][threadIdx.y] = src_pointer[y_ref * ow + x_ref];
	// }

	// begin compute
	if(x < tw && y < th) {
		float offset_x = tw/float(ow)/2 - 0.5;
		float offset_y = th/float(oh)/2 - 0.5;
		float x_on_ori = (x - offset_x) * (float(ow) / tw);  //3.5 is for 8x enlarge
		float y_on_ori = (y - offset_y) * (float(oh) / th);

		int x_nei[4];
		x_nei[1] = int(x_on_ori + 1e-5);
		x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
		x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
		x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
		x_nei[3] = (x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1);
		float dx = x_on_ori - x_nei[1];

		int y_nei[4];
		y_nei[1] = int(y_on_ori + 1e-5);
		y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
		y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
		y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
		y_nei[3] = (y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1);
		float dy = y_on_ori - y_nei[1];

		// if(x == 16 && y == 16){
		// 	printf("x: %d %d %d %d, y: %d %d %d %d\n", x_nei[0],x_nei[1],x_nei[2],x_nei[3], y_nei[0],y_nei[1],y_nei[2],y_nei[3]);
		// }

		Dtype temp[4];
		for(int i = 0; i < 4; i++){
			cubic_interpolation(temp[i], src_pointer[y_nei[i]*ow + x_nei[0]], 
				                         src_pointer[y_nei[i]*ow + x_nei[1]], 
				                         src_pointer[y_nei[i]*ow + x_nei[2]],
				                         src_pointer[y_nei[i]*ow + x_nei[3]], dx);
			// cubic_interpolation(temp[i], shared[x_nei[0]-min_x_ori][y_nei[i]-min_y_ori], 
			// 	                         shared[x_nei[1]-min_x_ori][y_nei[i]-min_y_ori], 
			// 	                         shared[x_nei[2]-min_x_ori][y_nei[i]-min_y_ori],
			// 	                         shared[x_nei[3]-min_x_ori][y_nei[i]-min_y_ori], dx);
		}
		cubic_interpolation(dst_pointer[y*tw+x], temp[0], temp[1], temp[2], temp[3], dy);
		//cubic_interpolation(temp[5], temp[0], temp[1], temp[2], temp[3], dy);
	}
}



template <typename Dtype>
__global__ void imresize_cubic_kernel(Dtype* src_ptr, Dtype* dst_pointer, int src_offset, int num, float scale_gap, 
									  float start_scale, int oriSpatialWidth, int oriSpatialHeight, int tw, int th){
	// get pixel location (x,y)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// begin compute
	if(x < tw && y < th) {
		Dtype d_temp = 0;
		Dtype sum = 0;
		for(int n = 0; n < num; n++){
			int padw = floor(oriSpatialWidth /2 * (1-start_scale + n * scale_gap) ); //n
			int padh = floor(oriSpatialHeight /2 * (1-start_scale + n * scale_gap) );
			int ow = oriSpatialWidth - 2*padw;
			int oh = oriSpatialHeight - 2*padh;
			//LOG(ERROR) << "GPU padw " << padw << " padh " << padh;
			Dtype* src_pointer = src_ptr + n * src_offset;

			float offset_x = tw/float(ow)/2 - 0.5;
			float offset_y = th/float(oh)/2 - 0.5;
			float x_on_ori = (x - offset_x) * (float(ow) / tw);  //3.5 is for 8x enlarge
			float y_on_ori = (y - offset_y) * (float(oh) / th);

			int x_nei[4];
			x_nei[1] = int(x_on_ori + 1e-5);
			x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
			x_nei[0] = ((x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1)) + padw;
			x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
			x_nei[3] = ((x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1)) + padw;
			float dx = x_on_ori - x_nei[1];
			x_nei[1] = x_nei[1] + padw;
			x_nei[2] = x_nei[2] + padw;

			int y_nei[4];
			y_nei[1] = int(y_on_ori + 1e-5);
			y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
			y_nei[0] = ((y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1)) + padh;
			y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
			y_nei[3] = ((y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1)) + padh;
			float dy = y_on_ori - y_nei[1];
			y_nei[1] = y_nei[1] + padh;
			y_nei[2] = y_nei[2] + padh;

			Dtype temp[4];
			for(int i = 0; i < 4; i++){
				cubic_interpolation(temp[i], src_pointer[y_nei[i]*(ow+2*padw) + x_nei[0]], 
					                         src_pointer[y_nei[i]*(ow+2*padw) + x_nei[1]], 
					                         src_pointer[y_nei[i]*(ow+2*padw)+ x_nei[2]],
					                         src_pointer[y_nei[i]*(ow+2*padw) + x_nei[3]], dx);
			}
			//cubic_interpolation(dst_pointer[y*tw+x], temp[0], temp[1], temp[2], temp[3], dy);
			cubic_interpolation(d_temp, temp[0], temp[1], temp[2], temp[3], dy);
			sum = sum + d_temp;
		}
		dst_pointer[y*tw+x] = sum / num;
	}
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//Forward_cpu(bottom, top);
	//magic here
	Dtype* src_pointer = bottom[0]->mutable_gpu_data();
	Dtype* dst_pointer = top[0]->mutable_gpu_data();
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);
	int num = bottom[0]->shape(0); //scale number
	int channel = bottom[0]->shape(1);
	//LOG(ERROR) << "GPU num " << num << " channel " << channel;
	//LOG(ERROR) << "top[0] " << top[0]->shape(0) << " top[0]->shape(1) " << top[0]->shape(1);

	dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
	dim3 numBlocks(updiv(targetSpatialWidth, threadsPerBlock.x), updiv(targetSpatialHeight, threadsPerBlock.y));
	int offset_src = oriSpatialHeight * oriSpatialWidth;
	int offset_dst = targetSpatialWidth * targetSpatialHeight;
	//int sm_width = numThreadsPerBlock_1d / (float(targetSpatialWidth) / oriSpatialWidth) + 3;
	//int sm_height = numThreadsPerBlock_1d / (float(targetSpatialHeight) / oriSpatialHeight) + 3;

	
		for(int c = 0; c < channel; c++){
			// imresize_cubic_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + (n * channel + c) * offset_src, 
			// 	                                                dst_pointer + (n * channel + c) * offset_dst, 
			// 	                                    			oriSpatialWidth, oriSpatialHeight,
			// 	                                    			targetSpatialWidth, targetSpatialHeight);
			imresize_cubic_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + c * offset_src, dst_pointer + c * offset_dst, 
																channel* offset_src, num, scale_gap, start_scale, 
				                                    			oriSpatialWidth, oriSpatialHeight,
				                                    			targetSpatialWidth, targetSpatialHeight);
			//LOG(ERROR) << "GPU oriSpatialHeight - 2*padh " << oriSpatialHeight - 2*padh;
		}

		//fuse_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + (n * channel + c) * offset_src, 
		//		                                                targetSpatialWidth, targetSpatialHeight);
}

template <typename Dtype>
void ImResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ImResizeLayer);

} // namespace caffe