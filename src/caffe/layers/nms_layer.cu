#include "caffe/layers/nms_layer.hpp"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <iostream>


#define numThreadsPerBlock_1d 16
#define numThreadsPerBlock 256

namespace caffe {


template <typename Dtype>
__global__ void nms_register_kernel(Dtype* src_pointer, int* workspace, int w, int h, Dtype threshold) {
	// get pixel location (x,y)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	Dtype value = src_pointer[y*w + x];
	if(value > threshold){
		Dtype top    = (y == 0   ? 0 : src_pointer[(y-1)*w + x]);
		Dtype bottom = (y == h-1 ? 0 : src_pointer[(y+1)*w + x]);
		Dtype left   = (x == 0   ? 0 : src_pointer[y*w + (x-1)]);
		Dtype right  = (x == w-1 ? 0 : src_pointer[y*w + (x+1)]);

		if(value > top && value > bottom && value > left && value > right){
			workspace[y*w + x] = 1;
		}
		else {
			workspace[y*w + x] = 0;
		}
	}
	else {
		workspace[y*w + x] = 0;
	}
}


template <typename Dtype>
__global__ void writeResultKernel(int length, int* input, Dtype* output, int width, Dtype* image, int num_peak){
    __shared__ int local[numThreadsPerBlock+1]; // one more
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIdx < length){
        local[threadIdx.x] = input[globalIdx];
        if(threadIdx.x == numThreadsPerBlock - 1 && globalIdx != length - 1){ 
            //last thread in the block but not globally last, load one more
            local[threadIdx.x+1] = input[globalIdx+1];
        }
        __syncthreads();
        // see difference, except the globally last one
        if(globalIdx != length - 1){ 
            if(local[threadIdx.x] != local[threadIdx.x + 1]) {
                //means A[globalIdx] == A[globalIdx + 1] as the input[globalIdx]-th repeat
                int peak_index = input[globalIdx]; //0-index
                int peak_loc = globalIdx;
                int peak_loc_x = peak_loc % width;
                int peak_loc_y = peak_loc / width;

                if(peak_index < num_peak){ //limitation
	                //output[input[globalIdx]] = globalIdx;
	                output[(peak_index + 1) * 3] = peak_loc_x;
	                output[(peak_index + 1) * 3 + 1] = peak_loc_y;
	                output[(peak_index + 1) * 3 + 2] = image[peak_loc];
	            }
            }
        }
        else {
        	output[0] = input[globalIdx]; //number of peaks
        }
    }
}


template <typename Dtype>
void NmsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//Forward_cpu(bottom, top);
	int num_peak = this->layer_param_.nms_param().num_peak();

	int num = bottom[0]->shape(0);
	int height = bottom[0]->shape(2);
	int width = bottom[0]->shape(3);
	int offset = height * width;
	int offset_dst = 3 * (num_peak + 1);

	dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
	dim3 numBlocks(updiv(width, threadsPerBlock.x), updiv(height, threadsPerBlock.y));

	for(int n = 0; n < num; n++){
		int* w_pointer1 = workspace.mutable_gpu_data() + n * offset;
		Dtype* src = bottom[0]->mutable_gpu_data() + n * offset;
		Dtype* dst = top[0]->mutable_gpu_data() + n * offset_dst;

		nms_register_kernel<<<numBlocks, threadsPerBlock>>>(src, w_pointer1, width, height, threshold);//[0,0,0,0,1,0,0,0,0,1,0,0,0,0]
		//LOG(ERROR) << "register done";
		thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(w_pointer1);
		//LOG(ERROR) << "pointer done";

		// //debug
		// int* w_pointer1_local = workspace.mutable_cpu_data() + n * offset;
		// for(int i = 0; i < offset; i++){
		// 	if(w_pointer1_local[i] == 1){
		// 		std::cout << i << " ";
		// 	}
		// }
		// std::cout << std::endl;

		thrust::exclusive_scan(dev_ptr, dev_ptr + offset, dev_ptr); //[0,0,0,0,0,1,1,1,1,1,2,2,2,2]
		//LOG(ERROR) << "thrust done";
		writeResultKernel<<<updiv(offset,numThreadsPerBlock), numThreadsPerBlock>>>(offset, w_pointer1, dst, width, src, num_peak);
		//LOG(ERROR) << "write done";
	}
	//w_pointer
}

template <typename Dtype>
void NmsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(NmsLayer);

} // namespace caffe