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

	if( x==0 || x==(w-1) || y>0 || y==(h-1) ){
		workspace[y*w + x] = 0;
	}

	if( x>1 && x<(w-1) && y>1 && y<(h-1) ){
		Dtype value = src_pointer[y*w + x];
		if(value > threshold){
			Dtype top    = src_pointer[(y-1)*w + x];
			Dtype bottom = src_pointer[(y+1)*w + x];
			Dtype left   = src_pointer[y*w + (x-1)];
			Dtype right  = src_pointer[y*w + (x+1)];
			Dtype top_left = src_pointer[(y-1)*w + x-1];
			Dtype top_right = src_pointer[(y-1)*w + x+1];
			Dtype bottom_left = src_pointer[(y+1)*w + x-1];
			Dtype bottom_right = src_pointer[(y+1)*w + x+1];

			if(value > top && value > bottom && value > left && value > right && value > top_left
				&& value > bottom_left && value > bottom_right && value > top_right ){
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
}


template <typename Dtype>
__global__ void writeResultKernel(int length, int* input, Dtype* src_pointer, Dtype* output, int width, int max_peaks){
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

                if(peak_index < max_peaks){ //limitation
	                //output[input[globalIdx]] = globalIdx;

								if (1) {
									float x_acc = 0;
									float y_acc = 0;
									float score_acc = 0;
									int count = 0;
									for (int dy=-3;dy<4;dy++) {
										if ((peak_loc_y+dy)>0 && (peak_loc_y+dy)<width) {
											for (int dx=-3;dx<4;dx++) {
												if ((peak_loc_x+dx)>0 && (peak_loc_x+dx)<width) {
														float score = src_pointer[(peak_loc_y+dy)*width + peak_loc_x+dx];
														float x = peak_loc_x+dx;
														float y = peak_loc_y+dy;
														if (score>0) {
															x_acc += x*score;
															y_acc += y*score;
															score_acc += score;
															count += 1;
														}
												}
											}
										}
									}

									output[(peak_index + 1) * 3] = x_acc/score_acc;
	                output[(peak_index + 1) * 3 + 1] = y_acc/score_acc;
	                output[(peak_index + 1) * 3 + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
								} else {
	                output[(peak_index + 1) * 3] = peak_loc_x;
	                output[(peak_index + 1) * 3 + 1] = peak_loc_y;
	                output[(peak_index + 1) * 3 + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
								}
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
	int num = bottom[0]->shape(0);
	//int channel = bottom[0]->shape(1);
	int height = bottom[0]->shape(2);
	int width = bottom[0]->shape(3);
	int offset = height * width;
	int offset_dst = (max_peaks_+1)*3;

	dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
	dim3 numBlocks(updiv(width, threadsPerBlock.x), updiv(height, threadsPerBlock.y));

	//std::cout << "channel: " << channel << std::endl;
	for(int n = 0; n < num; n++){ // batch
		for(int c = 0; c < num_parts_; c++){
			//std::cout << "channel: " << c << std::endl;
			int* w_pointer1 = workspace.mutable_gpu_data() + n * num_parts_ * offset + c * offset;
			Dtype* src = bottom[0]->mutable_gpu_data() + n * num_parts_ * offset + c * offset;
			Dtype* dst = top[0]->mutable_gpu_data() + n * num_parts_ * offset_dst + c * offset_dst;
			// old model
			// if(c==14){
			// 	Dtype* src = bottom[0]->mutable_gpu_data() + n * parts_num * offset + 28 * offset;
			// }

			nms_register_kernel<<<numBlocks, threadsPerBlock>>>(src, w_pointer1, width, height, threshold_);//[0,0,0,0,1,0,0,0,0,1,0,0,0,0]
			//LOG(ERROR) << "register done";
			thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(w_pointer1);
			//LOG(ERROR) << "pointer done";

			//debug
			// if(c==3){
			// 	char filename[50];
			// 	sprintf(filename, "work%02d.txt", c);
			//     std::ofstream fout(filename);
			// 	int* w_pointer1_local = workspace.mutable_cpu_data() + n * parts_num * offset + c * offset;
			// 	for(int y = 0; y < height; y++){
			// 		for(int x = 0; x < width; x++){
			// 			fout << w_pointer1_local[y*width + x] << "\t";
			// 		}
			// 		fout<< std::endl;
			// 	}
			// 	fout.close();
			// }

			thrust::exclusive_scan(dev_ptr, dev_ptr + offset, dev_ptr); //[0,0,0,0,0,1,1,1,1,1,2,2,2,2]
			//LOG(ERROR) << "thrust done";
			writeResultKernel<<<updiv(offset,numThreadsPerBlock), numThreadsPerBlock>>>(offset, w_pointer1, src, dst, width, max_peaks_);
			//LOG(ERROR) << "write done";
		}
	}
	//w_pointer
}

template <typename Dtype>
void NmsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(NmsLayer);

} // namespace caffe
