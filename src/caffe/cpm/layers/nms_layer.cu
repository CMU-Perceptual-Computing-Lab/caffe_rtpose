#include "caffe/cpm/layers/nms_layer.hpp"
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "caffe/cpm/util/math_functions.hpp"  // caffe::updiv

#define NUMBER_THREADS_PER_BLOCK_1D 16
#define NUMBER_THREADS_PER_BLOCK 256


namespace caffe {


template <typename Dtype>
__global__ void nms_register_kernel(const Dtype* const src_pointer, int* workspace, const int w, const int h, const Dtype threshold) {
	// get pixel location (x,y)
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if( x>0 && x<(w-1) && y>0 && y<(h-1) ){
		const Dtype value = src_pointer[y*w + x];
		if(value > threshold){
			const Dtype top    = src_pointer[(y-1)*w + x];
			const Dtype bottom = src_pointer[(y+1)*w + x];
			const Dtype left   = src_pointer[y*w + (x-1)];
			const Dtype right  = src_pointer[y*w + (x+1)];
			const Dtype top_left = src_pointer[(y-1)*w + x-1];
			const Dtype top_right = src_pointer[(y-1)*w + x+1];
			const Dtype bottom_left = src_pointer[(y+1)*w + x-1];
			const Dtype bottom_right = src_pointer[(y+1)*w + x+1];

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
	}	else if( x==0 || x==(w-1) || y==0 || y==(h-1) ){
		workspace[y*w + x] = 0;
	}
}


template <typename Dtype>
__global__ void writeResultKernel(const int length, const int* const input, const Dtype* const src_pointer, Dtype* output, const int width, const int max_peaks){
    __shared__ int local[NUMBER_THREADS_PER_BLOCK+1]; // one more
    const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIdx < length){
      local[threadIdx.x] = input[globalIdx];
      if(threadIdx.x == NUMBER_THREADS_PER_BLOCK - 1 && globalIdx != length - 1){
          //last thread in the block but not globally last, load one more
          local[threadIdx.x+1] = input[globalIdx+1];
      }
      __syncthreads();
      // see difference, except the globally last one
      if(globalIdx != length - 1){
	      if(local[threadIdx.x] != local[threadIdx.x + 1]) {
	          //means A[globalIdx] == A[globalIdx + 1] as the input[globalIdx]-th repeat
	          const int peak_index = input[globalIdx]; //0-index
	          const int peak_loc = globalIdx;
	          const int peak_loc_x = peak_loc % width;
	          const int peak_loc_y = peak_loc / width;

	          if(peak_index < max_peaks){ //limitation
	            //output[input[globalIdx]] = globalIdx;

							// if (1) {
								float x_acc = 0.f;
								float y_acc = 0.f;
								float score_acc = 0.f;
								// int count = 0;
								for (int dy=-3;dy<4;dy++) {
									if ((peak_loc_y+dy)>0 && (peak_loc_y+dy)<width) {
										for (int dx=-3;dx<4;dx++) {
											if ((peak_loc_x+dx)>0 && (peak_loc_x+dx)<width) {
												const float score = src_pointer[(peak_loc_y+dy)*width + peak_loc_x+dx];
												const float x = peak_loc_x+dx;
												const float y = peak_loc_y+dy;
												if (score>0) {
													x_acc += x*score;
													y_acc += y*score;
													score_acc += score;
													// count += 1;
												}
											}
										}
									}
								}

								const int output_index = (peak_index + 1) * 3;
								output[output_index] = x_acc/score_acc;
	              output[output_index + 1] = y_acc/score_acc;
	              output[output_index + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
							// } else {
								// const int output_index = (peak_index + 1) * 3;
	              // output[output_index] = peak_loc_x;
	              // output[output_index + 1] = peak_loc_y;
	              // output[output_index + 2] = src_pointer[peak_loc_y*width + peak_loc_x];
							// }
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
	const int num = bottom[0]->shape(0);
	//int channel = bottom[0]->shape(1);
	const int height = bottom[0]->shape(2);
	const int width = bottom[0]->shape(3);
	const int offset = height * width;
	const int offset_dst = (max_peaks_+1)*3;

	const dim3 threadsPerBlock(NUMBER_THREADS_PER_BLOCK_1D, NUMBER_THREADS_PER_BLOCK_1D);
	const dim3 numBlocks(updiv(width, threadsPerBlock.x), updiv(height, threadsPerBlock.y));
	// const int count = bottom[0]->count();
	// std::cout << count << "\t\t"
	//           << CAFFE_GET_BLOCKS(count) << " " << CAFFE_CUDA_NUM_THREADS << "\t\t"
	//  					<< updiv(offset,NUMBER_THREADS_PER_BLOCK) << " " << NUMBER_THREADS_PER_BLOCK << "\t\t"
	//  					<< numBlocks.x << " " << threadsPerBlock.x << std::endl;

	// std::cout << "num_t: " << top[0]->shape(0) << "\t";          // = 1
  // std::cout << "channel_t: " << top[0]->shape(1) << "\t";      // = 18
	// std::cout << "height_t: " << top[0]->shape(2) << "\t";       // = 3
	// std::cout << "width_t: " << top[0]->shape(3) << "\n";        // = 65

	// std::cout << "num_b: " << bottom[0]->shape(0) << "\t";       // = 1
  // std::cout << "channel_b: " << bottom[0]->shape(1) << "\t";   // = 57
  // std::cout << "height_b: " << bottom[0]->shape(2) << "\t";    // = 368
	// std::cout << "width_b: " << bottom[0]->shape(3) << std::endl;// = 656
	for(int n = 0; n < num; n++){ // batch
		for(int c = 0; c < num_parts_; c++){
			//std::cout << "channel: " << c << std::endl;
			int* w_pointer1 = workspace.mutable_gpu_data() + n * num_parts_ * offset + c * offset;
			const Dtype* const src = bottom[0]->gpu_data() + n * num_parts_ * offset + c * offset;
			Dtype* dst = top[0]->mutable_gpu_data() + n * num_parts_ * offset_dst + c * offset_dst;
			// old model
			// if(c==14){
			// 	Dtype* src = bottom[0]->mutable_gpu_data() + n * parts_num * offset + 28 * offset;
			// }

			// This returns w_pointer1, a binary array with 0s & 1s. 1s in the local maximum positions (size = size(src))
			nms_register_kernel<<<numBlocks, threadsPerBlock>>>(src, w_pointer1, width, height, threshold_);//[0,0,0,0,1,0,0,0,0,1,0,0,0,0]
			//LOG(ERROR) << "register done";;

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

			thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(w_pointer1);
			//LOG(ERROR) << "pointer done"
			// This modifies w_pointer1, now it indicates the local maximum indexes. Format: 0,0,0,1,1,1,1,2,2,2,... First maximum: 2, second: 6, etc...
			thrust::exclusive_scan(dev_ptr, dev_ptr + offset, dev_ptr); //[0,0,0,0,0,1,1,1,1,1,2,2,2,2]
			//LOG(ERROR) << "thrust done";
			// This returns dst, with the NMS applied over it
			writeResultKernel<<<updiv(offset,NUMBER_THREADS_PER_BLOCK), NUMBER_THREADS_PER_BLOCK>>>(offset, w_pointer1, src, dst, width, max_peaks_);
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
