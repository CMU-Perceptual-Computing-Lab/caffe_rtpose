#include "caffe/layers/maximum_layer.hpp"
#include "caffe/cpm/util/math_functions.hpp"  // caffe::updiv

namespace caffe {

#define numThreadsPerBlock 256

template <typename Dtype>
__global__ void rowReduceMax_kernel(Dtype* d_scoreMap, Dtype* d_intermediate, int num_template,
	                                int image_width, int image_height){

	const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id (0 - 32 * total_threads)
	int duty_row = i >> 5; //i / 32; //duty row indexing is skipping [301 - 512] row
    int duty_template = duty_row / image_height;
    int h = duty_row % image_height; //duty_row_local

    if (duty_row < num_template * image_width && h < image_height) {
        int offset = i % 32;
	    int warp_id = (i % 256) / 32; //0-7
	    int pitch1 = image_width;
	    int pitch2 = image_height * pitch1;

	    __shared__ Dtype row_maxVals[8][32]; //256
	    __shared__ int row_xargmax[8][32];

	    Dtype maxVal = -1e10;
	    int xargmax = 0;
	    int index = duty_template * pitch2 + h * pitch1;
	    for(int w = offset; w < image_width; w += 32){
	    	Dtype read = d_scoreMap[index+w];
	    	//if(duty_row == 3*image_height && offset == 0){
	    	//	printf("d_scoreMap[%d] = %f\n", index, read);
	    	//}
	    	if(read > maxVal){
	    		maxVal = read;
	    		xargmax = w;
	    	}
	    }
	    row_maxVals[warp_id][offset] = maxVal;
	    row_xargmax[warp_id][offset] = xargmax;
	    //__syncthreads(); //ensure share memory are all filled

	    // if(offset == 0 && duty_row == 0){
	    // 	for(int i = 0; i < 32; i++) printf("%f ", row_minVals[warp_id][i]);
	    // 	printf("\n");
	    // 	for(int i = 0; i < 32; i++) printf("%d ", row_xargmin[warp_id][i]);
	    // 	printf("\n");
	    // }
	    // if(duty_row == 3*image_height){
	    // 	printf("thread %d: minval %f, index %d, warp_id %d\n", offset, minVal, xargmin, warp_id);
	    // }

	    if(offset % 2 == 0){
	    	if(row_maxVals[warp_id][offset] < row_maxVals[warp_id][offset+1]){
	    		row_maxVals[warp_id][offset] = row_maxVals[warp_id][offset+1];
	    		row_xargmax[warp_id][offset] = row_xargmax[warp_id][offset+1];
	    	}
	    }
	    if(offset % 4 == 0){
	    	if(row_maxVals[warp_id][offset] < row_maxVals[warp_id][offset+2]){
	    		row_maxVals[warp_id][offset] = row_maxVals[warp_id][offset+2];
	    		row_xargmax[warp_id][offset] = row_xargmax[warp_id][offset+2];
	    	}
	    }
	    if(offset % 8 == 0){
	    	if(row_maxVals[warp_id][offset] < row_maxVals[warp_id][offset+4]){
	    		row_maxVals[warp_id][offset] = row_maxVals[warp_id][offset+4];
	    		row_xargmax[warp_id][offset] = row_xargmax[warp_id][offset+4];
	    	}
	    }
	    if(offset % 16 == 0){
	    	if(row_maxVals[warp_id][offset] < row_maxVals[warp_id][offset+8]){
	    		row_maxVals[warp_id][offset] = row_maxVals[warp_id][offset+8];
	    		row_xargmax[warp_id][offset] = row_xargmax[warp_id][offset+8];
	    	}
	    }
	    if(offset % 32 == 0){
	    	if(row_maxVals[warp_id][offset] < row_maxVals[warp_id][offset+16]){
	    		row_maxVals[warp_id][offset] = row_maxVals[warp_id][offset+16];
	    		row_xargmax[warp_id][offset] = row_xargmax[warp_id][offset+16];
	    	}
 			//                                <--- pitch 2 --->       pitch 1
	    	int index_inter = duty_template * (image_height * 2) + h * 2;
	    	d_intermediate[index_inter] = row_xargmax[warp_id][0];
	    	d_intermediate[index_inter + 1] = (Dtype)row_maxVals[warp_id][0];
	    }
    }
}

template <typename Dtype>
__global__ void columnReduceMax_kernel(Dtype* d_intermediate, Dtype* d_final,
	                                int num_template, int image_height){
	//32 threads per template
	const int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread id
	int duty_template = i >> 5; //i / 32;

	if(duty_template < num_template){
	    int offset = i % 32;
	    int warp_id = (i % 256) / 32; //0-7
	    int pitch1 = 2; // on d_intermediate
	    int pitch2 = image_height * pitch1;

	    __shared__ Dtype col_maxVals[8][32];//256
	    __shared__ int col_xargmax[8][32];
	    __shared__ int col_yargmax[8][32];

	    Dtype maxVal = -1e10;
	    int xargmax = 0;
	    int yargmax = 0;

	    for(int h = offset; h < image_height; h += 32){
	    	int index = duty_template * pitch2 + h * pitch1;
	    	Dtype read = d_intermediate[index+1]; // value
	    	if(read > maxVal){
	    		maxVal = read;
	    		xargmax = (int)(d_intermediate[index] + 0.5); // x-index
	    		yargmax = h;
	    	}
	    }
	    // if(duty_template == 0){
	    // 	printf("thread %d: minval %f, xargmin %d, yargmin %d, warp_id %d\n",
	    // 		offset, minVal, xargmin, yargmin, warp_id);
	    // }
	    col_maxVals[warp_id][offset] = maxVal;
	    col_xargmax[warp_id][offset] = xargmax;
	    col_yargmax[warp_id][offset] = yargmax;

	    if(offset % 2 == 0){
	    	if(col_maxVals[warp_id][offset] < col_maxVals[warp_id][offset+1]){
	    		col_maxVals[warp_id][offset] = col_maxVals[warp_id][offset+1];
	    		col_xargmax[warp_id][offset] = col_xargmax[warp_id][offset+1];
	    		col_yargmax[warp_id][offset] = col_yargmax[warp_id][offset+1];
	    	}
	    }
	    if(offset % 4 == 0){
	    	if(col_maxVals[warp_id][offset] < col_maxVals[warp_id][offset+2]){
	    		col_maxVals[warp_id][offset] = col_maxVals[warp_id][offset+2];
	    		col_xargmax[warp_id][offset] = col_xargmax[warp_id][offset+2];
	    		col_yargmax[warp_id][offset] = col_yargmax[warp_id][offset+2];
	    	}
	    }
	    if(offset % 8 == 0){
	    	if(col_maxVals[warp_id][offset] < col_maxVals[warp_id][offset+4]){
	    		col_maxVals[warp_id][offset] = col_maxVals[warp_id][offset+4];
	    		col_xargmax[warp_id][offset] = col_xargmax[warp_id][offset+4];
	    		col_yargmax[warp_id][offset] = col_yargmax[warp_id][offset+4];
	    	}
	    }
	    if(offset % 16 == 0){
	    	if(col_maxVals[warp_id][offset] < col_maxVals[warp_id][offset+8]){
	    		col_maxVals[warp_id][offset] = col_maxVals[warp_id][offset+8];
	    		col_xargmax[warp_id][offset] = col_xargmax[warp_id][offset+8];
	    		col_yargmax[warp_id][offset] = col_yargmax[warp_id][offset+8];
	    	}
	    }
	    if(offset % 32 == 0){
	    	if(col_maxVals[warp_id][offset] < col_maxVals[warp_id][offset+16]){
	    		col_maxVals[warp_id][offset] = col_maxVals[warp_id][offset+16];
	    		col_xargmax[warp_id][offset] = col_xargmax[warp_id][offset+16];
	    		col_yargmax[warp_id][offset] = col_yargmax[warp_id][offset+16];
	    	}
	    	//d_final is n * c * 1 * 3
	    	int pitch_on_final = 3;
	    	d_final[duty_template * pitch_on_final] = col_xargmax[warp_id][0];
	    	d_final[duty_template * pitch_on_final + 1] = col_yargmax[warp_id][0];
	    	d_final[duty_template * pitch_on_final + 2] = col_maxVals[warp_id][0];
	    }
	}
}


template <typename Dtype>
void MaximumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//Forward_cpu(bottom, top);
	//magic here
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);
	int num = bottom[0]->shape(0);
	int channel = bottom[0]->shape(1);

	int num_template = num * channel;

	//Phase 1: 32 threads per row for row_reduce
	int total_rows = num * channel * oriSpatialHeight;
	rowReduceMax_kernel<<<updiv(total_rows * 32, numThreadsPerBlock), numThreadsPerBlock>>>
               					(bottom[0]->mutable_gpu_data(), rowReduce.mutable_gpu_data(), num_template,
	                                oriSpatialWidth, oriSpatialHeight);


	//Phase 2: 32 threads per channel for column reduce
    int total_threads = 32 * num_template;
    //printf("Launching columnReduceMin_kernel<<<%d,%d>>>\n", updiv(total_threads, numThreadsPerBlock), numThreadsPerBlock);
    columnReduceMax_kernel<<<updiv(total_threads, numThreadsPerBlock), numThreadsPerBlock>>>
                (rowReduce.mutable_gpu_data(), top[0]->mutable_gpu_data(), num_template, oriSpatialHeight);
}

template <typename Dtype>
void MaximumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(MaximumLayer);

} // namespace caffe
