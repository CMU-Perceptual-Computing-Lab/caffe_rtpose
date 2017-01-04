#include "caffe/cpm/util/math_functions.hpp"

#define NUMBER_THREADS_PER_BLOCK_1D 32

namespace caffe {

__global__ void fill_image(const float* src_pointer, int w, int h,
                           float* dst_pointer, int boxsize, const float* info, int p) {
  // get pixel location (x,y) within (boxsize, boxsize)
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(x < boxsize && y < boxsize){
    int xr_center = int(info[2*(p+1)] + 0.5);
    int yr_center = int(info[2*(p+1)+1] + 0.5);

    int x_src = xr_center - boxsize/2 + x;
    int y_src = yr_center - boxsize/2 + y;

    int offset_dst = boxsize * boxsize;
    int offset_src = w * h;

    if(x_src >= 0 && x_src < w && y_src >= 0 && y_src < h){
      dst_pointer[                 y * boxsize + x] = src_pointer[                 y_src * w + x_src];
      dst_pointer[offset_dst     + y * boxsize + x] = src_pointer[offset_src     + y_src * w + x_src];
      dst_pointer[offset_dst * 2 + y * boxsize + x] = src_pointer[offset_src * 2 + y_src * w + x_src];
    }
    else {
      dst_pointer[                 y * boxsize + x] = 0;
      dst_pointer[offset_dst     + y * boxsize + x] = 0;
      dst_pointer[offset_dst * 2 + y * boxsize + x] = 0;
    }
  }
}

__global__ void fill_gassian(float* dst_pointer, int boxsize, float sigma){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(x < boxsize && y < boxsize){
    float center_x, center_y;
    center_x = center_y = boxsize / 2;
    float d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
    float exponent = d2 / 2.0 / sigma / sigma;
    if(exponent > 4.6052){ //ln(100) = -ln(1%)
      dst_pointer[y * boxsize + x] = 0;
    }
    else {
      dst_pointer[y * boxsize + x] = exp(-exponent);
    }
  }
}

void fill_pose_net(const float* image, int width, int height,
                   float* dst, int boxsize,
                   const float* peak_pointer_gpu, vector<int> num_people, int limit){
  //image            in width * height * 3 * N
  //dst              in boxsize * boxsize * 4 * (P1+P2+...+PN)
  //peak_pointer_gpu in 2 * 11 * 1 * N
  //num_people has length P, indicating P1, ..., PN
  CHECK(0) << "FIX THIS FUNCTION";
  int N = num_people.size();
  int count = 0;
  bool full = false;
  int offset_src = width * height * 3;
  int offset_dst_2 = boxsize * boxsize;
  int offset_info = 22;
  dim3 threadsPerBlock(NUMBER_THREADS_PER_BLOCK_1D, NUMBER_THREADS_PER_BLOCK_1D);
  dim3 numBlocks(updiv(boxsize, threadsPerBlock.x), updiv(boxsize, threadsPerBlock.y));

  for(int i = 0; i < N; i++){
    //LOG(ERROR) << "copying " << num_people[i] << " people.";
    for(int p = 0; p < num_people[i]; p++){
      fill_image<<<threadsPerBlock, numBlocks>>>(image + i * offset_src, width, height,
                                                 dst + count * (4 * offset_dst_2), boxsize,
                                                 peak_pointer_gpu + i * offset_info, p);
      //src, w, h, dst, boxsize, info, p

      fill_gassian<<<threadsPerBlock, numBlocks>>>(dst + count * (4 * offset_dst_2) + 3 * offset_dst_2, boxsize, 21);
      //dst, boxsize

      count++;
      if(count >= limit){
        full = true;
        break;
      }
    }
    if(full) break;
  }
  cudaDeviceSynchronize();
}

}  // namespace caffe
