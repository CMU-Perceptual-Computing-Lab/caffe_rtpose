#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#define numThreadsPerBlock_1d 32
#define numThreadsPerBlock 1024


namespace caffe {


// **** kernel code ****
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

__global__ void render_pose(float* dst_pointer, float* image_ref, int w, int h, 
                             float* centers, float* poses, int boxsize, 
                             int num_people, float threshold){
  //poses has length 3 * 15 * num_people

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int plotted = 0;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float shared_poses[450];
  __shared__ float shared_centers[20];

  if(global_idx < (num_people + 1) * 2){
    shared_centers[global_idx] = centers[global_idx];
  }
  if(global_idx < num_people * 15){
    shared_poses[3*global_idx] = poses[3*global_idx];
    shared_poses[3*global_idx+1] = poses[3*global_idx+1];
    shared_poses[3*global_idx+2] = poses[3*global_idx+2];
  }

  __syncthreads();

  int nlimb = 9;
  int limb[18] = {0, 1,  2, 3,  3, 4,  5,  6,  
                  6, 7,  8, 9,  9, 10, 11, 12, 12, 13};
  int color[27] =   {255,   0, 0,
                     255, 170, 0,
                     170, 255, 0,
                       0, 255, 0,
                       0, 255, 170,
                       0, 170, 255,
                       0, 0,   255,
                     170, 0,   255,
                     255, 0,   170};

  if(x < w && y < h){
  //if(x == 0 && y == 0){
    float b, g, r;

    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = (image_ref[y*w + x] + 0.5) * 256;
    g =  (image_ref[w*h + y*w + x] + 0.5) * 256;
    r =  (image_ref[2*w*h + y*w + x] + 0.5) * 256;


    for(int p = 0; p < num_people; p++){
      float center_x = shared_centers[2*(p+1)];
      float center_y = shared_centers[2*(p+1)+1];

      for(int l = 0; l < nlimb; l++){
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        float x_a = shared_poses[p*45 + part_a*3] - boxsize/2 + center_x;
        float x_b = shared_poses[p*45 + part_b*3] - boxsize/2 + center_x;
        float y_a = shared_poses[p*45 + part_a*3 + 1] - boxsize/2 + center_y;
        float y_b = shared_poses[p*45 + part_b*3 + 1] - boxsize/2 + center_y;
        float value_a = shared_poses[p*45 + part_a*3 + 2];
        float value_b = shared_poses[p*45 + part_b*3 + 2];
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          float angle = atan2f(y_b - y_a, x_b - x_a);
          float sine = sinf(angle);
          float cosine = cosf(angle);
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);
          float b_sqrt = 25; //fixed

          float A = cosine * (x - x_p) + sine * (y - y_p);
          float B = sine * (x - x_p) - cosine * (y - y_p);
          float judge = A * A / a_sqrt + B * B / b_sqrt;

          if(judge <= 1){
            b = 0.4 * b + 0.6 * color[l*3+2];
            g = 0.4 * g + 0.6 * color[l*3+1];
            r = 0.4 * r + 0.6 * color[l*3];
            //plotted = 1;
          }
        }
      }

      for(int i = 0; i < 14; i++) { //for every point
        float local_x = shared_poses[p*45 + i*3];
        float local_y = shared_poses[p*45 + i*3 + 1];
        float value = shared_poses[p*45 + i*3 + 2];
        float pose_x_on_image = local_x - boxsize/2 + center_x;
        float pose_y_on_image = local_y - boxsize/2 + center_y;

        // if(x==0 && y==0){
        //   printf("p = %d, i = %d, center_x = %f, center_y = %f\n", p, i, center_x, center_y);
        //   printf("p = %d, i = %d, local_x = %f, local_y = %f, value = %f\n", p, i, local_x, local_y, value);
        //   printf("p = %d, i = %d, pose_x_on_image = %f, pose_y_on_image = %f\n", p, i, pose_x_on_image, pose_y_on_image);
        // }
        if(value > threshold) {
          if((x - pose_x_on_image) * (x - pose_x_on_image) + (y - pose_y_on_image) * (y - pose_y_on_image) <= 4){
            b = g = r = 0;
          }
        }
      }
    }
    dst_pointer[y*w + x] = b; //plot dot
    dst_pointer[w*h + y*w + x] = g;
    dst_pointer[2*w*h + y*w + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}

__global__ void render_pose_website(float* dst_pointer, int w_canvas, int h_canvas, float ratio_to_origin, 
                             float* centers, float* poses, int boxsize, 
                             int num_people, float threshold){
  //poses has length 3 * 15 * num_people

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int plotted = 0;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float shared_poses[450];
  __shared__ float shared_centers[20];

  if(global_idx < (num_people + 1) * 2){
    shared_centers[global_idx] = centers[global_idx];
  }
  if(global_idx < num_people * 15){
    shared_poses[3*global_idx] = poses[3*global_idx]; //x
    shared_poses[3*global_idx+1] = poses[3*global_idx+1]; //y
    shared_poses[3*global_idx+2] = poses[3*global_idx+2]; //v
  }

  __syncthreads();

  int nlimb = 9;
  int limb[18] = {0, 1,  2, 3,  3, 4,  5,  6,  
                  6, 7,  8, 9,  9, 10, 11, 12, 12, 13};
  int color[27] =   {255,   0, 0,
                     255, 170, 0,
                     170, 255, 0,
                       0, 255, 0,
                       0, 255, 170,
                       0, 170, 255,
                       0, 0,   255,
                     170, 0,   255,
                     255, 0,   170};
  float offset = ratio_to_origin * 0.5 - 0.5;
  float radius = h_canvas / 200.0f;
  float stickwidth = h_canvas / 60.0f;

  if(x < w_canvas && y < h_canvas){
  //if(x == 0 && y == 0){
    float b, g, r;

    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < num_people; p++){
      float center_x = shared_centers[2*(p+1)];
      float center_y = shared_centers[2*(p+1)+1];

      for(int l = 0; l < nlimb; l++){
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        float x_a = (shared_poses[p*45 + part_a*3] - boxsize/2 + center_x) * ratio_to_origin + offset;
        float x_b = (shared_poses[p*45 + part_b*3] - boxsize/2 + center_x) * ratio_to_origin + offset;
        float y_a = (shared_poses[p*45 + part_a*3 + 1] - boxsize/2 + center_y) * ratio_to_origin + offset;
        float y_b = (shared_poses[p*45 + part_b*3 + 1] - boxsize/2 + center_y) * ratio_to_origin + offset;
        float value_a = shared_poses[p*45 + part_a*3 + 2];
        float value_b = shared_poses[p*45 + part_b*3 + 2];
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          float angle = atan2f(y_b - y_a, x_b - x_a);
          float sine = sinf(angle);
          float cosine = cosf(angle);
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);
          float b_sqrt = stickwidth * stickwidth; //fixed

          float A = cosine * (x - x_p) + sine * (y - y_p);
          float B = sine * (x - x_p) - cosine * (y - y_p);
          float judge = A * A / a_sqrt + B * B / b_sqrt;

          if(judge <= 1){
            b = 0.4 * b + 0.6 * color[l*3+2];
            g = 0.4 * g + 0.6 * color[l*3+1];
            r = 0.4 * r + 0.6 * color[l*3];
            //plotted = 1;
          }
        }
      }

      for(int i = 0; i < 14; i++) { //for every point
        float local_x = shared_poses[p*45 + i*3];
        float local_y = shared_poses[p*45 + i*3 + 1];
        float value = shared_poses[p*45 + i*3 + 2];
        float pose_x_on_image = (local_x - boxsize/2 + center_x) * ratio_to_origin + offset;
        float pose_y_on_image = (local_y - boxsize/2 + center_y) * ratio_to_origin + offset;

        // if(x==1279 && y==719){
        //   printf("p = %d, i = %d, center_x = %f, center_y = %f\n", p, i, center_x, center_y);
        //   printf("p = %d, i = %d, local_x = %f, local_y = %f, value = %f\n", p, i, local_x, local_y, value);
        //   printf("p = %d, i = %d, pose_x_on_image = %f, pose_y_on_image = %f\n", p, i, pose_x_on_image, pose_y_on_image);
        // }
        if(value > threshold) {
          if((x - pose_x_on_image) * (x - pose_x_on_image) + (y - pose_y_on_image) * (y - pose_y_on_image) <= radius * radius){
            b = g = r = 0;
          }
        }

      }
    }

    dst_pointer[                          y * w_canvas + x] = b; //plot dot
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}

inline __device__ void getColor(float* c, float v, float vmin, float vmax)
{
   c[0] = c[1] = c[2] = 255; // b, g, r, white
   float dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.125 * dv)) {
      c[0] = 256 * (0.5 + (v * 4)); //B: 0.5 ~ 1
      c[1] = c[2] = 0;
   } else if (v < (vmin + 0.375 * dv)) {
      c[0] = 255;
      c[1] = 256 * (v - 0.125) * 4; //G: 0 ~ 1
      c[2] = 0;
   } else if (v < (vmin + 0.625 * dv)) {
      c[0] = 256 * (-4 * v + 2.5);  //B: 1 ~ 0
      c[1] = 255;
      c[2] = 256 * (4 * (v - 0.375)); //R: 0 ~ 1
   } else if (v < (vmin + 0.875 * dv)) {
      c[0] = 0;
      c[1] = 256 * (-4 * v + 3.5);  //G: 1 ~ 0
      c[2] = 255;
   } else {
      c[0] = 0;
      c[1] = 0;
      c[2] = 256 * (-4 * v + 4.5); //R: 1 ~ 0.5
   }
}

inline __device__ void cubic_interpolation(float &out, float &v0, float &v1, float &v2, float &v3, float dx) {
    // Dtype a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // Dtype b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // Dtype c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}

__global__ void render_pose_website_heatmap(float* dst_pointer, int w_canvas, int h_canvas, float ratio_to_origin, 
                             float* centers, float* heatmaps, int boxsize, 
                             int num_people, float threshold, int part){
  //heatmaps has length boxsize * boxsize * 15 * num_people

  int offset3 = boxsize * boxsize * 15;
  int offset2 = boxsize * boxsize;

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  //__shared__ float shared_poses[450];
  __shared__ float shared_centers[20];

  if(global_idx < (num_people + 1) * 2){
    shared_centers[global_idx] = centers[global_idx];
  }
  // if(global_idx < num_people * 15){
  //   shared_poses[3*global_idx] = poses[3*global_idx]; //x
  //   shared_poses[3*global_idx+1] = poses[3*global_idx+1]; //y
  //   shared_poses[3*global_idx+2] = poses[3*global_idx+2]; //v
  // }

  __syncthreads();

  if(x < w_canvas && y < h_canvas){
  //if(x == 0 && y == 0){
    float b, g, r;
    float value = (part == 14) ? 1 : 0;
    float r_inv = 1.0f/ratio_to_origin;
    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < num_people; p++){
      float center_x = shared_centers[2*(p+1)];
      float center_y = shared_centers[2*(p+1)+1];

      float x_on_box = r_inv * x + (0.5 * r_inv - 0.5) - (center_x - boxsize/2);
      float y_on_box = r_inv * y + (0.5 * r_inv - 0.5) - (center_y - boxsize/2);

      if(x_on_box >= 0 && x_on_box < boxsize && y_on_box >=0 && y_on_box < boxsize){
        float value_this;
        int x_nei[4];
        x_nei[1] = int(x_on_box + 1e-5);
        x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
        x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
        x_nei[2] = (x_nei[1] + 1 >= boxsize) ? (boxsize - 1) : (x_nei[1] + 1);
        x_nei[3] = (x_nei[2] + 1 >= boxsize) ? (boxsize - 1) : (x_nei[2] + 1);
        float dx = x_on_box - x_nei[1];

        int y_nei[4];
        y_nei[1] = int(y_on_box + 1e-5);
        y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
        y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
        y_nei[2] = (y_nei[1] + 1 >= boxsize) ? (boxsize - 1) : (y_nei[1] + 1);
        y_nei[3] = (y_nei[2] + 1 >= boxsize) ? (boxsize - 1) : (y_nei[2] + 1);
        float dy = y_on_box - y_nei[1];

        float temp[4];
        int offset_src = p * offset3 + part * offset2;
        for(int i = 0; i < 4; i++){
          cubic_interpolation(temp[i], heatmaps[offset_src + y_nei[i]*boxsize + x_nei[0]], 
                                       heatmaps[offset_src + y_nei[i]*boxsize + x_nei[1]], 
                                       heatmaps[offset_src + y_nei[i]*boxsize + x_nei[2]],
                                       heatmaps[offset_src + y_nei[i]*boxsize + x_nei[3]], dx);
        }
        cubic_interpolation(value_this, temp[0], temp[1], temp[2], temp[3], dy);
        if(part != 14){
          if(value_this > value) 
          value = value_this;
        } else {
          if(value_this < value) 
          value = value_this;
        }
      }
    }
    float c[3];
    getColor(c, value, 0, 1);
    b = 0.5 * b + 0.5 * c[0];
    g = 0.5 * g + 0.5 * c[1];
    r = 0.5 * r + 0.5 * c[2];

    dst_pointer[                          y * w_canvas + x] = b; //plot dot
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}

__global__ void render_pose_website_heatmap_empty(float* dst_pointer, int w_canvas, int h_canvas){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(x < w_canvas && y < h_canvas){
  //if(x == 0 && y == 0){
    float b, g, r;

    b = 0.5 * dst_pointer[                          y * w_canvas + x] + 128;
    g = 0.5 * dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = 0.5 * dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    dst_pointer[                          y * w_canvas + x] = b;
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
  }
}


// ***** normal function *****

void fill_pose_net(const float* image, int width, int height, 
                   float* dst, int boxsize, 
                   const float* peak_pointer_gpu, vector<int> num_people, int limit){
  //image            in width * height * 3 * N
  //dst              in boxsize * boxsize * 4 * (P1+P2+...+PN)
  //peak_pointer_gpu in 2 * 11 * 1 * N
  //num_people has length P, indicating P1, ..., PN

  int N = num_people.size();
  int count = 0;
  bool full = false;
  int offset_src = width * height * 3;
  int offset_dst_2 = boxsize * boxsize;
  int offset_info = 22;
  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
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


void render_in_cuda(float* canvas, float* image, int w, int h, 
                    float* heatmaps, int boxsize, 
                    float* centers, float* poses, vector<int> num_people){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in boxsize * boxsize * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN

  int N = num_people.size();
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  int offset_person = w * h * 3; // 3 because we only render one image here
  //int offset_pose_2 = boxsize * boxsize;
  int offset_info = 22;
  int offset_pose = 45;
  float threshold = 0.15;
  int offset_pose_so_far = 0;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(updiv(w, threadsPerBlock.x), updiv(h, threadsPerBlock.y));

  for(int i = 0; i < N; i++){
    int num_people_this_frame = num_people[i];
    //LOG(ERROR) << "num_people_this_frame: " << num_people_this_frame;
    
    render_pose<<<threadsPerBlock, numBlocks>>>(canvas+i*offset_person, image+i*offset_person, w, h, 
                                                centers+i*offset_info, poses+offset_pose_so_far, boxsize, 
                                                num_people_this_frame, threshold);

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    cudaDeviceSynchronize();
    offset_pose_so_far += offset_pose * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}

void render_in_cuda_website(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net, 
                    float* heatmaps, int boxsize, 
                    float* centers, float* poses, vector<int> num_people){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in boxsize * boxsize * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN

  int N = num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  //int offset_pose_2 = boxsize * boxsize;
  int offset_heatmap = boxsize * boxsize * 15;
  int offset_info = 22;
  int offset_pose = 45;
  float threshold = 0.15;
  int offset_pose_so_far = 0;
  int offset_heatmap_so_far = 0;
  float ratio_to_origin = (float)h_canvas / (float)h_net;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(updiv(w_canvas, threadsPerBlock.x), updiv(h_canvas, threadsPerBlock.y));

  for(int i = 0; i < N; i++){ //N is always 1 for website
    int num_people_this_frame = num_people[i];
    //LOG(ERROR) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;
    
    render_pose_website<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                        centers+i*offset_info, poses+offset_pose_so_far, boxsize, 
                                                        num_people_this_frame, threshold);

    for(int part = 0; part < 15; part++){
      render_pose_website_heatmap<<<threadsPerBlock, numBlocks>>>(canvas+(part+1)*offset_canvas, w_canvas, h_canvas, ratio_to_origin,
                                                        centers+i*offset_info, heatmaps+offset_heatmap_so_far, boxsize, 
                                                        num_people_this_frame, threshold, part);
    }

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    cudaDeviceSynchronize();
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}


void render_in_cuda_website_indi(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net, 
                    float* heatmaps, int boxsize, 
                    float* centers, float* poses, vector<int> num_people, int part){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in boxsize * boxsize * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN

  int N = num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  //int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  //int offset_pose_2 = boxsize * boxsize;
  int offset_heatmap = boxsize * boxsize * 15;
  int offset_info = 22;
  int offset_pose = 45;
  float threshold = 0.15;
  int offset_pose_so_far = 0;
  int offset_heatmap_so_far = 0;
  float ratio_to_origin = (float)h_canvas / (float)h_net;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(updiv(w_canvas, threadsPerBlock.x), updiv(h_canvas, threadsPerBlock.y));

  for(int i = 0; i < N; i++){ //N is always 1 for website
    int num_people_this_frame = num_people[i];
    //LOG(ERROR) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;
    
    if(num_people_this_frame != 0){
      if(part == 0){
        render_pose_website<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                            centers+i*offset_info, poses+offset_pose_so_far, boxsize, 
                                                            num_people_this_frame, threshold);
      }
      else if (part > 0) {
        render_pose_website_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                          centers+i*offset_info, heatmaps+offset_heatmap_so_far, boxsize, 
                                                          num_people_this_frame, threshold, part-1);
      }
    } else {
      if (part > 0) 
        render_pose_website_heatmap_empty<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas);
    }

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    cudaDeviceSynchronize();
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}









template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
