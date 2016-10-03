#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/render_functions.hpp"

#define numThreadsPerBlock_1d 32
#define numThreadsPerBlock 1024

#define MAX_PEOPLE 32

#define LIMB_MPI {0,1, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 11,12, 12,13}
#define LIMB_COCO {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17, 2,16, 5,17}

namespace caffe {


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



__global__ void render_pose_29parts(float* dst_pointer, int w_canvas, int h_canvas, float ratio_to_origin,
                             float* poses, int boxsize, int num_people, float threshold){
   const int NUM_PARTS = 15;

  //poses has length 3 * 15 * num_people
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int plotted = 0;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float shared_poses[NUM_PARTS*3*MAX_PEOPLE];
  if(global_idx < num_people * NUM_PARTS){
    shared_poses[3*global_idx] = poses[3*global_idx]; //x
    shared_poses[3*global_idx+1] = poses[3*global_idx+1]; //y
    shared_poses[3*global_idx+2] = poses[3*global_idx+2]; //v
  }

  __syncthreads();

  const int limb[] = LIMB_MPI;
  const int nlimb = sizeof(limb)/(2*sizeof(int));

  int color[27] =   {255,   0, 0,
                     255, 170, 0,
                     170, 255, 0,
                       0, 255, 0,
                       0, 255, 170,
                       0, 170, 255,
                       0, 0,   255,
                     170, 0,   255,
                     255, 0,   170};
  //float offset = ratio_to_origin * 0.5 - 0.5;
  float radius = 3*h_canvas / 200.0f;
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
      for(int l = 0; l < nlimb; l++){
        float b_sqrt = stickwidth * stickwidth; //fixed
        float alpha = 0.6;
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        float x_a = (shared_poses[p*NUM_PARTS*3 + part_a*3]); // * ratio_to_origin + offset;
        float x_b = (shared_poses[p*NUM_PARTS*3 + part_b*3]); // * ratio_to_origin + offset;
        float y_a = (shared_poses[p*NUM_PARTS*3 + part_a*3 + 1]); // * ratio_to_origin + offset;
        float y_b = (shared_poses[p*NUM_PARTS*3 + part_b*3 + 1]); // * ratio_to_origin + offset;
        float value_a = shared_poses[p*NUM_PARTS*3 + part_a*3 + 2];
        float value_b = shared_poses[p*NUM_PARTS*3 + part_b*3 + 2];
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          float angle = atan2f(y_b - y_a, x_b - x_a);
          float sine = sinf(angle);
          float cosine = cosf(angle);
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);

          if (l==0) {
            a_sqrt *=1.2;
            b_sqrt = a_sqrt;
            // alpha *= 0.5;
          }


          float A = cosine * (x - x_p) + sine * (y - y_p);
          float B = sine * (x - x_p) - cosine * (y - y_p);
          float judge = A * A / a_sqrt + B * B / b_sqrt;
          float minV = 0;
          if (l==0) {
            minV = 0.8;
          }
          if(judge>= minV && judge <= 1){
            b = (1-alpha) * b + alpha * color[l*3+2];
            g = (1-alpha) * g + alpha * color[l*3+1];
            r = (1-alpha) * r + alpha * color[l*3];
            //plotted = 1;
          }
        }
      }

      for(int i = 0; i < NUM_PARTS; i++) { //for every point
        float local_x = shared_poses[p*NUM_PARTS*3 + i*3];
        float local_y = shared_poses[p*NUM_PARTS*3 + i*3 + 1];
        float value = shared_poses[p*NUM_PARTS*3 + i*3 + 2];
        float pose_x_on_image = local_x; // * ratio_to_origin + offset;
        float pose_y_on_image = local_y; // * ratio_to_origin + offset;

        if(value > threshold) {
          if((x - pose_x_on_image) * (x - pose_x_on_image) + (y - pose_y_on_image) * (y - pose_y_on_image) <= radius * radius){

            b = 0.6 * b + 0.4 * color[(i%9)*3+2];
            g = 0.6 * g + 0.4 * color[(i%9)*3+1];
            r = 0.6 * r + 0.4 * color[(i%9)*3];
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

__global__ void render_pose_29parts_heatmap(float* dst_pointer, int w_canvas, int h_canvas, int w_net,
                                            int h_net, float* heatmaps, int num_people, int part){

  const int NUM_PARTS = 15;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __syncthreads();

  if(x < w_canvas && y < h_canvas){
    //heatmaps has length w_net * h_net * 15
    int offset3 = w_net * h_net * NUM_PARTS;
    int offset2 = w_net * h_net;

    float b, g, r;
    float value = (part == NUM_PARTS-1) ? 1 : 0;
    float h_inv = (float)h_net / (float)h_canvas;
    float w_inv = (float)w_net / (float)w_canvas;
    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < 1; p++){

      float x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
      float y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

      if(x_on_box >= 0 && x_on_box < w_net && y_on_box >=0 && y_on_box < h_net){
        float value_this;
        int x_nei[4];
        x_nei[1] = int(x_on_box + 1e-5);
        x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
        x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
        x_nei[2] = (x_nei[1] + 1 >= w_net) ? (w_net - 1) : (x_nei[1] + 1);
        x_nei[3] = (x_nei[2] + 1 >= w_net) ? (w_net - 1) : (x_nei[2] + 1);
        float dx = x_on_box - x_nei[1];

        int y_nei[4];
        y_nei[1] = int(y_on_box + 1e-5);
        y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
        y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
        y_nei[2] = (y_nei[1] + 1 >= h_net) ? (h_net - 1) : (y_nei[1] + 1);
        y_nei[3] = (y_nei[2] + 1 >= h_net) ? (h_net - 1) : (y_nei[2] + 1);
        float dy = y_on_box - y_nei[1];

        float temp[4];
        int offset_src = p * offset3 + part * offset2;
        for(int i = 0; i < 4; i++){
          cubic_interpolation(temp[i], heatmaps[offset_src + y_nei[i]*w_net + x_nei[0]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[1]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[2]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[3]], dx);
        }
        cubic_interpolation(value_this, temp[0], temp[1], temp[2], temp[3], dy);
        // if(part != 14){
        //   if(value_this > value)
        //     value = value_this;
        // } else {
        //   if(value_this < value)
            value = value_this;
        // }
      }
    }
    float c[3];
    if (part<16){
      getColor(c, value, 0, 1);
    } else {
      getColor(c, value, -1, 1);
    }
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

void render_mpi_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses, vector<int> num_people, int part){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in w_net * h_net * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN
  const int NUM_PARTS = 15;
  int N = num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  //int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  int offset_heatmap = w_net * h_net * NUM_PARTS; // boxsize * boxsize * 15
  //int offset_info = 33; //22
  int offset_pose = NUM_PARTS*3;
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
        // render_pose_website<<<threadsPerBlock, numBlocks>>>
        VLOG(4) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;
        render_pose_29parts<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                            poses+offset_pose_so_far, boxsize,
                                                            num_people_this_frame, threshold);
      }
      else if (part > 0) {
        //render_pose_website_heatmap<<<threadsPerBlock, numBlocks>>>
        //LOG(ERROR) << "GPU part num: " << part-1;
        render_pose_29parts_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                          heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                          part-1);
      }
    } else {
      if (part > 0) {
        render_pose_29parts_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                          heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                          part-1);
        // render_pose_website_heatmap_empty<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas);
      }
    }

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    cudaDeviceSynchronize();
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}

//////////////////////////////////////////////////////////////////////////////
// COCO

__global__ void render_pose_coco_parts(float* dst_pointer, int w_canvas, int h_canvas, float ratio_to_origin,
                             float* poses, int boxsize, int num_people, float threshold){
   const int NUM_PARTS = 18;

  //poses has length 3 * 15 * num_people
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int plotted = 0;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float shared_poses[NUM_PARTS*3*MAX_PEOPLE];
  if(global_idx < num_people * NUM_PARTS){
    shared_poses[3*global_idx] = poses[3*global_idx]; //x
    shared_poses[3*global_idx+1] = poses[3*global_idx+1]; //y
    shared_poses[3*global_idx+2] = poses[3*global_idx+2]; //v
  }

  __syncthreads();

  const int limb[] = LIMB_COCO;
  const int nlimb = sizeof(limb)/(2*sizeof(int));

  const int color[27] =   {255,   0, 0,
                     255, 170, 0,
                     170, 255, 0,
                       0, 255, 0,
                       0, 255, 170,
                       0, 170, 255,
                       0, 0,   255,
                     170, 0,   255,
                     255, 0,   170};
  const int nColor = sizeof(color)/(3*sizeof(int));
  //float offset = ratio_to_origin * 0.5 - 0.5;
  float radius = 2*h_canvas / 200.0f;
  float stickwidth = h_canvas / 120.0f;

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
      for(int l = 0; l < nlimb; l++){
        float b_sqrt = stickwidth * stickwidth; //fixed
        float alpha = 0.6;
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        float x_a = (shared_poses[p*NUM_PARTS*3 + part_a*3]); // * ratio_to_origin + offset;
        float x_b = (shared_poses[p*NUM_PARTS*3 + part_b*3]); // * ratio_to_origin + offset;
        float y_a = (shared_poses[p*NUM_PARTS*3 + part_a*3 + 1]); // * ratio_to_origin + offset;
        float y_b = (shared_poses[p*NUM_PARTS*3 + part_b*3 + 1]); // * ratio_to_origin + offset;
        float value_a = shared_poses[p*NUM_PARTS*3 + part_a*3 + 2];
        float value_b = shared_poses[p*NUM_PARTS*3 + part_b*3 + 2];
        if(value_a > threshold && value_b > threshold){
          float x_p = (x_a + x_b) / 2;
          float y_p = (y_a + y_b) / 2;
          float angle = atan2f(y_b - y_a, x_b - x_a);
          float sine = sinf(angle);
          float cosine = cosf(angle);
          float a_sqrt = (x_a - x_p) * (x_a - x_p) + (y_a - y_p) * (y_a - y_p);


          float A = cosine * (x - x_p) + sine * (y - y_p);
          float B = sine * (x - x_p) - cosine * (y - y_p);
          float judge = A * A / a_sqrt + B * B / b_sqrt;
          float minV = 0;
          if(judge>= minV && judge <= 1){
            b = (1-alpha) * b + alpha * color[(p%nColor)*3+2];
            g = (1-alpha) * g + alpha * color[(p%nColor)*3+1];
            r = (1-alpha) * r + alpha * color[(p%nColor)*3];
            //plotted = 1;
          }
        }
      }

      for(int i = 0; i < NUM_PARTS; i++) { //for every point
        float local_x = shared_poses[p*NUM_PARTS*3 + i*3];
        float local_y = shared_poses[p*NUM_PARTS*3 + i*3 + 1];
        float value = shared_poses[p*NUM_PARTS*3 + i*3 + 2];
        float pose_x_on_image = local_x; // * ratio_to_origin + offset;
        float pose_y_on_image = local_y; // * ratio_to_origin + offset;

        if(value > threshold) {
          if((x - pose_x_on_image) * (x - pose_x_on_image) + (y - pose_y_on_image) * (y - pose_y_on_image) <= radius * radius){

            b = 0.6 * b + 0.4 * color[(i%9)*3+2];
            g = 0.6 * g + 0.4 * color[(i%9)*3+1];
            r = 0.6 * r + 0.4 * color[(i%9)*3];
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

__global__ void render_pose_coco_heatmap(float* dst_pointer, int w_canvas, int h_canvas, int w_net,
                                            int h_net, float* heatmaps, int num_people, int part){

  const int NUM_PARTS = 18;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __syncthreads();

  if(x < w_canvas && y < h_canvas){
    //heatmaps has length w_net * h_net * 15
    int offset3 = w_net * h_net * NUM_PARTS;
    int offset2 = w_net * h_net;

    float b, g, r;
    float value = (part == NUM_PARTS-1) ? 1 : 0;
    float h_inv = (float)h_net / (float)h_canvas;
    float w_inv = (float)w_net / (float)w_canvas;
    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < 1; p++){

      float x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
      float y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

      if(x_on_box >= 0 && x_on_box < w_net && y_on_box >=0 && y_on_box < h_net){
        float value_this;
        int x_nei[4];
        x_nei[1] = int(x_on_box + 1e-5);
        x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
        x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
        x_nei[2] = (x_nei[1] + 1 >= w_net) ? (w_net - 1) : (x_nei[1] + 1);
        x_nei[3] = (x_nei[2] + 1 >= w_net) ? (w_net - 1) : (x_nei[2] + 1);
        float dx = x_on_box - x_nei[1];

        int y_nei[4];
        y_nei[1] = int(y_on_box + 1e-5);
        y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
        y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
        y_nei[2] = (y_nei[1] + 1 >= h_net) ? (h_net - 1) : (y_nei[1] + 1);
        y_nei[3] = (y_nei[2] + 1 >= h_net) ? (h_net - 1) : (y_nei[2] + 1);
        float dy = y_on_box - y_nei[1];

        float temp[4];
        int offset_src = p * offset3 + part * offset2;
        for(int i = 0; i < 4; i++){
          cubic_interpolation(temp[i], heatmaps[offset_src + y_nei[i]*w_net + x_nei[0]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[1]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[2]],
                                       heatmaps[offset_src + y_nei[i]*w_net + x_nei[3]], dx);
        }
        cubic_interpolation(value_this, temp[0], temp[1], temp[2], temp[3], dy);
        // if(part != 14){
        //   if(value_this > value)
        //     value = value_this;
        // } else {
        //   if(value_this < value)
            value = value_this;
        // }
      }
    }
    float c[3];
    if (part<NUM_PARTS+1){
      getColor(c, value, 0, 1);
    } else {
      getColor(c, value, -1, 1);
    }
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

void render_coco_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses, vector<int> num_people, int part){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in w_net * h_net * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN
  const int NUM_PARTS = 18;
  int N = num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  //int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  int offset_heatmap = w_net * h_net * NUM_PARTS; // boxsize * boxsize * 15
  //int offset_info = 33; //22
  int offset_pose = NUM_PARTS*3;
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
      if(part == 0 ){
        // render_pose_website<<<threadsPerBlock, numBlocks>>>
        VLOG(4) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;
        render_pose_coco_parts<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                            poses+offset_pose_so_far, boxsize,
                                                            num_people_this_frame, threshold);
      }
      else if (part > 0) {
        //render_pose_website_heatmap<<<threadsPerBlock, numBlocks>>>
        //LOG(ERROR) << "GPU part num: " << part-1;
        render_pose_coco_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                          heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                          part-1);
      }
    } else {
      if (part > 0) {
        render_pose_coco_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                          heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                          part-1);
        // render_pose_website_heatmap_empty<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas);
      }
    }

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    cudaDeviceSynchronize();
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}
}  // namespace caffe
