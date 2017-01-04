#include "rtpose/renderFunctions.h"
#include "caffe/cpm/util/math_functions.hpp"  // caffe::updiv

#define numThreadsPerBlock_1d 32
#define numThreadsPerBlock 1024

#define LIMB_MPI {0,1, 2,3, 3,4, 5,6, 6,7, 8,9, 9,10, 11,12, 12,13}
#define LIMB_COCO {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17, 2,16, 5,17}
#define LIMB_COCO_NOEAR {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,0, 0,14, 14,16, 0,15, 15,17}
//#define LIMB_COCO_NOEAR {1,2, 1,5, 2,3, 3,4, 5,6, 6,7, 1,8, 8,9, 9,10, 1,11, 11,12, 12,13, 1,15, 14,15, 14,16, 15,17, 0,15}

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

inline __device__ void getColor2(float* c, float v, float vmin, float vmax)
{
   c[0] = c[1] = c[2] = 255; // b, g, r, white

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;

   v = 55*v;
   const int RY = 15;
   const int YG = 6;
   const int GC = 4;
   const int CB = 11;
   const int BM = 13;
   const int MR = 6;
   //const int ncols = RY+YG+GC+CB+BM+MR; // 55

   if (v < RY) {
      c[0] = 255; //B: 0.5 ~ 1
      c[1] = 255*(v/(RY));
      c[2] = 0;
   } else if (v < RY+YG) {
      c[0] = 255 - 255*((v-RY)/(YG));
      c[1] = 255;
      c[2] = 0;
   } else if (v < RY+YG+GC) {
      c[0] = 0;
      c[1] = 255;
      c[2] = 255*((v-RY-YG)/(GC));
   } else if (v < RY+YG+GC+CB) {
      c[0] = 0;
      c[1] = 255 - 255*((v-RY-YG-GC)/(CB));
      c[2] = 255;
   } else if (v < RY+YG+GC+CB+BM) {
      c[0] = 255*((v-RY-YG-GC-CB)/(BM));
      c[1] = 0;
      c[2] = 255;
   } else if (v < RY+YG+GC+CB+BM+MR) {
      c[0] = 255;
      c[1] = 0;
      c[2] = 255-255*((v-RY-YG-GC-CB-BM)/(MR));
   } else {
     c[0] = 255;
     c[1] = 0;
     c[2] = 0;
   }
}

inline __device__ void getColorXY(float* c, float x, float y) {
  float rad = sqrt( x*x + y*y );
  float a = atan2(-y,-x)/M_PI;
  float fk = (a+1)/2.0; // 0 to 1
  if (::isnan(fk)) fk = 0;
  // fk = 1-exp(-fk*2);
  if (rad>1) rad = 1;
  //if (rad>0.5) rad = 1;
  getColor2(c, fk, 0, 1);
  // c[0] = 255*(1 - rad*(1-c[0]/255));
  // c[1] = 255*(1 - rad*(1-c[1]/255));
  // c[2] = 255*(1 - rad*(1-c[2]/255));
  c[0] = 255*(rad*(c[0]/255));
  c[1] = 255*(rad*(c[1]/255));
  c[2] = 255*(rad*(c[2]/255));
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

  __shared__ float shared_poses[NUM_PARTS*3*RENDER_MAX_PEOPLE];
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
            a_sqrt *= 1.2;
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
                    float* heatmaps, int boxsize, float* centers, float* poses, std::vector<int> num_people, int part){
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
  float threshold = 0.0;
  int offset_pose_so_far = 0;
  int offset_heatmap_so_far = 0;
  float ratio_to_origin = (float)h_canvas / (float)h_net;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(caffe::updiv(w_canvas, threadsPerBlock.x), caffe::updiv(h_canvas, threadsPerBlock.y));

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
                             float* poses, int boxsize, int num_people, float threshold, bool googly_eyes){
   const int NUM_PARTS = 18;

  //poses has length 3 * 15 * num_people
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int plotted = 0;
  int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ float shared_poses[NUM_PARTS*3*RENDER_MAX_PEOPLE];
  __shared__ float2 shared_mins[RENDER_MAX_PEOPLE];
  __shared__ float2 shared_maxs[RENDER_MAX_PEOPLE];
  __shared__ float2 shared_scalef[RENDER_MAX_PEOPLE];
  if(global_idx < num_people ){
    int p = global_idx;
    shared_mins[p].x = w_canvas;
    shared_mins[p].y = h_canvas;
    shared_maxs[p].x = 0;
    shared_maxs[p].y = 0;
    for (int part=0;part<NUM_PARTS;part++) {
      float x = poses[p*NUM_PARTS*3 + part*3];
      float y = poses[p*NUM_PARTS*3 + part*3+1];
      float z = poses[p*NUM_PARTS*3 + part*3+2];
      shared_poses[p*NUM_PARTS*3 + part*3] = x;
      shared_poses[p*NUM_PARTS*3 + part*3+1] = y; //y
      shared_poses[p*NUM_PARTS*3 + part*3+2] = z; //v
      if (z>threshold) {
        if (x<shared_mins[p].x) shared_mins[p].x = x;
        if (x>shared_maxs[p].x) shared_maxs[p].x = x;
        if (y<shared_mins[p].y) shared_mins[p].y = y;
        if (y>shared_maxs[p].y) shared_maxs[p].y = y;
      }
    }
    shared_scalef[p].x = shared_maxs[p].x-shared_mins[p].x;
    shared_scalef[p].y = shared_maxs[p].y-shared_mins[p].y;
    shared_scalef[p].x = (shared_scalef[p].x+shared_scalef[p].y)/2.0;
    if (shared_scalef[p].x<200) {
      shared_scalef[p].x = shared_scalef[p].x/200;
      if (shared_scalef[p].x<0.33) shared_scalef[p].x = 0.33;
    } else {
      shared_scalef[p].x = 1.0;
    }
    shared_maxs[p].x += 50;
    shared_maxs[p].y += 50;
    shared_mins[p].x -= 50;
    shared_mins[p].y -= 50;
  }


  __syncthreads();

  const int limb[] = LIMB_COCO_NOEAR;
  const int nlimb = sizeof(limb)/(2*sizeof(int));
/*
  const int color[27] =   {255,   0, 0,
                     255, 170, 0,
                     170, 255, 0,
                       0, 255, 0,
                       0, 255, 170,
                       0, 170, 255,
                       0, 0,   255,
                     170, 0,   255,
                     255, 0,   170};
  */
 const int color[] = {
   255,     0,     0,
   255,    85,     0,
   255,   170,     0,
   255,   255,     0,
   170,   255,     0,
    85,   255,     0,
     0,   255,    0,
     0,   255,    85,
     0,   255,   170,
     0,   255,   255,
     0,   170,   255,
     0,    85,   255,
     0,     0,   255,
    85,     0,   255,
   170,     0,   255,
   255,     0,   255,
   255,     0,   170,
   255,     0,    85};
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
      if (x>shared_maxs[p].x || x<shared_mins[p].x
          || y>shared_maxs[p].y || y<shared_mins[p].y) {
            continue;
          }
      for(int l = 0; l < nlimb; l++){
        float b_sqrt = shared_scalef[p].x*shared_scalef[p].x*stickwidth * stickwidth; //fixed
        float alpha = 0.5;
        int part_a = limb[2*l];
        int part_b = limb[2*l+1];
        float x_a = (shared_poses[p*NUM_PARTS*3 + part_a*3]); // * ratio_to_origin + offset;
        float x_b = (shared_poses[p*NUM_PARTS*3 + part_b*3]); // * ratio_to_origin + offset;
        float y_a = (shared_poses[p*NUM_PARTS*3 + part_a*3 + 1]); // * ratio_to_origin + offset;
        float y_b = (shared_poses[p*NUM_PARTS*3 + part_b*3 + 1]); // * ratio_to_origin + offset;
        float value_a = shared_poses[p*NUM_PARTS*3 + part_a*3 + 2];
        float value_b = shared_poses[p*NUM_PARTS*3 + part_b*3 + 2];
        if (0 && (l==nlimb-1 || l==nlimb-5)) {
          float x_c = (shared_poses[p*NUM_PARTS*3 + 14*3 + 0]); // * ratio_to_origin + offset;
          float y_c = (shared_poses[p*NUM_PARTS*3 + 14*3 + 1]); // * ratio_to_origin + offset;
          float value_c = shared_poses[p*NUM_PARTS*3 + 14*3 + 2];
          if (value_c>threshold) {
            x_b = (x_c+x_b)/2;
            y_b = (y_c+y_b)/2;
          } else {
            continue;
          }
        }

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
          float maxV = 1;
          float3 co;
          co.x = color[(l%nColor)*3+0];
          co.y = color[(l%nColor)*3+1];
          co.z = color[(l%nColor)*3+2];

          if ( 0 && (l==nlimb-4 || l==nlimb-1 || l==nlimb-2 || l==nlimb-3 || l==nlimb-5)) {
            // float nx = cosine;
            // float ny = sine;
            // float px = nx*(x-x_a) + ny*(y-y_a);
            float lw = 8;
            if (l==nlimb-1) {
              lw = 2;
            }
            if (B>-lw && B<lw) {
              judge = A/sqrt(a_sqrt);
            } else {
              judge = 2;
            }

            minV = -1;
            maxV = 1;

            alpha = 0.9;
            co.x = 0; co.y = 0; co.z = 0;

            if (l==nlimb-5) {
              maxV = -0.3;
              alpha = 0.3*(1-(judge+1)/0.8);
              co.x = 255; co.y = 255; co.z = 255;
            }
          }

          if(judge>= minV && judge <= maxV){
            b = (1-alpha) * b + alpha * co.z;
            g = (1-alpha) * g + alpha * co.y;
            r = (1-alpha) * r + alpha * co.x;
            //plotted = 1;
          }
        }
      }

      for(int i = 0; i < NUM_PARTS; i++) { //for every point
        float local_x = shared_poses[p*NUM_PARTS*3 + i*3];
        float local_y = shared_poses[p*NUM_PARTS*3 + i*3 + 1];
        float value = shared_poses[p*NUM_PARTS*3 + i*3 + 2];

        if(value > threshold) {
          float dist2 = (x - local_x) * (x - local_x) + (y - local_y) * (y - local_y);
          float minr2 = 0;
          float maxr2 = shared_scalef[p].x*shared_scalef[p].x*radius * radius;
          float alpha = 0.6;
          float3 co;
          co.x = color[(i%nColor)*3+0];
          co.y = color[(i%nColor)*3+1];
          co.z = color[(i%nColor)*3+2];

          if (googly_eyes && (i==14 || i==15)) {
            maxr2 = shared_scalef[p].x*shared_scalef[p].x*2.5*2.5*radius*radius;
            minr2 = shared_scalef[p].x*shared_scalef[p].x*(2.5*radius-2)*(2.5*radius-2);
            alpha = 0.9;
            co.x = 0; co.y = 0; co.z = 0;
            if(dist2 <= maxr2){
              if(dist2 <= minr2) {
                co.x = 255; co.y = 255; co.z = 255;
              }
              if(dist2 <= minr2*0.6) {
                float dist3 = (x-4 - local_x) * (x-4 - local_x) + (y - local_y+4) * (y - local_y+4);
                if (dist3>3.75*3.75) {
                  co.x = 0; co.y = 0; co.z = 0;
                }
              }
              b = (1-alpha) * b + alpha * co.z;
              g = (1-alpha) * g + alpha * co.y;
              r = (1-alpha) * r + alpha * co.x;
            }
          } else {
            if (0 && i==0) {
              alpha = 0.9;
              maxr2 = maxr2*2;
              co.x = 0; co.y = 0; co.x = 255;
            }
          if(dist2>=minr2 && dist2 <= maxr2){
            b = (1-alpha) * b + alpha * co.z;
            g = (1-alpha) * g + alpha * co.y;
            r = (1-alpha) * r + alpha * co.x;
          }
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
      getColor(c, value, -1,1);
    }
    float alpha = 0.7;
    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];

    dst_pointer[                          y * w_canvas + x] = b; //plot dot
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}

__global__ void render_pose_coco_heatmap2(float* dst_pointer, int w_canvas, int h_canvas, int w_net,
                                            int h_net, float* heatmaps, int num_people, int in_part){

  const int NUM_PARTS = 18;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int global_idx = threadIdx.y * blockDim.x + threadIdx.x;

  const int color[] = {
    255,     0,     0,
    255,    85,     0,
    255,   170,     0,
    255,   255,     0,
    170,   255,     0,
     85,   255,     0,
      0,   255,    0,
      0,   255,    85,
      0,   255,   170,
      0,   255,   255,
      0,   170,   255,
      0,    85,   255,
      0,     0,   255,
     85,     0,   255,
    170,     0,   255,
    255,     0,   255,
    255,     0,   170,
    255,     0,    85};
   const int nColor = sizeof(color)/(3*sizeof(int));

  __syncthreads();

  if(x < w_canvas && y < h_canvas){
    //heatmaps has length w_net * h_net * 15
    int offset3 = w_net * h_net * NUM_PARTS;
    int offset2 = w_net * h_net;

    float b, g, r;
    float c[3];
    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    float value = 0;
    float h_inv = (float)h_net / (float)h_canvas;
    float w_inv = (float)w_net / (float)w_canvas;
    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < 1; p++){
      for (int part=in_part;part<NUM_PARTS;part++) {

      float x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
      float y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

      if(x_on_box >= 0 && x_on_box < w_net && y_on_box >=0 && y_on_box < h_net){
        //float value_this;
        int x_nei[4];
        x_nei[1] = int(x_on_box + 1e-5);
        x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
        x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
        x_nei[2] = (x_nei[1] + 1 >= w_net) ? (w_net - 1) : (x_nei[1] + 1);
        x_nei[3] = (x_nei[2] + 1 >= w_net) ? (w_net - 1) : (x_nei[2] + 1);
        //float dx = x_on_box - x_nei[1];

        int y_nei[4];
        y_nei[1] = int(y_on_box + 1e-5);
        y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
        y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
        y_nei[2] = (y_nei[1] + 1 >= h_net) ? (h_net - 1) : (y_nei[1] + 1);
        y_nei[3] = (y_nei[2] + 1 >= h_net) ? (h_net - 1) : (y_nei[2] + 1);
        //float dy = y_on_box - y_nei[1];

        //float temp[4];
        int offset_src = p * offset3 + part * offset2;
        value = heatmaps[offset_src + y_nei[1]*w_net + x_nei[1]];
        // if(part != 14){
        //   if(value_this > value)
        //     value = value_this;
        // } else {
        //   if(value_this < value)
            __saturatef(value);
            c[0] += value*color[(part%nColor)*3+0];
            c[1] += value*color[(part%nColor)*3+1];
            c[2] += value*color[(part%nColor)*3+2];

        // }
      }
    }
    }
    // if (part<NUM_PARTS+1){
    //   getColor(c, value, 0, 1);
    // } else {
    //   getColor(c, value, -1,1);
    // }
    float alpha = 0.7;
    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];

    dst_pointer[                          y * w_canvas + x] = b; //plot dot
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}

__global__ void render_pose_coco_affinity(float* dst_pointer, int w_canvas, int h_canvas, int w_net,
                                            int h_net, float* heatmaps, int num_parts_accum, int num_people, int in_part){

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
    float c[3];
    int count = 0;
    c[0] = 0; c[1] = 0; c[2] = 0;
    float value = 0;
    float value2 = 0;
    float h_inv = (float)h_net / (float)h_canvas;
    float w_inv = (float)w_net / (float)w_canvas;
    // b = 255 * 0.7 + 0.3 * (image_ref[y*w + x] + 0.5) * 256;
    // g = 255 * 0.7 + 0.3 * (image_ref[w*h + y*w + x] + 0.5) * 256;
    // r = 255 * 0.7 + 0.3 * (image_ref[2*w*h + y*w + x] + 0.5) * 256;

    b = dst_pointer[                          y * w_canvas + x];
    g = dst_pointer[    w_canvas * h_canvas + y * w_canvas + x];
    r = dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x];

    for(int p = 0; p < 1; p++){
      for (int part=in_part;part<in_part+num_parts_accum*2;part+=2) {

        float x_on_box = w_inv * x + (0.5 * w_inv - 0.5);
        float y_on_box = h_inv * y + (0.5 * h_inv - 0.5);

        if(x_on_box >= 0 && x_on_box < w_net && y_on_box >=0 && y_on_box < h_net){
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

          //float temp[4];
          int offset_src = p * offset3 + part * offset2;
          if (num_parts_accum==1) {
          // for(int i = 0; i < 4; i++){
          //   cubic_interpolation(temp[i], heatmaps[offset_src + y_nei[i]*w_net + x_nei[0]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[1]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[2]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[3]], dx);
          // }
          //
          // cubic_interpolation(value, temp[0], temp[1], temp[2], temp[3], dy);
          {
            float a = heatmaps[offset_src + y_nei[1]*w_net + x_nei[1]];
            float b = heatmaps[offset_src + y_nei[1]*w_net + x_nei[2]];
            float c = heatmaps[offset_src + y_nei[2]*w_net + x_nei[1]];
            float d = heatmaps[offset_src + y_nei[2]*w_net + x_nei[2]];
            value = (1-dx)*(1-dy)*a
                  + (dx)*(1-dy)*b
                  + (1-dx)*(dy)*c
                  + (dx)*(dy)*d;
          }
          offset_src = p * offset3 + (part+1) * offset2;
          {
            float a = heatmaps[offset_src + y_nei[1]*w_net + x_nei[1]];
            float b = heatmaps[offset_src + y_nei[1]*w_net + x_nei[2]];
            float c = heatmaps[offset_src + y_nei[2]*w_net + x_nei[1]];
            float d = heatmaps[offset_src + y_nei[2]*w_net + x_nei[2]];
            value2 = (1-dx)*(1-dy)*a
                  + (dx)*(1-dy)*b
                  + (1-dx)*(dy)*c
                  + (dx)*(dy)*d;
          }
          // for(int i = 0; i < 4; i++){
          //   cubic_interpolation(temp[i], heatmaps[offset_src + y_nei[i]*w_net + x_nei[0]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[1]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[2]],
          //     heatmaps[offset_src + y_nei[i]*w_net + x_nei[3]], dx);
          // }
          // cubic_interpolation(value2, temp[0], temp[1], temp[2], temp[3], dy);
        } else {
           value = heatmaps[offset_src + y_nei[1]*w_net + x_nei[1]];
           offset_src = p * offset3 + (part+1) * offset2;
           value2 = heatmaps[offset_src + y_nei[1]*w_net + x_nei[1]];
        }

          //
              // if(part != 14){
              //   if(value_this > value)
              //     value = value_this;
              // } else {
              //   if(value_this < value)
              float c2[3];
              // if (part%2==0) {
              //   value = (x-320)/sqrtf( (180)*(180) + (180)*(180));
              //   value2 = (y-180)/sqrtf( (180)*(180) + (180)*(180));
              // }
              getColorXY(c2, value, value2);
              c[0] += c2[0];
              c[1] += c2[1];
              c[2] += c2[2];
              count++;
              // }
            }
          }
        }
        if (c[0]>255) c[0] = 255;
        if (c[1]>255) c[1] = 255;
        if (c[2]>255) c[2] = 255;

    // c[0] /= count;
    // c[1] /= count;
    // c[2] /= count;
    float alpha = 0.7;
    b = (1-alpha) * b + alpha * c[2];
    g = (1-alpha) * g + alpha * c[1];
    r = (1-alpha) * r + alpha * c[0];
    dst_pointer[                          y * w_canvas + x] = b; //plot dot
    dst_pointer[    w_canvas * h_canvas + y * w_canvas + x] = g;
    dst_pointer[2 * w_canvas * h_canvas + y * w_canvas + x] = r;
    // if(x==0 && y==0){
    //   printf("exiting\n");
    // }
  }
}


void render_coco_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses, std::vector<int> num_people, int part, bool googly_eyes){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in w_net * h_net * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN
  const int NUM_PARTS = 18;
  int N = 1;//num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  //int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  int offset_heatmap = w_net * h_net * NUM_PARTS; // boxsize * boxsize * 15
  //int offset_info = 33; //22
  int offset_pose = NUM_PARTS*3;
  float threshold = 0.01;
  int offset_pose_so_far = 0;
  int offset_heatmap_so_far = 0;
  float ratio_to_origin = (float)h_canvas / (float)h_net;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(caffe::updiv(w_canvas, threadsPerBlock.x), caffe::updiv(h_canvas, threadsPerBlock.y));

  for(int i = 0; i < N; i++){ //N is always 1 for website
    int num_people_this_frame = num_people[i];
    //LOG(ERROR) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;

    if(part == 0 ){
      // render_pose_website<<<threadsPerBlock, numBlocks>>>
      VLOG(4) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;
      if(num_people_this_frame != 0){
        render_pose_coco_parts<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, ratio_to_origin,
                                                          poses+offset_pose_so_far, boxsize,
                                                          num_people_this_frame, threshold, googly_eyes);
      }
    } else if (part > 0 && part<58) {

      //render_pose_website_heatmap<<<threadsPerBlock, numBlocks>>>
      //LOG(ERROR) << "GPU part num: " << part-1;
      if (part-1==NUM_PARTS) {

        render_pose_coco_heatmap2<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                        heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                        0);
      } else {
        render_pose_coco_heatmap<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                        heatmaps+offset_heatmap_so_far, num_people_this_frame,
                                                        part-1);
      }
    }

    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    CUDA_CHECK(cudaDeviceSynchronize());
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}

void render_coco_aff(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses,
                    std::vector<int> num_people, int part, int num_parts_accum){
  //canvas, image    in width * height * 3 * N
  //heatmaps         in w_net * h_net * 15 * (P1+P2+...+PN)
  //centers          in 2 * 11 * 1 * N
  //poses            in 3 * 1 * 15 * (P1+P2+...+PN)
  //num_people has length P, indicating P1, ..., PN
  const int NUM_PARTS = 18;
  int N = 1;//num_people.size(); //batch size
  //LOG(ERROR) << "Number of frames in batch: " << N;
  //int count = 0;
  //int offset_canvas = w_canvas * h_canvas * 3; // 3 because we only render one image here
  int offset_heatmap = w_net * h_net * NUM_PARTS; // boxsize * boxsize * 15
  //int offset_info = 33; //22
  int offset_pose = NUM_PARTS*3;
  //float threshold = 0.01;
  int offset_pose_so_far = 0;
  int offset_heatmap_so_far = 0;
  //float ratio_to_origin = (float)h_canvas / (float)h_net;

  dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
  dim3 numBlocks(caffe::updiv(w_canvas, threadsPerBlock.x), caffe::updiv(h_canvas, threadsPerBlock.y));

  for(int i = 0; i < N; i++){ //N is always 1 for website
    int num_people_this_frame = num_people[i];
    //LOG(ERROR) << "num_people_this_frame: " << num_people_this_frame << " ratio_to_origin: " << ratio_to_origin;

    int aff_part = part;
    render_pose_coco_affinity<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas, w_net, h_net,
                                                      heatmaps+offset_heatmap_so_far, num_parts_accum, num_people_this_frame,
                                                      aff_part);
        // render_pose_website_heatmap_empty<<<threadsPerBlock, numBlocks>>>(canvas, w_canvas, h_canvas);


    //LOG(ERROR) << "num_people[i] = " << num_people[i];
    CUDA_CHECK(cudaDeviceSynchronize());
    offset_pose_so_far += offset_pose * num_people[i];
    offset_heatmap_so_far += offset_heatmap * num_people[i];
  }
  //
  //LOG(ERROR) << "render_done";
}
