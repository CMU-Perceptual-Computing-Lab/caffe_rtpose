#ifndef RENDER_FUNCTIONS_H
#define RENDER_FUNCTIONS_H

#include <vector>

#define RENDER_MAX_PEOPLE 96

void render_mpi_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize,
                    float* centers, float* poses, std::vector<int> num_people, int part);
void render_coco_parts(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize,
                    float* centers, float* poses, std::vector<int> num_people, int part, bool googly_eyes=0);
void render_coco_aff(float* canvas, int w_canvas, int h_canvas, int w_net, int h_net,
                    float* heatmaps, int boxsize, float* centers, float* poses,
                    std::vector<int> num_people, int part, int num_parts_accum);

#endif  // RENDER_FUNCTIONS_H
