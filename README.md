Realtime Multiperson Pose Estimation
====================================
Repo for the ECCV 2016 demo, "Realtime Multi­person Pose Estimation", Zhe Cao, Shih-En Wei, Tomas Simon, Yaser Sheikh.

## Quick Start
1. See `model/getModels.sh` for caffe model downloads
2. Build `caffe` & `rtpose.bin`:
  ```bash
cd caffe_demo
make all
  ```

## Running on an webcam:
Run `rtpose.bin`:
  ```bash
  ./build/examples/rtpose/rtpose.bin
  ```

## Running on a video:
./build/examples/rtpose/rtpose.bin --video ${vid} --num_gpu 4 --logtostderr --no_frame_drops --write_frames ${opath}/images/${fname}/frame --net_resolution 496x368 --resolution 640x480 --num_scales 3 --scale_gap 0.15 --write_json ${opath}/json/${fname}/frame

## Important options:
--video input.mp4 <--- input video. If omitted, will use webcam (can be specified using --camera # ).

--write_frames path/frame  <--- render images with this prefix: path/frame%06d.jpg

--write_json path/frame  <--- output json file with joints with this prefix: path/frame%06d.json

--no_frame_drops <--- Don't drop frames. Important for making offline results.

--num_gpu 4 <--- Parallelize over this number of GPUs. Default is 1.

--num_scales 3 --scale_gap 0.15  <--- Use 3 scales, 1, (1-0.15), (1-0.15*2). Default is one scale=1.

(HD)
--net_resolution 656x368 --resolution 1280x720 (These are the default values.)

(VGA)
--net_resolution 496x368 --resolution 640x480

## Coco parts: (see examples/rtpose/modeldesc.h )

	part2name {
		{0,  "Nose"},
		{1,  "Neck"},
		{2,  "RShoulder"},
		{3,  "RElbow"},
		{4,  "RWrist"},
		{5,  "LShoulder"},
		{6,  "LElbow"},
    {7,  "LWrist"},
		{8,  "RHip"},
		{9,  "RKnee"},
		{10, "RAnkle"},
		{11, "LHip"},
		{12, "LKnee"},
		{13, "LAnkle"},
    {14, "REye"},
		{15, "LEye"},
		{16, "REar"},
		{17, "LEar"},
		{18, "Bkg"},
	}
