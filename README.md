Realtime Multiperson Pose Estimation
====================================
C++ code repo for the ECCV 2016 demo, "Realtime Multiperson Pose Estimation", Zhe Cao, Shih-En Wei, Tomas Simon, Yaser Sheikh.

The [full project repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes matlab and python version, and training code.

## Quick Start
1. Required: CUDA & cuDNN installed on your machine.
2. Build `caffe` & `rtpose.bin` + download the required caffe models (script tested on Ubuntu 14.04 & 16.04, it uses all the available cores in your machine):
```
chmod u+x install_caffe_and_cpm.sh
./install_caffe_and_cpm.sh
```

## Running on a video:
```
./build/examples/rtpose/rtpose.bin --video video_file.mp4
```

## Running on your webcam:
```
./build/examples/rtpose/rtpose.bin
```

## Important options:
`--help` <--- It displays all the available options.

`--video input.mp4` <--- Input video. If omitted, will use webcam.

`--camera #` <--- Choose webcam number (default: 0).

`--image_dir path_to_images/` <--- Run on all jpg, png, or bmp images in `path_to_images/`. If omitted, will use webcam.

`--write_frames path/`  <--- Render images with this prefix: path/frame%06d.jpg

`--write_json path/`  <--- Output JSON file with joints with this prefix: path/frame%06d.json

`--no_frame_drops` <--- Don't drop frames. Important for making offline results.

`--no_display` <--- Don't open a display window. Useful if there's no X server.

`--num_gpu 4` <--- Parallelize over this number of GPUs. Default is 1.

`--num_scales 3 --scale_gap 0.15`  <--- Use 3 scales, 1, (1-0.15), (1-0.15*2). Default is one scale=1.

(HD)
`--net_resolution 656x368 --resolution 1280x720` (These are the default values.)

(VGA)
`--net_resolution 496x368 --resolution 640x480`

`--logtostderr` <--- Log messages to standard error.

## Example:
Run on a video `vid.mp4`, render image frames as `output/frame%06d.jpg` and output JSON files as `output/frame%06d.json`, using 3 scales (1.00, 0.85, and 0.70), parallelized over 2 GPUs:
```
./build/examples/rtpose/rtpose.bin --video vid.mp4 --num_gpu 2 --no_frame_drops --write_frames output/ --write_json output/ --num_scales 3 --scale_gap 0.15
```

## Output format:
Each JSON file has a `bodies` array of objects, where each object has an array `joints` containing the joint locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`, where `c` is the confidence in [0,1].

```
{
"version":0.1,
"bodies":[
{"joints":[1114.15,160.396,0.846207,...]},
{"joints":[...]},
]
}
```

where the joint order of the COCO parts is: (see src/rtpose/modelDescriptorFactory.cpp )
```
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
```

## Citation
Please cite the paper in your publications if it helps your research:

    
    
    @article{cao2016realtime,
	  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
	  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
	  journal={arXiv preprint arXiv:1611.08050},
	  year={2016}
	  }
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }

