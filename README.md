Realtime Multiperson Pose Estimation
====================================
C++ code repo for the ECCV 2016 demo, "Realtime Multiperson Pose Estimation", Zhe Cao, Shih-En Wei, Tomas Simon, Yaser Sheikh. Thanks Ginés Hidalgo Martínez for restructuring the code. 

The [full project repo](https://github.com/ZheC/Multi-Person-Pose-Estimation) includes matlab and python version, and training code.

This project is under the terms of the [license](LICENSE).

## Quick Start
1. Required: CUDA & cuDNN installed on your machine.
2. If you have installed OpenCV 2.4 in your system, go to step 3. If you are using OpenCV 3, uncomment the line `# OPENCV_VERSION := 3` on the file `Makefile.config.Ubuntu14.example` (for Ubuntu 14) and/or `Makefile.config.Ubuntu16.example` (for Ubuntu 15 or 16). In addition, OpenCV 3 does not incorporate the `opencv_contrib` module by default. Assuming you have manually installed it and you need to use it, append `opencv_contrib` at the end of the line `LIBRARIES += opencv_core opencv_highgui opencv_imgproc` in the `Makefile` file.
3. Build `caffe` & `rtpose.bin` + download the required caffe models (script tested on Ubuntu 14.04 & 16.04, it uses all the available cores in your machine):**
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

## Custom Caffe:
We modified and added several Caffe files in `include/caffe` and `src/caffe`. In case you want to use your own Caffe distribution, these are the files we added and modified:

1. Added folders in `include/caffe` and `src/caffe`: `include/caffe/cpm` and `src/caffe/cpm`.
2. Modified files in `include/caffe` (search for `// CPM extra code:` to find the modified code): `data_transformer.hpp`.
3. Modified files in `src/caffe` (search for `// CPM extra code:` to find the modified code): `data_transformer.cpp`, `proto/caffe.proto` and `util/blocking_queue.cpp`.
4. Replaced files: `README.md`.
5. Added files: `install_caffe_and_cpm.sh`, `Makefile.config.Ubuntu14.example` (extracted from `Makefile.config.example`) and `Makefile.config.Ubuntu16.example` (extracted from `Makefile.config.example`).
6. Other added folders: `model/`, `examples/rtpose`, `/include/rtpose` and `/src/rtpose`.
7. Other modified files: `Makefile`.
8. Optional - deleted Caffe files and folders (only to save space): `Makefile.config.example`, `data/`, `examples/` (do not delete `examples/rtpose`) and `models/`.


## Custom Caffe layers:
We created a few Caffe layers (located in `include/caffe/cpm/layers` and `src/caffe/cpm/layers`):

1. ImResizeLayer: Only used for testing (backward pass not implemented). This layer performs 2-D resize over the 4-D data. I.e., given a 4-D input of size (`num` x `channels` x `height_input` x `width_input`), the layer returns a 4-D output of size (`num` x `channels` x `height_output` x `width_output`). It is independently applied to each dimension of `num` and `channels`. Its parameters are:
	1. `factor`: Scaling factor with respect to the input width and height. `factor` is the alternative to the pair of variables [`target_spatial_width`, `target_spatial_height`]. If `factor != 0`, the latter are ignored.
	2. `scale_gap` and `start_scale`: These parameters are related and used for doing scale search in testing mode. If `start_scale = 1` (default), the CNN input patch size is the net resolution (set with `--net_resolution`). `scale_gap` is used to calculate the scale difference between scales. This parameters are related with the flag `--num_scales`. For instance, using `--start_scale 1 --num_scales 3 --scale_gap 0.1` means using 3 scales: 1, 1-0.1, 1-2*0.1, hence the different patch sizes correspond to the net resolution multiplied by these scales values.
	3. `target_spatial_height`: Alternative to `factor`. It sets the output height. Ignored if `factor != 0`.
	4. `target_spatial_width`: Alternative to `factor`. It sets the output width. Ignored if `factor != 0`.
2. NmsLayer: Only used for testing (backward pass not implemented). This layer performs 3-D Non-Maximum Suppression over the 4-D data. I.e., given a 4-D input of size (`num` x `channels` x `height` x `width`), it returns a 4-D output of size (`num` x `num_parts` x `max_peaks+1` x `3`). It is independently applied to each dimension of `num`. The seconds dimension corresponds to the number of limbs (`num_parts`). The third dimension indicates the maximum number of peaks to be analyzed (`max_peaks+1`). Finally, the last one corresponds to the `x`, `y` and `score` values (`3`). Its parameters are:
	1. `max_peaks`: The number of peaks to be considered. The last `total_peaks` - `max_peaks` peaks are discarded.
	2. `num_parts`: The number of limbs to detect (e.g. 15 for MPI and 18 for COCO).
	3. `threshold`: Any input value smaller than this threshold is set to 0.


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
