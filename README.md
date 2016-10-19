Realtime Multiperson Pose Estimation
====================================
Repo for the ECCV 2016 demo, "Realtime Multi­person Pose Estimation", Zhe Cao, Shih-En Wei, Tomas Simon, Yaser Sheikh.

### Quick Start
1. See `model/getModels.sh` for caffe model downloads
2. Build `caffe` & `rtpose.bin`:
  ```bash
cd caffe_demo
make all
  ```
3. Run `rtpose.bin`:
  ```bash
  ./build/examples/rtpose/rtpose.bin
  ```
