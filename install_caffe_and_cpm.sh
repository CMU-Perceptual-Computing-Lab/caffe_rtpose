#!/bin/bash



echo "------------------------- INSTALLING CAFFE AND CPM -------------------------"
echo "NOTE: This script assumes that CUDA and cuDNN are already installed on your machine. Otherwise, it might fail."



function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}



echo "------------------------- Checking Ubuntu Version -------------------------"
ubuntu_version="$(lsb_release -r)"
echo "Ubuntu $ubuntu_version"
if [[ $ubuntu_version == *"14."* ]]; then
    ubuntu_le_14=true
elif [[ $ubuntu_version == *"16."* || $ubuntu_version == *"15."* || $ubuntu_version == *"17."* || $ubuntu_version == *"18."* ]]; then
    ubuntu_le_14=false
else
    echo "Ubuntu release older than version 14. This installation script might fail."
    ubuntu_le_14=true
fi
exitIfError
echo "------------------------- Ubuntu Version Checked -------------------------"
echo ""



echo "------------------------- Checking Number of Processors -------------------------"
NUM_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
echo "$NUM_CORES cores"
exitIfError
echo "------------------------- Number of Processors Checked -------------------------"
echo ""



echo "------------------------- Installing some Caffe Dependencies -------------------------"
# Basic
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install build-essential
#General dependencies
sudo apt-get --assume-yes install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
if [[ $ubuntu_le_14 == true ]]; then
    sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
fi
# OpenCV 2.4
# sudo apt-get --assume-yes install libopencv-dev
exitIfError
echo "------------------------- Some Caffe Dependencies Installed -------------------------"
echo ""



echo "------------------------- Compiling Caffe & CPM -------------------------"
if [[ $ubuntu_le_14 == true ]]; then
    cp Makefile.config.Ubuntu14.example Makefile.config
else
    cp Makefile.config.Ubuntu16.example Makefile.config
fi
make all -j$NUM_CORES
# make test -j$NUM_CORES
# make runtest -j$NUM_CORES
exitIfError
echo "------------------------- Caffe & CPM Compiled -------------------------"
echo ""



# echo "------------------------- Installing CPM -------------------------"
# echo "Compiled"
# exitIfError
# echo "------------------------- CPM Installed -------------------------"
# echo ""



echo "------------------------- Downloading CPM Models -------------------------"
models_folder="./model/"
# COCO
coco_folder="$models_folder"coco/""
coco_model="$coco_folder"pose_iter_440000.caffemodel""
if [ ! -f $coco_model ]; then
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/coco/pose_iter_440000.caffemodel -P $coco_folder
fi
exitIfError
# MPI
mpi_folder="$models_folder"mpi/""
mpi_model="$mpi_folder"pose_iter_160000.caffemodel""
if [ ! -f $mpi_model ]; then
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel -P $mpi_folder
fi
exitIfError
echo "Models downloaded"
echo "------------------------- CPM Models Downloaded -------------------------"
echo ""



echo "------------------------- CAFFE AND CPM INSTALLED -------------------------"
echo ""
