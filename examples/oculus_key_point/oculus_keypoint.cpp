#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <utility> //for pair
#include <pthread.h>
#include <unistd.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/thread/thread.hpp>

#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
using namespace std;
using namespace cv;

//#define boxsize 368
#define boxsize 184
#define SIGMA 21
//#define num_parts 24

string caffemodel;
string proto;

int DEVICE_ID;
int NUM_GPU;
int BATCH_SIZE; 
Net<float>* NET;
string IMAGE_PATH;
Mat img_opencv;

void set_nets(){
	//eye example
	//caffemodel = "model/orp/eyes/pose_iter_12000.caffemodel"; 
	//proto = "model/orp/eyes/pose_deploy.prototxt";

	//mouth example
	caffemodel = "model/orp/mouth/pose_exp80_vgg_to3_mask/dani_160316/pose_iter_12000.caffemodel"; //"/media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/model/pose_iter_70000.caffemodel";
	proto = "model/orp/mouth/pose_exp80_vgg_to3_mask/dani_160316/pose_deploy.prototxt"; ///media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/pose_deploy_copy_4sg_resize.prototxt";

	caffemodel = "model/orp/mouth/pose_exp81_vgg_to3_s4/dani_160316/pose_iter_45000.caffemodel";
	proto = "model/orp/mouth/pose_exp81_vgg_to3_s4/dani_160316/pose_deploy.prototxt";
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time, NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void warmup(int device_id){
	
	LOG(ERROR) << "Setting GPU " << device_id;
	Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
	Caffe::set_mode(Caffe::GPU); //

	LOG(ERROR) << "GPU " << device_id << ": copying to person net";
	NET = new Net<float>(proto, caffe::TEST);
	NET->CopyTrainedLayersFrom(caffemodel);
	
	vector<int> shape(4); 
	shape[0] = BATCH_SIZE;
	shape[1] = 4;
	shape[2] = boxsize;
	shape[3] = boxsize;
	NET->blobs()[0]->Reshape(shape);
	NET->Reshape();

	// for(int i = 0; i < nc[device_id].nblob_person; i++){
	// 	LOG(ERROR) << "person_net blob: " << i << " " << nc[device_id].person_net->blob_names()[i] << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(0) << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(1) << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(2) << " "
	// 	           << nc[device_id].person_net->blobs()[i]->shape(3) << " ";
	// }
	
	//dry run
	LOG(INFO) << "Dry running...";
	NET->ForwardFrom(0);
}

void process_and_pad_image(float* target, Mat oriImg, int tw, int th, bool normalize){
	int ow = oriImg.cols;
	int oh = oriImg.rows;
	int offset2_target = tw * th;
	//LOG(ERROR) << tw << " " << th << " " << ow << " " << oh;

	//parallel here
	unsigned char* pointer = (unsigned char*)(oriImg.data);

	for(int c = 0; c < 3; c++){
		for(int y = 0; y < th; y++){
			for(int x = 0; x < tw; x++){
				if(x < ow && y < oh){
					if(normalize)
						target[c * offset2_target + y * tw + x] = float(pointer[(y * ow + x) * 3 + c])/256.0f - 0.5f;
					else
						target[c * offset2_target + y * tw + x] = float(pointer[(y * ow + x) * 3 + c]);
					//cout << target[c * offset2_target + y * tw + x] << " ";
				}
				else {
					target[c * offset2_target + y * tw + x] = 0;
				}
			}
		}
	}

	//check first channel
	// Mat test(th, tw, CV_8UC1);
	// for(int y = 0; y < th; y++){
	// 	for(int x = 0; x < tw; x++){
	// 		test.data[y * tw + x] = (unsigned int)((target[y * tw + x] + 0.5) * 256);
	// 	}
	// }
	// cv::imwrite("validate.jpg", test);
}

void putGaussianMaps(float* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1) 
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

void fill_net(){
	//eyes example
	//Mat image_uchar = imread("images/eyes/dani_180316/images/0_00000000.png");
	//Point center(167, 400);

	//mouth example
	Mat image_uchar = imread("images/mouth/dani_160316/0/00000000.png");
	Point center(300, 236);
	float scale = 0.4;
	resize(image_uchar, image_uchar, Size(), scale, scale, INTER_CUBIC); //if needed
	center *= scale;

	unsigned char* pointer = (unsigned char*)(image_uchar.data);
	float* data_for_net = NET->blobs()[0]->mutable_cpu_data(); //GPU sync happens here!
	int ih = image_uchar.rows;
	int iw = image_uchar.cols;
	int offset2 = boxsize * boxsize;

	//cout << ih << "  " << iw;

	for(int c = 0; c < 3; c++){
		for(int y = 0; y < boxsize; y++){
			for(int x = 0; x < boxsize; x++){
				int nx = x + center.x - boxsize/2;
				int ny = y + center.y - boxsize/2;
				if(nx < iw && nx >= 0 && ny < ih && ny >= 0){
					data_for_net[c * offset2 + y * boxsize + x] = float(pointer[(ny * iw + nx) * 3 + c])/256.0f - 0.5f;
				}
				else {
					data_for_net[c * offset2 + y * boxsize + x] = 0;
				}
			}
		}
	}
	// Gaussian at 4th channel
	putGaussianMaps(data_for_net + 3*offset2, Point(boxsize/2, boxsize/2), 1, boxsize, boxsize, SIGMA);

	Mat test(boxsize, boxsize, CV_8UC1);
	for(int y = 0; y < boxsize; y++){
		for(int x = 0; x < boxsize; x++){
			test.data[y * boxsize + x] = (unsigned int)((data_for_net[1*offset2 + y * boxsize + x]) * 256 + 128);
		}
	}
	namedWindow("image", CV_WINDOW_AUTOSIZE);
	imshow("image", test);
	waitKey();
}

void show_output() {

	int nblob = NET->blob_names().size();
	float* output_pointer = NET->blobs()[nblob-1]->mutable_cpu_data();

	Mat test(boxsize/4, boxsize/4, CV_8UC1);
	int offset2 = boxsize/4 * boxsize/4;
	int num_parts = NET->blobs()[nblob-1]->shape()[1]-1;
	//cout << "there are " << num_parts << "!!!" << endl;

	for(int p = 0; p < num_parts+1; p++){
		for(int y = 0; y < boxsize/4; y++){
			for(int x = 0; x < boxsize/4; x++){
				float value = output_pointer[p*offset2 + y * boxsize/4 + x] * 256;
				test.data[y * boxsize/4 + x] = value < 0 ? 0 : (value > 255 ? 255 : (unsigned char)value);
			}
		}
		namedWindow("image", CV_WINDOW_AUTOSIZE);
		imshow("image", test);
		waitKey();
	}
}


int main(int argc, char** argv) {
	DEVICE_ID = atoi(argv[1]); // 4
	BATCH_SIZE = atoi(argv[2]); //1

	/* server warming up */
	set_nets();
	warmup(DEVICE_ID); // should be done just once

	::google::InitGoogleLogging("oculus_keypoint");
	
	// preprocess
	double start_time = get_wall_time();
	fill_net();
	double end_time = get_wall_time();
	cout << "fill_net elapsed: " << end_time - start_time << " sec" << endl;

	start_time = get_wall_time();
	NET->Forward();
	cudaDeviceSynchronize();
	end_time = get_wall_time();
	cout << "CPM elapsed: " << end_time - start_time << " sec" << endl;

	show_output();
}