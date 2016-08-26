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
#include <time.h>
#include <sys/time.h>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <mutex> 

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
using namespace std;
using namespace cv;


#define INIT_TOTAL_NUM_PEOPLE 4
#define boxsize 368
#define fixed_scale_height 368
#define peak_blob_offset 22
#define INIT_PERSON_NET_HEIGHT 368
#define INIT_PERSON_NET_WIDTH 496  //656
#define origin_height 480
#define origin_width 640
#define MAX_PEOPLE_IN_BATCH 32
#define BUFFER_SIZE 4    //affects latency
#define MAX_LENGTH_INPUT_QUEUE 500 //affects memory usage
#define FPS_SRC 30

int NUM_GPU;  //4
int batch_size;  //2
double INIT_TIME = -999;

//person detector
string person_detector_caffemodel;
string person_detector_proto;

//pose estimator
string pose_estimator_caffemodel;
string pose_estimator_proto;

// network copy for each gpu thread
struct NET_COPY {
	Net<float>* person_net;
	Net<float>* pose_net;
	vector<int> num_people;
	int nblob_person;
	int nblob_pose;
	int total_num_people;
	float* canvas; // GPU memory
};


// global queues for I/O
struct GLOBAL {
	caffe::BlockingQueue<Frame> input_queue; //have to pop
	caffe::BlockingQueue<pair<int, string> > frame_file_queue; 
	caffe::BlockingQueue<Frame> output_queue; //have to pop
	caffe::BlockingQueue<Frame> output_queue_ordered;
	caffe::BlockingQueue<Frame> output_queue_mated;
	std::priority_queue<int, std::vector<int>, std::greater<int> > dropped_index;
	std::mutex mtx;
	int part_to_show;
};

NET_COPY nc[4];
GLOBAL global;

void set_nets();
int rtcpm();
void putGaussianMaps(float* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
void warmup(int);
void dostuff(int); /* function prototype */
void error(const char *msg);
void process_and_pad_image(float* target, Mat oriImg, int tw, int th, bool normalize);


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
    //return (double)time.tv_usec;
}

void printGlobal(string src){
	LOG(ERROR) << src << "\t input_queue: " << global.input_queue.size() 
	           << " | output_queue: " << global.output_queue.size()
	           << " | output_queue_ordered: " << global.output_queue_ordered.size() 
	           << " | output_queue_mated: " << global.output_queue_mated.size();
}

void set_nets(){
	person_detector_caffemodel = "model/pose_iter_70000.caffemodel"; //"/media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/model/pose_iter_70000.caffemodel";
	person_detector_proto = "model/pose_deploy_copy_4sg_resize.prototxt"; ///media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/pose_deploy_copy_4sg_resize.prototxt";

	pose_estimator_caffemodel = "model/pose_iter_320000.caffemodel"; //  "/media/posenas1/Users/shihenw/caffe_model/MPI/pose_exp78.6_vgg_to4-2/nc0/pose_iter_320000.caffemodel";
	pose_estimator_proto = "model/pose_deploy_resize.prototxt"; // "/media/posenas1/Users/shihenw/caffe_model/MPI/pose_exp78.6_vgg_to4-2/nc0/pose_deploy_resize.prototxt";
}

void warmup(int device_id){
	
	LOG(ERROR) << "Setting GPU " << device_id;
	Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
	Caffe::set_mode(Caffe::GPU); //

	LOG(ERROR) << "GPU " << device_id << ": copying to person net";
	nc[device_id].person_net = new Net<float>(person_detector_proto, caffe::TEST);
	nc[device_id].person_net->CopyTrainedLayersFrom(person_detector_caffemodel);
	nc[device_id].nblob_person = nc[device_id].person_net->blob_names().size();
	nc[device_id].num_people.resize(batch_size);

	vector<int> shape(4); 
	shape[0] = batch_size;
	shape[1] = 3;
	shape[2] = INIT_PERSON_NET_HEIGHT;
	shape[3] = INIT_PERSON_NET_WIDTH;
	nc[device_id].person_net->blobs()[0]->Reshape(shape);
	nc[device_id].person_net->Reshape();

	// for(int i = 0; i < nc[device_id].nblob_person; i++){
	// 	LOG(ERROR) << "person_net blob: " << i << " " << nc[device_id].person_net->blob_names()[i] << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(0) << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(1) << " " 
	// 	           << nc[device_id].person_net->blobs()[i]->shape(2) << " "
	// 	           << nc[device_id].person_net->blobs()[i]->shape(3) << " ";
	// }

	LOG(ERROR) << "GPU " << device_id << ": copying to pose net";
	nc[device_id].pose_net = new Net<float>(pose_estimator_proto, caffe::TEST);
	nc[device_id].pose_net->CopyTrainedLayersFrom(pose_estimator_caffemodel);
	nc[device_id].total_num_people = INIT_TOTAL_NUM_PEOPLE;
	nc[device_id].nblob_pose = nc[device_id].pose_net->blob_names().size();

	shape[0] = nc[device_id].total_num_people;
	shape[1] = 4;
	shape[2] = boxsize;
	shape[3] = boxsize;
	nc[device_id].pose_net->blobs()[0]->Reshape(shape);
	nc[device_id].pose_net->Reshape();

	// for(int i = 0; i < nc[device_id].nblob_pose; i++) {
	// 	LOG(ERROR) << "pose_net Blob: " << i << " " << nc[device_id].pose_net->blob_names()[i] << " " 
	// 	           << nc[device_id].pose_net->blobs()[i]->shape(0) << " " 
	// 	           << nc[device_id].pose_net->blobs()[i]->shape(1) << " " 
	// 	           << nc[device_id].pose_net->blobs()[i]->shape(2) << " "
	// 	           << nc[device_id].pose_net->blobs()[i]->shape(3) << " ";
	// }
	
	//dry run
	LOG(INFO) << "Dry running...";
	nc[device_id].person_net->ForwardFrom(0);
	nc[device_id].pose_net->ForwardFrom(0);

	cudaMalloc(&nc[device_id].canvas, origin_width * origin_height * 3 * sizeof(float));
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

void render(int gid) {
	// LOG(ERROR) << "begin render";
	// LOG(ERROR) << nc.person_net->blob_names()[3] << " " << 
	// 			  nc.person_net->blobs()[3]->shape(0) << " " << 
	//               nc.person_net->blobs()[3]->shape(1) << " " << 
	//               nc.person_net->blobs()[3]->shape(2) << " " <<
	//               nc.person_net->blobs()[3]->shape(3);

	//float* canvas = nc[gid].person_net->blobs()[3]->mutable_gpu_data(); //render layer
	//float* image_ref = nc[gid].person_net->blobs()[0]->mutable_gpu_data();
	float* centers  = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_gpu_data();
	float* heatmaps = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-2]->mutable_gpu_data();
	float* poses    = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-1]->mutable_gpu_data(); 
	
	//LOG(ERROR) << "begin render_in_cuda";
	caffe::render_in_cuda_website_indi(nc[gid].canvas, origin_width, origin_height, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT, 
		                  heatmaps, boxsize, 
		                  centers, poses, nc[gid].num_people, global.part_to_show);

	// float* poses_cpu = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-1]->mutable_cpu_data();
	// for(int i = 0; i < nc[gid].total_num_people * 45; i++){
	// 	cout << poses_cpu[i] << " ";
	// 	if(i % 45 == 44) cout << endl; 
	// }

	// float* pp = nc.person_net->blobs()[3]->mutable_cpu_data();
	// Mat visualize(nc.target_height, nc.target_width, CV_8UC3);
	// for(int c = 0; c < 3; c++){
	// 	for(int i = 0; i < nc.target_height; i++){
	// 		for(int j = 0; j < nc.target_width; j++){
	// 			int value = int(pp[c*nc.target_height*nc.target_width + i*nc.target_width + j] + 0.5);
	// 			value = value<0 ? 0 : (value > 255 ? 255 : value);
	// 			visualize.data[3*(i*nc.target_width + j) + c] = (unsigned char)(value);
	// 		}
	// 	}
	// }
	// imwrite("validate.jpg", visualize);
}

void copy_to_posenet_and_reshape(int gid){
	float* image = nc[gid].person_net->blobs()[0]->mutable_gpu_data();
	//float* person_input = nc.pose_net->blobs()[0]->mutable_gpu_data();

	int total_num_people = 0;
	float* peak_pointer_cpu = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_cpu_data();

	for(int i = 0; i < batch_size; i++){
		nc[gid].num_people[i] = int(peak_pointer_cpu[i * peak_blob_offset]+0.5);
		total_num_people += nc[gid].num_people[i];
	}
	//LOG(ERROR) << "total " << total_num_people << " people";
	if(total_num_people == 0){
		nc[gid].total_num_people = 0;
		return;
	}

	if(total_num_people != nc[gid].total_num_people){
		vector<int> new_shape(4);
		new_shape[0] = total_num_people;
		new_shape[1] = 4;
		new_shape[2] = new_shape[3] = boxsize;
		nc[gid].pose_net->blobs()[0]->Reshape(new_shape);
		nc[gid].pose_net->Reshape();
		//LOG(ERROR) << "pose_net reshaping to " << total_num_people;
		nc[gid].total_num_people = total_num_people;
	}

	float* dst = nc[gid].pose_net->blobs()[0]->mutable_gpu_data();

	float* peak_pointer_gpu = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_gpu_data();
	//LOG(ERROR) << "Getting into fill_pose_net";
	caffe::fill_pose_net(image, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT, dst, boxsize, peak_pointer_gpu, nc[gid].num_people, MAX_PEOPLE_IN_BATCH);
	//src, dst, peak_info, num_people

	// float* pp = nc[gid].pose_net->blobs()[0]->mutable_cpu_data();
	// Mat visualize(boxsize, boxsize, CV_8UC1);
	// for(int i = 0; i < boxsize; i++){
	// 	for(int j = 0; j < boxsize; j++){
	// 		visualize.data[i*boxsize + j] = (unsigned char)(256 * (pp[0*boxsize*boxsize + i*boxsize + j] + 0.5));
	// 	}
	// }
	// imwrite("validate.jpg", visualize);
}

void* getFrameFromCam(void *i){
	VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0)){
        printf("no cam.\n");
        return 0;
    }

    int counter = 1;
    while(1) {
		Mat image_uchar;
		cap >> image_uchar;
		if( image_uchar.empty() ) continue;

		Frame f;
		f.index = counter++;

		// char filename[100];
		// sprintf(filename, "input%03d.jpg", f.index);
		// imwrite(filename, image_uchar);

		f.data_for_wrap = new unsigned char [origin_height * origin_width * 3]; //fill after process
		f.data_for_mat = new float [origin_height * origin_width * 3];
		process_and_pad_image(f.data_for_mat, image_uchar, origin_width, origin_height, 0);

		//resize
		int target_rows = fixed_scale_height; //some fixed number that is multiplier of 8
		int target_cols = float(fixed_scale_height) / image_uchar.rows * image_uchar.cols;
		resize(image_uchar, image_uchar, Size(target_cols, target_rows), 0, 0, INTER_CUBIC);
		if(target_cols % 8 != 0) {
			target_cols = 8 * (target_cols / 8 + 1);
		}

		if(INIT_PERSON_NET_WIDTH != target_cols || INIT_PERSON_NET_HEIGHT != target_rows){
			LOG(ERROR) << "Size not match: " << INIT_PERSON_NET_WIDTH << "  " << target_cols;
			continue;
		}

		//f.init_time = 0; //std::chrono::system_clock::now();
		//LOG(ERROR) << fixed << get_wall_time();
		// if(index == 1){
		// 	INIT_TIME = get_wall_time();
		// 	f.commit_time = INIT_TIME;
		// } else {
		// 	while(INIT_TIME == -999){};
		// 	f.commit_time = INIT_TIME + (1.0f/FPS_SRC) * (index - 1); //relative to 1st frame
		// }
		f.commit_time = get_wall_time();
		
		//pad and transform to float
		f.data = new float [target_rows * target_cols * 3];
		process_and_pad_image(f.data, image_uchar, target_cols, target_rows, 1);

		f.preprocessed_time = get_wall_time();

		// while(global.input_queue.size() > MAX_LENGTH_INPUT_QUEUE){
		//  	//sleep(1);
		// }
		global.input_queue.push(f);
		//LOG(ERROR) << "Frame " << f.index << " committed with init_time " << fixed << f.commit_time;
		//LOG(ERROR) << "pushing frame " << index << " to input_queue, now size " << global.input_queue.size();
		//printGlobal("prepareFrame    ");
		//if(counter == 3) break;
    }
    return nullptr;
}

void* processFrame(void *i){
	int tid = *((int *) i);
	warmup(tid);
	LOG(ERROR) << "GPU " << tid << " is ready";
	Frame f;

	int offset = INIT_PERSON_NET_WIDTH * INIT_PERSON_NET_HEIGHT * 3;
	//bool empty = false;
	
	Frame frame_batch[batch_size];

	//while(!empty){
	while(1){
		//LOG(ERROR) << "start";
		int valid_data = 0;
		for(int n = 0; n < batch_size; n++){
			if(global.input_queue.try_pop(&f)) {
				//consider dropping it
				f.gpu_fetched_time = get_wall_time();
				double elaspsed_time = f.gpu_fetched_time - f.commit_time;
				//LOG(ERROR) << "frame " << f.index << " is copied to GPU after " << elaspsed_time << " sec";
				if(elaspsed_time > 0.1){
					//drop frame
					//LOG(ERROR) << "skip frame " << f.index;
					delete [] f.data;
					delete [] f.data_for_mat;
					delete [] f.data_for_wrap;
					n--;
					global.mtx.lock();
					global.dropped_index.push(f.index);
					global.mtx.unlock();
					continue;
				}
				cudaMemcpy(nc[tid].canvas, f.data_for_mat, origin_width * origin_height * 3 * sizeof(float), cudaMemcpyHostToDevice);

				frame_batch[n] = f;
				//LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
				float* pointer = nc[tid].person_net->blobs()[0]->mutable_gpu_data();

				cudaMemcpy(pointer + n * offset, frame_batch[n].data, offset * sizeof(float), cudaMemcpyHostToDevice);
				valid_data++;
			}
			else {
				//empty = true;
				break;
			}
		}
		if(valid_data == 0) continue;

		//timer.Stop();
		//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		//timer.Start();
		//LOG(ERROR) << "GPU " << tid << ": Running forward person_net";
		//nc.person_net->ForwardFromTo(0,nlayer-1);
		nc[tid].person_net->ForwardFrom(0);
		//cudaDeviceSynchronize();
		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


		/* debug ---------------------------------------*/
		// float* heatmap_pointer = nc[tid].person_net->blobs()[nc[tid].nblob_person-2]->mutable_cpu_data();
		// Mat heatmap(INIT_PERSON_NET_HEIGHT, INIT_PERSON_NET_WIDTH, CV_8UC1); //constructor: height first
		// for (int y = 0; y < INIT_PERSON_NET_HEIGHT; y++){
		// 	for (int x = 0; x < INIT_PERSON_NET_WIDTH; x++){
		// 		float num = heatmap_pointer[y * INIT_PERSON_NET_WIDTH + x]; //0 ~ 1;
		// 		num = (num > 1 ? 1 : (num < 0 ? 0 : num)); //prevent overflow for uchar
		// 		heatmap.data[(y * INIT_PERSON_NET_WIDTH + x)] = (unsigned char)(num * 255);
		// 	}
		// }

		// float* peaks = nc[tid].person_net->blobs()[nc[tid].nblob_person-1]->mutable_cpu_data();
		// for(int i = 0; i < 22; i++){
		// 	cout << peaks[i] << " ";
		// }
		// cout << endl;

		// for(int i = 0; i < peaks[0]; i++){
		// 	circle(heatmap, Point2f(peaks[2*(i+1)], peaks[2*(i+1)+1]), 2, Scalar(0,0,0), -1);
		// }

		// cv::imwrite("person_map.jpg", heatmap);
		/* debug ---------------------------------------*/



		// //copy data to pose_net
		// timer.Start();
		//LOG(ERROR) << "GPU " << tid << ": copy to posenet and reshape";
		copy_to_posenet_and_reshape(tid);
		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		if(nc[tid].total_num_people != 0){
			// timer.Start();
			//LOG(ERROR) << "GPU " << tid << ": Forwarding pose net";
			nc[tid].pose_net->ForwardFrom(0);
			// // nblob = nc.pose_net->blob_names().size();
			// // for(int i = 0; i < nblob; i++) {
			// // 	LOG(ERROR) << "pose_net Blob: " << i << " " << nc.pose_net->blob_names()[i] << " " 
			// // 	           << nc.pose_net->blobs()[i]->shape(0) << " " 
			// // 	           << nc.pose_net->blobs()[i]->shape(1) << " " 
			// // 	           << nc.pose_net->blobs()[i]->shape(2) << " "
			// // 	           << nc.pose_net->blobs()[i]->shape(3) << " ";
			// // }
			// cudaDeviceSynchronize();
			// timer.Stop();
			// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

			// timer.Start();
			// LOG(ERROR) << "Rendering";
			render(tid); //only support batch size = 1!!!!

			//float* out = nc[tid].person_net->blobs()[3]->mutable_gpu_data();

			// float* pp = nc[tid].person_net->blobs()[3]->mutable_cpu_data();
			// Mat visualize(INIT_PERSON_NET_HEIGHT, INIT_PERSON_NET_WIDTH, CV_8UC3);
			// for(int c = 0; c < 3; c++){
			// 	for(int i = 0; i < INIT_PERSON_NET_HEIGHT; i++){
			// 		for(int j = 0; j < INIT_PERSON_NET_WIDTH; j++){
			// 			int value = int(pp[c*INIT_PERSON_NET_HEIGHT*INIT_PERSON_NET_WIDTH + i*INIT_PERSON_NET_WIDTH + j] + 0.5);
			// 			value = value<0 ? 0 : (value > 255 ? 255 : value);
			// 			//cout << value << " ";
			// 			visualize.data[3*(i*INIT_PERSON_NET_WIDTH + j) + c] = (unsigned char)(value);
			// 		}
			// 	}
			// }
			// imwrite("validate.jpg", visualize);


			for(int n = 0; n < valid_data; n++){
				frame_batch[n].numPeople = nc[tid].num_people[n];
				frame_batch[n].gpu_computed_time = get_wall_time();
				cudaMemcpy(frame_batch[n].data_for_mat, nc[tid].canvas, origin_height * origin_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
				global.output_queue.push(frame_batch[n]);
				//LOG(ERROR) << "Pushing data " << index_array[n] << " to output_queue, now size " << global.output_queue.size();
				//printGlobal("processFrame   ");
			}
		}
		else {
			render(tid);
			//frame_batch[n].data should revert to 0-255
			for(int n = 0; n < valid_data; n++){
				frame_batch[n].numPeople = 0;
				frame_batch[n].gpu_computed_time = get_wall_time();
				cudaMemcpy(frame_batch[n].data_for_mat, nc[tid].canvas, origin_height * origin_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
				global.output_queue.push(frame_batch[n]);
			}
		}
		//LOG(ERROR) << "end";
		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";
	}
	return nullptr;
}

class FrameCompare{
public:
    bool operator() (const Frame &a, const Frame &b) const{
        return a.index > b.index;
    }
};

void* buffer_and_order(void* threadargs){ //only one thread can execute this

	FrameCompare comp;
	priority_queue<Frame, vector<Frame>, FrameCompare> buffer(comp);
	Frame f;

	int frame_waited = 1;
	while(1) {
		bool success = global.output_queue_mated.try_pop(&f);
		f.buffer_start_time = get_wall_time();
		if(success){
			//LOG(ERROR) << "buffer getting " << f.index << ", waiting for " << frame_waited;
			global.mtx.lock();
			while(global.dropped_index.size()!=0 && global.dropped_index.top() == frame_waited){
				frame_waited++;
				global.dropped_index.pop();
			}
			global.mtx.unlock();
			//LOG(ERROR) << "while end";

			if(f.index == frame_waited){ //if this is the frame we want, just push it
				f.buffer_end_time = get_wall_time();
				global.output_queue_ordered.push(f);
				frame_waited++;
				while(buffer.size() != 0 && buffer.top().index == frame_waited){
					Frame next = buffer.top();
					buffer.pop();
					next.buffer_end_time = get_wall_time();
					global.output_queue_ordered.push(next);
					frame_waited++;
				}
			}
			else {
				buffer.push(f);
			}

			if(buffer.size() > BUFFER_SIZE){
				//LOG(ERROR) << "buffer squeezed";
				Frame extra = buffer.top();
				buffer.pop();
				//LOG(ERROR) << "popping " << get<0>(extra);
				extra.buffer_end_time = get_wall_time();
				global.output_queue_ordered.push(extra);
				//printGlobal("buffer_and_order");
				frame_waited = extra.index + 1;
				while(buffer.size() != 0 && buffer.top().index == frame_waited){
					Frame next = buffer.top();
					buffer.pop();
					next.buffer_end_time = get_wall_time();
					global.output_queue_ordered.push(next);
					frame_waited++;
				}
			}
		}
		else {
			//output_queue
		}
	}
	return nullptr;
}

void* postProcessFrame(void *i){
	//int tid = *((int *) i);
	Frame f;

	while(1) {
		f = global.output_queue.pop();
		f.postprocesse_begin_time = get_wall_time();
		//printGlobal("postProcessFrame");
		//LOG(ERROR) << "pointer retrieved";
		//cudaMemcpy(f.data_for_mat, f.canvas, origin_width * origin_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

		//Mat visualize(INIT_PERSON_NET_HEIGHT, INIT_PERSON_NET_WIDTH, CV_8UC3);
		int offset = origin_width * origin_height;
		for(int c = 0; c < 3; c++) {
			for(int i = 0; i < origin_height; i++){
				for(int j = 0; j < origin_width; j++){
					int value = int(f.data_for_mat[c*offset + i*origin_width + j] + 0.5);
					value = value<0 ? 0 : (value > 255 ? 255 : value);
					f.data_for_wrap[3*(i*origin_width + j) + c] = (unsigned char)(value);
				}
			}
		}
		f.postprocesse_end_time = get_wall_time();
		global.output_queue_mated.push(f);
	}
	return nullptr;
}

void* displayFrame(void *i) { //single thread
	Frame f;
	int counter = 0;
	double last_time = get_wall_time();
    double this_time;
    float FPS = 0;

	while(1) {
		f = global.output_queue_ordered.pop();
		Mat wrap_frame(origin_height, origin_width, CV_8UC3, f.data_for_wrap);

		//LOG(ERROR) << "processed";
		imshow("video", wrap_frame);

		counter++;
		if(counter % 30 == 0){
            this_time = get_wall_time();
            //LOG(ERROR) << frame.cols << "  " << frame.rows;
            FPS = 30.0f / (this_time - last_time);
            last_time = this_time;

            char msg[1000];
			sprintf(msg, "# %d, NP %d, Latency %.3f, Preprocess %.3f, QueueA %.3f, GPU %.3f, QueueB %.3f, Postproc %.3f, QueueC %.3f, Buffered %.3f, QueueD %.3f, FPS = %.1f", 
			                  f.index, f.numPeople, 
			                  this_time - f.commit_time, 
			                  f.preprocessed_time - f.commit_time, 
			                  f.gpu_fetched_time - f.preprocessed_time,
			                  f.gpu_computed_time - f.gpu_fetched_time,
			                  f.postprocesse_begin_time - f.gpu_computed_time,
			                  f.postprocesse_end_time - f.postprocesse_begin_time,
			                  f.buffer_start_time - f.postprocesse_end_time, 
			                  f.buffer_end_time - f.buffer_start_time,
			                  this_time - f.buffer_end_time, 
			                  FPS);
			LOG(ERROR) << msg;
		}

		
		//LOG(ERROR) << msg;
		waitKey(1);
		//LOG(ERROR) << "showed_and_waited";

		// char filename[256];
		// sprintf(filename, "frame%04d.jpg", index);
		// imwrite(filename, visualize);
		// LOG(ERROR) << "Saved output " << index;
		delete [] f.data_for_mat;
		delete [] f.data_for_wrap;
		delete [] f.data;
	}
	return nullptr;
}

void* listenKey(void *i) { //single thread
	
	//int num; 

	int c;
	while(1){
		//puts ("Enter text. Include a dot ('.') in a sentence to exit:");
		do {
			c = getchar();
			putchar(c);
			int target; 
			if(c >= 48 && c <= 57){
				target = c - 48; // 0 ~ 9
			} else if (c >= 97 && c <= 102){
				target = 10 + (c - 97);
			} else {
				target = -1;
			}
			if(target >= 0 && target <= 16) {
				global.part_to_show = target;
				LOG(ERROR) << "you set to " << target;
			}
		} while (1);
	}
	return nullptr;
}

int rtcpm() {
	caffe::CPUTimer timer;
	timer.Start();

	pthread_t gpu_threads_pool[NUM_GPU];
	for(int gpu = 0; gpu < NUM_GPU; gpu++){
		int *arg = new int[1];
		*arg = gpu;
		int rc = pthread_create(&gpu_threads_pool[gpu], NULL, processFrame, (void *) arg);
		if(rc){
			LOG(ERROR) << "Error:unable to create thread," << rc << "\n";
			exit(-1);
	    }
	}
	LOG(ERROR) << "Finish spawning " << NUM_GPU << " threads. now waiting." << "\n";


	usleep(10 * 1e6);


	/* push all frames for now (single thread) */
	// for(int i = 1; i <= 449; i++){
	// 	char name[256];
	// 	sprintf(name, "../frame/frame%04d.png", i);
	// 	string name_to_push(name);
	// 	global.frame_file_queue.push(make_pair(i,name_to_push));
	// }

	/* grab file name and prepare (multi-thread) */
	int thread_pool_size = 1;
	pthread_t threads_pool[thread_pool_size];
	for(int i = 0; i < thread_pool_size; i++) {
		int *arg = new int[1];
		*arg = i;
    	int rc = pthread_create(&threads_pool[i], NULL, getFrameFromCam, (void *) arg);
    	if(rc){
			LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
			exit(-1);
	    }
	}
	LOG(ERROR) << "Finish spawning " << thread_pool_size << " threads. now waiting." << "\n";


	/* threads handling outputs */
	int thread_pool_size_out = 4;
	pthread_t threads_pool_out[thread_pool_size_out];
	for(int i = 0; i < thread_pool_size_out; i++) {
		int *arg = new int[1];
		*arg = i;
    	int rc = pthread_create(&threads_pool_out[i], NULL, postProcessFrame, (void *) arg);
    	if(rc){
			LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
			exit(-1);
	    }
	}
	LOG(ERROR) << "Finish spawning " << thread_pool_size_out << " threads. now waiting." << "\n";

	/* thread for buffer and ordering frame */
	pthread_t threads_order;
	int *arg = new int[1];
	int rc = pthread_create(&threads_order, NULL, buffer_and_order, (void *) arg);
	if(rc){
		LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
		exit(-1);
    }
	LOG(ERROR) << "Finish spawning the thread for ordering. now waiting." << "\n";

	/* display */
	pthread_t thread_display;
	rc = pthread_create(&thread_display, NULL, displayFrame, (void *) arg);
	if(rc){
		LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
		exit(-1);
    }
	LOG(ERROR) << "Finish spawning the thread for ordering. now waiting." << "\n";

	/* keyboard listener */
	pthread_t thread_key;
	rc = pthread_create(&thread_key, NULL, listenKey, (void *) arg);
	if(rc){
		LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
		exit(-1);
    }
	LOG(ERROR) << "Finish spawning the thread for ordering. now waiting." << "\n";


	for (int i = 0; i < thread_pool_size; i++){
		pthread_join(threads_pool[i], NULL);
	}
	for (int i = 0; i < NUM_GPU; i++){
		pthread_join(gpu_threads_pool[i], NULL);
	}


	LOG(ERROR) << "done";
	timer.Stop();
	LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

 // 	caffe::CPUTimer timer;
	// //copy to device
	// timer.Start();


	// timer.Start();
	// LOG(ERROR) << "Saving";
	// Mat visualize(nc.target_height, nc.target_width, CV_8UC3);
	// for(int c = 0; c < 3; c++){
	// 	for(int i = 0; i < nc.target_height; i++){
	// 		for(int j = 0; j < nc.target_width; j++){
	// 			int value = int(out[c*nc.target_height*nc.target_width + i*nc.target_width + j] + 0.5);
	// 			value = value<0 ? 0 : (value > 255 ? 255 : value);
	// 			visualize.data[3*(i*nc.target_width + j) + c] = (unsigned char)(value);
	// 		}
	// 	}
	// }
	// imwrite("out.jpg", visualize);
	// timer.Stop();
	// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


	return 0;
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

void dostuff (int sock) {
	/******** DOSTUFF() *********************
	There is a separate instance of this function 
	for each connection.  It handles all communication
	once a connnection has been established.
	*****************************************/
	int n;
	char buffer[256];
	  
	bzero(buffer,256);
	n = read(sock,buffer,255);
	if (n < 0) error("ERROR reading from socket");
	printf("Here is the message from client: %s\n", buffer);

	string filename(buffer);
	int status = rtcpm();
	//int status = 0;
	if(status == 0){
		char msg[] = "You image is processed";
		n = write(sock, msg, strlen(msg));
		if (n < 0) LOG(ERROR) << "Error when writing to socket";
	}
	else {
		char msg[] = "Have problems processing your image";
		n = write(sock, msg, strlen(msg));
		if (n < 0) LOG(ERROR) << "Error when writing to socket";
	}
}

void error(const char *msg){
    perror(msg);
    exit(1);
}

int main(int argc, char** argv) {

	NUM_GPU = atoi(argv[1]); // 4
	batch_size = atoi(argv[2]); //1

	/* server warming up */
	set_nets();
	//warmup();
	::google::InitGoogleLogging("rtcpm");
	rtcpm();


	// int frame_num = 1;
	// while (frame_num <= 449) {
	// 	LOG(ERROR) << "head";
	//     //clock_t startTime = clock();

	//     char filename[256];
	//     sprintf(filename, "frame%04d.jpg", frame_num);
	//     Mat frame = imread(filename, CV_LOAD_IMAGE_COLOR);
	//     imshow("video", frame);

	//     frame_num++;

	//     //while (clock() - startTime < delay) {
	//     LOG(ERROR) << "showed";
	//     waitKey(30);
	//     //boost::this_thread::sleep( boost::posix_time::milliseconds(1000) );
	//     LOG(ERROR) << "showed_and_waited";
	//     //}
	// }

	/* begin socket programming */
 	// int sockfd, newsockfd, portno; /*, pid;*/
  //   socklen_t clilen;
  //   struct sockaddr_in serv_addr, cli_addr;
  //   signal(SIGCHLD, SIG_IGN);

  //   if (argc < 2) {
  //       fprintf(stderr, "ERROR, no port provided\n");
  //       exit(1);
  //   }
  //   sockfd = socket(AF_INET, SOCK_STREAM, 0);
  //   if (sockfd < 0) 
  //       error("ERROR opening socket");
  //   bzero((char *) &serv_addr, sizeof(serv_addr));
  //   portno = atoi(argv[1]);
  //   serv_addr.sin_family = AF_INET;
  //   serv_addr.sin_addr.s_addr = INADDR_ANY;
  //   serv_addr.sin_port = htons(portno);
  //   if (bind(sockfd, (struct sockaddr *) &serv_addr,
  //             sizeof(serv_addr)) < 0) 
  //             error("ERROR on binding");
  //   listen(sockfd,5);
  //   clilen = sizeof(cli_addr);
  //   LOG(ERROR) << "Init ready, waiting for request....";
  //   while (1) {
  //       newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
  //       if (newsockfd < 0) 
  //           error("ERROR on accept");
  //       //pid = fork();
  //       //if (pid < 0)
  //       //    error("ERROR on fork");
  //       //if (pid == 0)  {
  //       //    close(sockfd);
  //           dostuff(newsockfd);
  //       //    exit(0);
  //       //}
  //       //else close(newsockfd);
  //   } /* end of while */
  //   close(sockfd);

    return 0; /* we never get here */
}

