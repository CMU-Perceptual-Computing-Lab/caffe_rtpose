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
#include <boost/filesystem.hpp>

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
#define MAX_LENGTH_INPUT_QUEUE 5000 //affects memory usage
#define FPS_SRC 30

int NUM_GPU;  //4
int batch_size;  //2
double INIT_TIME = -999;

//model
string caffemodel;
string prototxt;
string writeFolder;
string testSet;
string expName;
int iteration;

// network copy for each gpu thread
struct NET_COPY {
	Net<float>* net;
	vector<int> num_people;
	int nblob;
	int total_num_people;
	float* canvas; // GPU memory

	int ch_heat;
	int ch_vec;
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

	string test_image_info_file; // a txt file contains image id/w/h etc
	string image_file_header;

	vector<float> scale_search;
};

NET_COPY nc[4];
GLOBAL global;

void set_nets(char** argv);
int rtcpm();
void putGaussianMaps(float* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
void warmup(int);
void error(const char *msg);
//void process_and_pad_image(float* target, Mat oriImg, int tw, int th, bool normalize);


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

void set_nets(char** argv){
	caffemodel = argv[4]; //"/home/shihenw/Research/coco_challenge/CocoChallengeTrainTest/caffe_model/pose_exp04_zhe_exp14/pose_iter_316000.caffemodel"; //"/media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/model/pose_iter_70000.caffemodel";
	prototxt = argv[5]; //"/home/shihenw/Research/coco_challenge/CocoChallengeTrainTest/pose_exp_caffe/pose_exp04_zhe_exp14/pose_deploy.prototxt"; ///media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/pose_deploy_copy_4sg_resize.prototxt";
	writeFolder = argv[6]; //"/data1/val2014/pose_exp04_zhe_exp14/pose_iter_316000/";
	testSet = argv[7]; //"val2014";
	expName = argv[8];  //pose_exp04_zhe_exp14";
	iteration = atoi(argv[9]); //316000;
}

void warmup(int device_id){
	
	LOG(ERROR) << "Setting GPU " << device_id;
	Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
	Caffe::set_mode(Caffe::GPU); //

	LOG(ERROR) << "GPU " << device_id << ": copying to person net";
	nc[device_id].net = new Net<float>(prototxt, caffe::TEST);
	nc[device_id].net->CopyTrainedLayersFrom(caffemodel);
	nc[device_id].nblob = nc[device_id].net->blob_names().size();
	
	vector<int> shape(4);
	shape[0] = 1;
	shape[1] = 3;
	shape[2] = boxsize;
	shape[3] = boxsize;
	nc[device_id].net->blobs()[0]->Reshape(shape);
	nc[device_id].net->Reshape();

	nc[device_id].ch_heat = nc[device_id].net->blobs()[nc[device_id].nblob-1]->shape()[1];
	nc[device_id].ch_vec = nc[device_id].net->blobs()[nc[device_id].nblob-2]->shape()[1];

	for(int i = 0; i < nc[device_id].nblob; i++){
		LOG(ERROR) << "person_net blob: " << i << " " << nc[device_id].net->blob_names()[i] << " " 
		           << nc[device_id].net->blobs()[i]->shape(0) << " " 
		           << nc[device_id].net->blobs()[i]->shape(1) << " " 
		           << nc[device_id].net->blobs()[i]->shape(2) << " "
		           << nc[device_id].net->blobs()[i]->shape(3) << " ";
	}
	
	//dry run
	LOG(ERROR) << "Dry running...";
	nc[device_id].net->ForwardFrom(0);

	//cudaMalloc(&nc[device_id].canvas, origin_width * origin_height * 3 * sizeof(float));
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

	// static int counter = 0;
	// Mat test(th, tw, CV_8UC1);
	// for(int y = 0; y < th; y++){
	// 	for(int x = 0; x < tw; x++){
	// 		test.data[y * tw + x] = (unsigned int)((target[y * tw + x] + 0.5) * 256);
	// 	}
	// }
	// counter++;
	// char filename[100];
	// sprintf(filename, "validate_%04d.jpg", counter);
	// cv::imwrite(filename, test);
}

// void render(int gid) {
// 	// LOG(ERROR) << "begin render";
// 	// LOG(ERROR) << nc.person_net->blob_names()[3] << " " << 
// 	// 			  nc.person_net->blobs()[3]->shape(0) << " " << 
// 	//               nc.person_net->blobs()[3]->shape(1) << " " << 
// 	//               nc.person_net->blobs()[3]->shape(2) << " " <<
// 	//               nc.person_net->blobs()[3]->shape(3);

// 	//float* canvas = nc[gid].person_net->blobs()[3]->mutable_gpu_data(); //render layer
// 	//float* image_ref = nc[gid].person_net->blobs()[0]->mutable_gpu_data();
// 	float* centers  = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_gpu_data();
// 	float* heatmaps = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-2]->mutable_gpu_data();
// 	float* poses    = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-1]->mutable_gpu_data(); 
	
// 	//LOG(ERROR) << "begin render_in_cuda";
// 	caffe::render_in_cuda_website_indi(nc[gid].canvas, origin_width, origin_height, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT, 
// 		                  heatmaps, boxsize, 
// 		                  centers, poses, nc[gid].num_people, global.part_to_show);

// 	// float* poses_cpu = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-1]->mutable_cpu_data();
// 	// for(int i = 0; i < nc[gid].total_num_people * 45; i++){
// 	// 	cout << poses_cpu[i] << " ";
// 	// 	if(i % 45 == 44) cout << endl; 
// 	// }

// 	// float* pp = nc.person_net->blobs()[3]->mutable_cpu_data();
// 	// Mat visualize(nc.target_height, nc.target_width, CV_8UC3);
// 	// for(int c = 0; c < 3; c++){
// 	// 	for(int i = 0; i < nc.target_height; i++){
// 	// 		for(int j = 0; j < nc.target_width; j++){
// 	// 			int value = int(pp[c*nc.target_height*nc.target_width + i*nc.target_width + j] + 0.5);
// 	// 			value = value<0 ? 0 : (value > 255 ? 255 : value);
// 	// 			visualize.data[3*(i*nc.target_width + j) + c] = (unsigned char)(value);
// 	// 		}
// 	// 	}
// 	// }
// 	// imwrite("validate.jpg", visualize);
// }

// void copy_to_posenet_and_reshape(int gid){
// 	float* image = nc[gid].person_net->blobs()[0]->mutable_gpu_data();
// 	//float* person_input = nc.pose_net->blobs()[0]->mutable_gpu_data();

// 	int total_num_people = 0;
// 	float* peak_pointer_cpu = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_cpu_data();

// 	for(int i = 0; i < batch_size; i++){
// 		nc[gid].num_people[i] = int(peak_pointer_cpu[i * peak_blob_offset]+0.5);
// 		total_num_people += nc[gid].num_people[i];
// 	}
// 	//LOG(ERROR) << "total " << total_num_people << " people";
// 	if(total_num_people == 0){
// 		nc[gid].total_num_people = 0;
// 		return;
// 	}

// 	if(total_num_people != nc[gid].total_num_people){
// 		vector<int> new_shape(4);
// 		new_shape[0] = total_num_people;
// 		new_shape[1] = 4;
// 		new_shape[2] = new_shape[3] = boxsize;
// 		nc[gid].pose_net->blobs()[0]->Reshape(new_shape);
// 		nc[gid].pose_net->Reshape();
// 		//LOG(ERROR) << "pose_net reshaping to " << total_num_people;
// 		nc[gid].total_num_people = total_num_people;
// 	}

// 	float* dst = nc[gid].pose_net->blobs()[0]->mutable_gpu_data();

// 	float* peak_pointer_gpu = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_gpu_data();
// 	//LOG(ERROR) << "Getting into fill_pose_net";
// 	caffe::fill_pose_net(image, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT, dst, boxsize, peak_pointer_gpu, nc[gid].num_people, MAX_PEOPLE_IN_BATCH);
// 	//src, dst, peak_info, num_people

// 	// float* pp = nc[gid].pose_net->blobs()[0]->mutable_cpu_data();
// 	// Mat visualize(boxsize, boxsize, CV_8UC1);
// 	// for(int i = 0; i < boxsize; i++){
// 	// 	for(int j = 0; j < boxsize; j++){
// 	// 		visualize.data[i*boxsize + j] = (unsigned char)(256 * (pp[0*boxsize*boxsize + i*boxsize + j] + 0.5));
// 	// 	}
// 	// }
// 	// imwrite("validate.jpg", visualize);
// }

void* getFrameFromCam(void *i){
	
	printf("reading file %s....\n", global.test_image_info_file.c_str());
	
	ifstream infile;
	infile.open(global.test_image_info_file);

	int counter, id, height, width;
	string filename;

	while(!infile.eof()){
		infile >> counter >> id >> filename >> height >> width;

    	//cout << counter << id << filename << height << width << endl;
		Mat image_uchar = imread(global.image_file_header + filename, CV_LOAD_IMAGE_COLOR);

		assert(image_uchar.cols == width && image_uchar.rows == height);

		for(int s = 0; s < global.scale_search.size(); s++){ // throw different scales into queue

			char subFolder[1000];
			sprintf(subFolder, "/data1/COCO/%s/%s/pose_iter_%06d/scale_%.1f", testSet.c_str(), expName.c_str(), 
				                                                              iteration, global.scale_search[s]);
			char writeFile[1000];
			sprintf(writeFile, "%s/COCO_%s_%012d.binary", subFolder, testSet.c_str(), id);

			if(boost::filesystem::exists(writeFile)){
				//LOG(ERROR) << writeFile << "exist! ";
				continue;
			}

			Mat image_uchar_scaled;

			Frame f;
			f.index = id;
			f.counter = counter;
			f.ori_width = width;
			f.ori_height = height;

			// char filename[100];
			// sprintf(filename, "input%03d.jpg", f.index);
			// imwrite(filename, image_uchar);

			//resize
			double scale0 = double(boxsize) / height;
			double scale = global.scale_search[s] * scale0;

			resize(image_uchar, image_uchar_scaled, Size(ceil(scale*image_uchar.cols), ceil(scale*image_uchar.rows)), 0, 0, INTER_CUBIC);

			f.scaled_width = image_uchar_scaled.cols;
			f.scaled_height = image_uchar_scaled.rows;

			int target_rows = f.scaled_height; //some fixed number that is multiplier of 8
			if(target_rows % 8 != 0) {
				target_rows = 8 * (target_rows / 8 + 1);
			}
			int target_cols = f.scaled_width;
			if(target_cols % 8 != 0) {
				target_cols = 8 * (target_cols / 8 + 1);
			}

			f.padded_width = target_cols;
			f.padded_height = target_rows;

			f.scale = global.scale_search[s];
			//f.data_for_wrap = new unsigned char [origin_height * origin_width * 3]; //fill after process
			//f.data_for_mat = new float [origin_height * origin_width * 3];
			//process_and_pad_image(f.data_for_mat, image_uchar, origin_width, origin_height, 0);

			

			// if(INIT_PERSON_NET_WIDTH != target_cols || INIT_PERSON_NET_HEIGHT != target_rows){
			// 	LOG(ERROR) << "Size not match: " << INIT_PERSON_NET_WIDTH << "  " << target_cols;
			// 	continue;
			// }

			// //f.init_time = 0; //std::chrono::system_clock::now();
			// //LOG(ERROR) << fixed << get_wall_time();
			// // if(index == 1){
			// // 	INIT_TIME = get_wall_time();
			// // 	f.commit_time = INIT_TIME;
			// // } else {
			// // 	while(INIT_TIME == -999){};
			// // 	f.commit_time = INIT_TIME + (1.0f/FPS_SRC) * (index - 1); //relative to 1st frame
			// // }
			// f.commit_time = get_wall_time();
			
			// //pad and transform to float
			f.data = new float [target_rows * target_cols * 3];
			process_and_pad_image(f.data, image_uchar_scaled, target_cols, target_rows, 1);

			// f.preprocessed_time = get_wall_time();

			while(global.input_queue.size() > MAX_LENGTH_INPUT_QUEUE){
				sleep(1);
			}
			global.input_queue.push(f);
			//LOG(ERROR) << "Frame " << f.index << " committed with init_time " << fixed << f.commit_time;
			//LOG(ERROR) << "pushing frame " << index << " to input_queue, now size " << global.input_queue.size();
			//printGlobal("prepareFrame    ");
			//if(counter == 3) break;
		}
    }
    return nullptr;
}

void* processFrame(void *i){
	int tid = *((int *) i);
	warmup(tid);
	LOG(ERROR) << "GPU " << tid << " is ready";
	Frame f;
	bool empty = false;
	
	Frame frame_batch[batch_size];

	//while(!empty){
	while(1){
		//LOG(ERROR) << "start";

		for(int n = 0; n < batch_size; n++){
			if(global.input_queue.try_pop(&f)) {
				frame_batch[n] = f;

				vector<int> shape(4);
				shape[0] = 1;
				shape[1] = 3;
				shape[2] = frame_batch[n].padded_height;
				shape[3] = frame_batch[n].padded_width;
				nc[tid].net->blobs()[0]->Reshape(shape);
				nc[tid].net->Reshape();

				//LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
				float* pointer = nc[tid].net->blobs()[0]->mutable_gpu_data();

				int offset = frame_batch[n].padded_width * frame_batch[n].padded_height * 3;
				cudaMemcpy(pointer + n * offset, frame_batch[n].data, offset * sizeof(float), cudaMemcpyHostToDevice);
			}
			else {
				empty = true;
				LOG(ERROR) << "input_queue is empty!";
				break;
			}
		}

		if(empty) {
			return nullptr;
		}
		//timer.Stop();
		//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		//timer.Start();
		//LOG(ERROR) << "GPU " << tid << ": Running forward person_net";
		//nc.person_net->ForwardFromTo(0,nlayer-1);
		nc[tid].net->ForwardFrom(0);
		
		//do some copy here
		int num_blobs = nc[tid].nblob;
		float* pointer_heatmap = nc[tid].net->blobs()[num_blobs-1]->mutable_gpu_data();
		float* pointer_vecmap = nc[tid].net->blobs()[num_blobs-2]->mutable_gpu_data();

		//LOG(ERROR) << "ch_heat " << nc[tid].ch_heat << "ch_vec " << nc[tid].ch_vec;

		frame_batch[0].net_output = new float [frame_batch[0].padded_height * frame_batch[0].padded_width / 8 / 8 * (nc[tid].ch_heat + nc[tid].ch_vec)];
		//frame_batch[0].net_output = new float [frame_batch[0].padded_height * frame_batch[0].padded_width * (nc[tid].ch_heat + nc[tid].ch_vec)];

		//copy heatmap
		cudaMemcpy(frame_batch[0].net_output, 
				   pointer_heatmap, 
			       frame_batch[0].padded_height * frame_batch[0].padded_width / 8 / 8 * nc[tid].ch_heat * sizeof(float), 
			       cudaMemcpyDeviceToHost);
		// cudaMemcpy(frame_batch[0].net_output, 
		// 		   pointer_heatmap, 
		// 	       frame_batch[0].padded_height * frame_batch[0].padded_width * nc[tid].ch_heat * sizeof(float), 
		// 	       cudaMemcpyDeviceToHost);

		//copy vecmap
		cudaMemcpy(frame_batch[0].net_output + (frame_batch[0].padded_height * frame_batch[0].padded_width / 8 / 8 * nc[tid].ch_heat), 
				   pointer_vecmap, 
			       frame_batch[0].padded_height * frame_batch[0].padded_width / 8 / 8 * nc[tid].ch_vec * sizeof(float), 
			       cudaMemcpyDeviceToHost);
		// cudaMemcpy(frame_batch[0].net_output + (frame_batch[0].padded_height * frame_batch[0].padded_width * nc[tid].ch_heat), 
		// 		   pointer_vecmap, 
		// 	       frame_batch[0].padded_height * frame_batch[0].padded_width * nc[tid].ch_vec * sizeof(float), 
		// 	       cudaMemcpyDeviceToHost);

		global.output_queue.push(frame_batch[0]);

		printGlobal("processFrame   ");


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

		//LOG(ERROR) << "end";
		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";
	}
	return nullptr;
}

// class FrameCompare{
// public:
//     bool operator() (const Frame &a, const Frame &b) const{
//         return a.index > b.index;
//     }
// };

// void* buffer_and_order(void* threadargs){ //only one thread can execute this

// 	FrameCompare comp;
// 	priority_queue<Frame, vector<Frame>, FrameCompare> buffer(comp);
// 	Frame f;

// 	int frame_waited = 1;
// 	while(1) {
// 		bool success = global.output_queue_mated.try_pop(&f);
// 		f.buffer_start_time = get_wall_time();
// 		if(success){
// 			//LOG(ERROR) << "buffer getting " << f.index << ", waiting for " << frame_waited;
// 			global.mtx.lock();
// 			while(global.dropped_index.size()!=0 && global.dropped_index.top() == frame_waited){
// 				frame_waited++;
// 				global.dropped_index.pop();
// 			}
// 			global.mtx.unlock();
// 			//LOG(ERROR) << "while end";

// 			if(f.index == frame_waited){ //if this is the frame we want, just push it
// 				f.buffer_end_time = get_wall_time();
// 				global.output_queue_ordered.push(f);
// 				frame_waited++;
// 				while(buffer.size() != 0 && buffer.top().index == frame_waited){
// 					Frame next = buffer.top();
// 					buffer.pop();
// 					next.buffer_end_time = get_wall_time();
// 					global.output_queue_ordered.push(next);
// 					frame_waited++;
// 				}
// 			}
// 			else {
// 				buffer.push(f);
// 			}

// 			if(buffer.size() > BUFFER_SIZE){
// 				//LOG(ERROR) << "buffer squeezed";
// 				Frame extra = buffer.top();
// 				buffer.pop();
// 				//LOG(ERROR) << "popping " << get<0>(extra);
// 				extra.buffer_end_time = get_wall_time();
// 				global.output_queue_ordered.push(extra);
// 				//printGlobal("buffer_and_order");
// 				frame_waited = extra.index + 1;
// 				while(buffer.size() != 0 && buffer.top().index == frame_waited){
// 					Frame next = buffer.top();
// 					buffer.pop();
// 					next.buffer_end_time = get_wall_time();
// 					global.output_queue_ordered.push(next);
// 					frame_waited++;
// 				}
// 			}
// 		}
// 		else {
// 			//output_queue
// 		}
// 	}
// 	return nullptr;
// }

void* postProcessFrame(void *i){
	//int tid = *((int *) i);
	Frame f;

	while(1) {

		if(global.output_queue.try_pop(&f)){
			//f = global.output_queue.pop();
			//f.postprocesse_begin_time = get_wall_time();
			//printGlobal("postProcessFrame");
			//LOG(ERROR) << "pointer retrieved";
			//cudaMemcpy(f.data_for_mat, f.canvas, origin_width * origin_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

			//Mat visualize(INIT_PERSON_NET_HEIGHT, INIT_PERSON_NET_WIDTH, CV_8UC3);

			//f.net_output has padded_size
			
			//write into file
			char subFolder[1000];
			sprintf(subFolder, "/data1/COCO/%s/%s/pose_iter_%06d/scale_%.1f", testSet.c_str(), expName.c_str(), 
				                                                              iteration, f.scale);
			boost::filesystem::create_directories(subFolder);
			//LOG(ERROR) << "folder " << subFolder << " created";

			char writeFile[1000];
			sprintf(writeFile, "%s/COCO_%s_%012d.binary", subFolder, testSet.c_str(), f.index);

			ofstream OutFile;
			OutFile.open(writeFile, ios::out | ios::binary);


			// int offset2 = f.padded_height * f.padded_width;
			// int offset1 = f.padded_width;
			
			// for(int c = 0; c < (nc[0].ch_heat + nc[0].ch_vec); c++) {
			// 	Mat wrap_frame(f.scaled_height, f.scaled_width, CV_32F);
			// 	for(int y = 0; y < f.scaled_height; y++) {
			// 		for(int x = 0; x < f.scaled_width; x++) {
			// 			wrap_frame.at<float>(y,x) = f.net_output[c*offset2 + y*offset1 + x];
			// 		}
			// 	}
			// 	resize(wrap_frame, wrap_frame, Size(f.ori_width, f.ori_height), 0, 0, INTER_CUBIC);

			// 	for(int y = 0; y < f.ori_height; y++) {
			// 		for(int x = 0; x < f.ori_width; x++) {
			// 			float my_float = wrap_frame.at<float>(y,x);
			// 			OutFile.write( (char*)&my_float, sizeof(float));
			// 		}
			// 	}
			// }


			

			//LOG(ERROR) << "FILE " << writeFile << " opened";
			for(int c = 0; c < f.padded_width * f.padded_height / 8 / 8 * (nc[0].ch_heat + nc[0].ch_vec); c++){
			//for(int c = 0; c < f.padded_width * f.padded_height * (nc[0].ch_heat + nc[0].ch_vec); c++){
				float my_float = f.net_output[c];
				OutFile.write( (char*)&my_float, sizeof(float));
			}

			// OutFile.close();

			delete [] f.net_output;
			delete [] f.data;
			// int offset = origin_width * origin_height;
			// for(int c = 0; c < 3; c++) {
			// 	for(int i = 0; i < origin_height; i++){
			// 		for(int j = 0; j < origin_width; j++){
			// 			int value = int(f.data_for_mat[c*offset + i*origin_width + j] + 0.5);
			// 			value = value<0 ? 0 : (value > 255 ? 255 : value);
			// 			f.data_for_wrap[3*(i*origin_width + j) + c] = (unsigned char)(value);
			// 		}
			// 	}
			// }
			//f.postprocesse_end_time = get_wall_time();
			global.output_queue_mated.push(f);
		} else {
			//LOG(ERROR) << "output_queue empty";
			//break;
		}
	}
	return nullptr;
}

// void* displayFrame(void *i) { //single thread
// 	Frame f;
// 	int counter = 0;
// 	double last_time = get_wall_time();
//     double this_time;
//     float FPS = 0;

// 	while(1) {
// 		f = global.output_queue_ordered.pop();
// 		Mat wrap_frame(origin_height, origin_width, CV_8UC3, f.data_for_wrap);

// 		//LOG(ERROR) << "processed";
// 		imshow("video", wrap_frame);

// 		counter++;
// 		if(counter % 30 == 0){
//             this_time = get_wall_time();
//             //LOG(ERROR) << frame.cols << "  " << frame.rows;
//             FPS = 30.0f / (this_time - last_time);
//             last_time = this_time;

//             char msg[1000];
// 			sprintf(msg, "# %d, NP %d, Latency %.3f, Preprocess %.3f, QueueA %.3f, GPU %.3f, QueueB %.3f, Postproc %.3f, QueueC %.3f, Buffered %.3f, QueueD %.3f, FPS = %.1f", 
// 			                  f.index, f.numPeople, 
// 			                  this_time - f.commit_time, 
// 			                  f.preprocessed_time - f.commit_time, 
// 			                  f.gpu_fetched_time - f.preprocessed_time,
// 			                  f.gpu_computed_time - f.gpu_fetched_time,
// 			                  f.postprocesse_begin_time - f.gpu_computed_time,
// 			                  f.postprocesse_end_time - f.postprocesse_begin_time,
// 			                  f.buffer_start_time - f.postprocesse_end_time, 
// 			                  f.buffer_end_time - f.buffer_start_time,
// 			                  this_time - f.buffer_end_time, 
// 			                  FPS);
// 			LOG(ERROR) << msg;
// 		}

		
// 		//LOG(ERROR) << msg;
// 		waitKey(1);
// 		//LOG(ERROR) << "showed_and_waited";

// 		// char filename[256];
// 		// sprintf(filename, "frame%04d.jpg", index);
// 		// imwrite(filename, visualize);
// 		// LOG(ERROR) << "Saved output " << index;
// 		delete [] f.data_for_mat;
// 		delete [] f.data_for_wrap;
// 		delete [] f.data;
// 	}
// 	return nullptr;
// }

// void* listenKey(void *i) { //single thread
	
// 	//int num; 

// 	int c;
// 	while(1){
// 		//puts ("Enter text. Include a dot ('.') in a sentence to exit:");
// 		do {
// 			c = getchar();
// 			putchar(c);
// 			int target; 
// 			if(c >= 48 && c <= 57){
// 				target = c - 48; // 0 ~ 9
// 			} else if (c >= 97 && c <= 102){
// 				target = 10 + (c - 97);
// 			} else {
// 				target = -1;
// 			}
// 			if(target >= 0 && target <= 16) {
// 				global.part_to_show = target;
// 				LOG(ERROR) << "you set to " << target;
// 			}
// 		} while (1);
// 	}
// 	return nullptr;
// }

int rtcpm() {
	caffe::CPUTimer timer;
	timer.Start();

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


	usleep(15 * 1e6); // wait gpu to init


	/* threads handling outputs */
	int thread_pool_size_out = 6;
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




	for (int i = 0; i < thread_pool_size; i++){
		pthread_join(threads_pool[i], NULL);
		LOG(ERROR) << "all pushed to input queue!!!";
	}
	for (int i = 0; i < thread_pool_size_out; i++){
		pthread_join(threads_pool_out[i], NULL);
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

void error(const char *msg){
    perror(msg);
    exit(1);
}

int main(int argc, char** argv) {

	NUM_GPU = atoi(argv[1]); // 4
	batch_size = 1; //1

	string foo(argv[2]);
	global.test_image_info_file = foo;

	string foo2(argv[3]);
	global.image_file_header = foo2;

	char* pch;
	pch = strtok(argv[10], ",");
  	while (pch != NULL) {
    	printf ("%f\n", atof(pch));
    	global.scale_search.push_back(atof(pch));
    	pch = strtok (NULL, ",");
	}
	
	//global.scale_search.resize(4);
	//global.scale_search[0] = 0.5;
	//global.scale_search[1] = 1;
	//global.scale_search[2] = 1.5;
	//global.scale_search[3] = 2;

	/* server warming up */
	set_nets(argv);
	//warmup();
	::google::InitGoogleLogging("rtcpm");
	rtcpm();

    return 0; /* we never get here */
}

