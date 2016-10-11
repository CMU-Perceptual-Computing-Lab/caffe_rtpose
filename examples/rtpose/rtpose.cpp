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

#include <gflags/gflags.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <boost/thread/thread.hpp>

#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/switch_layer.hpp"
#include "caffe/layers/nms_layer.hpp"
#include "caffe/layers/imresize_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/render_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <sys/time.h>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <mutex>
#include <algorithm>

#include "modeldesc.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;
using namespace std;
using namespace cv;

DEFINE_bool(fullscreen, false,
						 "Run in fullscreen mode (press f during runtime to toggle)");
DEFINE_int32(part_to_show, 0,
						 "Part to show from the start.");
DEFINE_string(write_frames, "",
						 "Write frames with format prefix%06d.jpg");
DEFINE_bool(no_frame_drops, false,
						"Dont drop frames.");

DEFINE_int32(camera, 0,
						 "The camera index for VideoCapture.");
DEFINE_string(video, "",
						 "Use a video file instead of the camera.");
DEFINE_int32(start_frame, 0,
						 "Skip to frame # of video");


DEFINE_string(caffemodel, "model/coco/pose_iter_440000.caffemodel",
							"Caffe model.");
DEFINE_string(caffeproto, "model/coco/pose_deploy_linevec.prototxt",
							"Caffe deploy prototxt.");

DEFINE_string(resolution, "1280x720",
						 "The image resolution (display).");
DEFINE_string(net_resolution, "656x368",
						 "Multiples of 16.");
DEFINE_string(camera_resolution, "1280x720",
						 "Size of the camera frames to ask for.");

DEFINE_int32(start_device, 0,
						 "GPU device start number.");
DEFINE_int32(num_gpu, 1,
						 "The number of GPU devices to use.");

DEFINE_double(start_scale, 1,
						 "Initial scale. Must match net_resolution");
DEFINE_double(scale_gap, 0.3,
						 "Scale gap between scales. No effect unless num_scales>1");
DEFINE_int32(num_scales, 1,
						 "Number of scales to average");

// These are set to match FLAGS_resolution
int origin_width=960; //960 //1280 //640
int origin_height=540; //540 //720 //480

// These are set to match FLAGS_camera_resolution
// TODO: clean up the defines and duplicate vars and such
int camera_frame_width=1920;
int camera_frame_height=1080;
int init_person_net_height = (368);
int init_person_net_width = (356);

#define boxsize 368
#define fixed_scale_height 368
#define peak_blob_offset 33
#define INIT_PERSON_NET_HEIGHT init_person_net_height
#define INIT_PERSON_NET_WIDTH  init_person_net_width
#define MAX_PEOPLE_IN_BATCH 32
#define BUFFER_SIZE 4    //affects latency
#define MAX_LENGTH_INPUT_QUEUE 500 //affects memory usage
#define FPS_SRC 30
#define batch_size FLAGS_num_scales

#define MAX_NUM_PARTS 70
// This is defined in render_functions.hpp
#define MAX_PEOPLE RENDER_MAX_PEOPLE
#define MAX_MAX_PEAKS 96


#define start_scale FLAGS_start_scale
#define scale_gap FLAGS_scale_gap

int NUM_GPU;  //4
double INIT_TIME = -999;

//person detector
string person_detector_caffemodel;
string person_detector_proto;

//pose estimator
string pose_estimator_proto;

// network copy for each gpu thread
struct NET_COPY {
	Net<float>* person_net;
	Net<float>* pose_net;
	vector<int> num_people;
	int nblob_person;
	int nblob_pose;
	int total_num_people;
	int nms_max_peaks;
	int nms_num_parts;
	ModelDescriptor *modeldesc;
	float* canvas; // GPU memory
	float* joints; // GPU memory
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
	float target_size[1][2];
	bool quit_threads;
	// Parameters
	float nms_threshold;
	int connect_min_subset_cnt;
	float connect_min_subset_score;
	float connect_inter_threshold;
	int connect_inter_min_above_threshold;

	struct UIState {
		UIState() :
			is_fullscreen(0),
			is_video_paused(0),
			is_shift_down(0),
			is_googly_eyes(0),
			current_frame(0),
			seek_to_frame(-1),
			select_stage(-1),
			fps(0) {}
		bool is_fullscreen;
		bool is_video_paused;
		bool is_shift_down;
		bool is_googly_eyes;
		int current_frame;
		int seek_to_frame;
		int select_stage;
		double fps;
	};
	UIState uistate;
 };

struct ColumnCompare
{
    bool operator()(const std::vector<double>& lhs,
                    const std::vector<double>& rhs) const
    {
        return lhs[2] > rhs[2];
        //return lhs[0] > rhs[0];
    }
};

NET_COPY nc[4];
GLOBAL global;

// TODO: Clean this up
ModelDescriptor *model_descriptor = 0;

void set_nets();
int rtcpm();
bool handleKey(int c);
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
	VLOG(3) << src << "\t input_queue: " << global.input_queue.size()
	           << " | output_queue: " << global.output_queue.size()
	           << " | output_queue_ordered: " << global.output_queue_ordered.size()
	           << " | output_queue_mated: " << global.output_queue_mated.size();
}

void set_nets(){
	//person_detector_caffemodel = "../model/pose_iter_70000.caffemodel"; //"/media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/model/pose_iter_70000.caffemodel";
	//person_detector_proto = "../model/pose_deploy_copy_4sg_resize.prototxt"; ///media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/pose_deploy_copy_4sg_resize.prototxt";
	person_detector_caffemodel = FLAGS_caffemodel;	// "/media/posenas4b/User/zhe/arch/MPI_exp_caffe/poseDP/exp3/pose_deploy.prototxt"
	person_detector_proto = FLAGS_caffeproto; //_29parts.prototxt"; // "/media/posenas4b/User/zhe/arch/MPI_exp_caffe/poseDP/exp3/model/pose_iter_600000.caffemodel"
}

void warmup(int device_id){

	int logtostderr = FLAGS_logtostderr;

	LOG(INFO) << "Setting GPU " << device_id;

	Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
	Caffe::set_mode(Caffe::GPU); //

	LOG(INFO) << "GPU " << device_id << ": copying to person net";
	FLAGS_logtostderr = 0;
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
	FLAGS_logtostderr = logtostderr;

	caffe::NmsLayer<float> *nms_layer =
		(caffe::NmsLayer<float>*)nc[device_id].person_net->layer_by_name("nms").get();
	nc[device_id].nms_max_peaks = nms_layer->GetMaxPeaks();


	caffe::ImResizeLayer<float> *resize_layer =
		(caffe::ImResizeLayer<float>*)nc[device_id].person_net->layer_by_name("resize").get();

	resize_layer->SetStartScale(start_scale);
	resize_layer->SetScaleGap(scale_gap);
	LOG(INFO) << "start_scale = " << start_scale;

	nc[device_id].nms_max_peaks = nms_layer->GetMaxPeaks();

	// CHECK_EQ(nc[device_id].nms_max_peaks, 20)
	// 	<< "num_peaks not 20";

	nc[device_id].nms_num_parts = nms_layer->GetNumParts();
	CHECK_LE(nc[device_id].nms_num_parts, MAX_NUM_PARTS)
		<< "num_parts in NMS layer (" << nc[device_id].nms_num_parts << ") "
		<< "too big ( MAX_NUM_PARTS )";

	if (nc[device_id].nms_num_parts==15) {
		nc[device_id].modeldesc = new MPIModelDescriptor();
		global.nms_threshold = nms_layer->GetThreshold();
		global.connect_min_subset_cnt = 3;
		global.connect_min_subset_score = 0.4;
		global.connect_inter_threshold = 0.01;
		global.connect_inter_min_above_threshold = 8;
		LOG(INFO) << "Selecting MPI model.";
	} else if (nc[device_id].nms_num_parts==18) {
		nc[device_id].modeldesc = new COCOModelDescriptor();
		global.nms_threshold = 0.055;
		global.connect_min_subset_cnt = 3;
		global.connect_min_subset_score = 0.40;
		global.connect_inter_threshold = 0.055;
		global.connect_inter_min_above_threshold = 9;
	} else {
		CHECK(0) << "Unknown number of parts! Couldn't set model";
	}
	model_descriptor = nc[device_id].modeldesc;

	//dry run
	LOG(INFO) << "Dry running...";
	nc[device_id].person_net->ForwardFrom(0);
	//nc[device_id].pose_net->ForwardFrom(0);
	LOG(INFO) << "Success.";
	cudaMalloc(&nc[device_id].canvas, origin_width * origin_height * 3 * sizeof(float));
	cudaMalloc(&nc[device_id].joints, MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float) );
}

void process_and_pad_image(float* target, Mat oriImg, int tw, int th, bool normalize){
	int ow = oriImg.cols;
	int oh = oriImg.rows;
	int offset2_target = tw * th;

	int padw = (tw-ow)/2;
	int padh = (th-oh)/2;
	//LOG(ERROR) << " padw " << padw << " padh " << padh;

	//parallel here
	unsigned char* pointer = (unsigned char*)(oriImg.data);

	for(int c = 0; c < 3; c++){
		for(int y = 0; y < th; y++){
			int oy = y - padh;
			for(int x = 0; x < tw; x++){
				int ox = x - padw;
				if(ox>=0 && ox < ow && oy>=0 && oy < oh ){
					if(normalize)
						target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c])/256.0f - 0.5f;
					else
						target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c]);
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

void render(int gid, float *heatmaps /*GPU*/) {
	// LOG(ERROR) << "begin render";

	//float* canvas = nc[gid].person_net->blobs()[3]->mutable_gpu_data(); //render layer
	//float* image_ref = nc[gid].person_net->blobs()[0]->mutable_gpu_data();
	//float* centers  = nc[gid].person_net->blobs()[nc[gid].nblob_person-1]->mutable_gpu_data();
	float* centers = 0;
		//float* poses    = nc[gid].pose_net->blobs()[nc[gid].nblob_pose-1]->mutable_gpu_data();
	float* poses    = nc[gid].joints;

	//LOG(ERROR) << "begin render_in_cuda";
	//LOG(ERROR) << "CPU part num" << global.part_to_show;
	double tic = get_wall_time();
	if (nc[gid].modeldesc->num_parts()==15) {
		caffe::render_mpi_parts(nc[gid].canvas, origin_width, origin_height, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT,
									   heatmaps, boxsize, centers, poses, nc[gid].num_people, global.part_to_show);
  } else if (nc[gid].modeldesc->num_parts()==18) {
		// caffe::render_coco_parts(nc[gid].canvas, origin_width, origin_height, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT,
									  //  heatmaps, boxsize, centers, poses, nc[gid].num_people, global.part_to_show);
		if (global.part_to_show-1<=nc[gid].modeldesc->num_parts())
		{
			caffe::render_coco_parts(nc[gid].canvas,
				origin_width, origin_height,
				INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT,
		 		heatmaps, boxsize, centers, poses,
				nc[gid].num_people, global.part_to_show, global.uistate.is_googly_eyes);
		} else {
			int aff_part = ((global.part_to_show-1)-nc[gid].modeldesc->num_parts()-1)*2;
			int num_parts_accum = 1;
			if (aff_part==0) {
				num_parts_accum = 19;
			} else {
				aff_part = aff_part-2;
			}
			aff_part += 1+nc[gid].modeldesc->num_parts();
			caffe::render_coco_aff(nc[gid].canvas, origin_width, origin_height, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT,
		 								 heatmaps, boxsize, centers, poses, nc[gid].num_people, aff_part, num_parts_accum);
		}
	}
	VLOG(2) << "Render time " << (get_wall_time()-tic)*1000.0 << " ms.";
}


void* getFrameFromCam(void *i){
		VideoCapture cap;
		double target_frame_time = 0;
		double target_frame_rate = 0;
		if (FLAGS_video.empty()) {
			CHECK(cap.open(FLAGS_camera)) << "Couldn't open camera " << FLAGS_camera;
			cap.set(CV_CAP_PROP_FRAME_WIDTH,camera_frame_width);
			cap.set(CV_CAP_PROP_FRAME_HEIGHT,camera_frame_height);
		} else {
			CHECK(cap.open(FLAGS_video)) << "Couldn't open video file " << FLAGS_video;
			target_frame_rate = cap.get(CV_CAP_PROP_FPS);
			target_frame_time = 1.0/target_frame_rate;
			if (FLAGS_start_frame) {
				cap.set(CV_CAP_PROP_POS_FRAMES, FLAGS_start_frame);
			}
			// global.uistate.is_video_paused = true;
		}

    int global_counter = 1;
		int frame_counter = 0;
		Mat image_uchar;
		Mat image_uchar_prev;
		double last_frame_time = -1;
    while(1) {
			if (global.quit_threads) break;
			if (!FLAGS_video.empty() && FLAGS_no_frame_drops) {
				// If the queue is too long, wait for a bit
				if (global.input_queue.size()>10) {
					usleep(10*1000.0);
					continue;
				}
			}
			cap >> image_uchar;
			// Keep a count of how many frames we've seen in the video
			if (!FLAGS_video.empty()) {
				if (global.uistate.seek_to_frame!=-1) {
					cap.set(CV_CAP_PROP_POS_FRAMES, global.uistate.current_frame);
					global.uistate.seek_to_frame = -1;
				}
				frame_counter = cap.get(CV_CAP_PROP_POS_FRAMES);

				// This should probably be protected.
				global.uistate.current_frame = frame_counter-1;
				if (global.uistate.is_video_paused) {
					cap.set(CV_CAP_PROP_POS_FRAMES, frame_counter-1);
					frame_counter -= 1;
				}

			// If we reach the end of a video, loop
				if (frame_counter >= cap.get(CV_CAP_PROP_FRAME_COUNT)) {
					LOG(INFO) << "Looping video after " << frame_counter-1 << " frames";
			    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
					// Wait until the queues are clear before exiting
					if (!FLAGS_write_frames.empty()) {
						while (global.input_queue.size() || global.output_queue_ordered.size()) {
							// Should actually wait until they finish writing to disk
							usleep(250*1000.0);
							continue;
						}
						global.quit_threads = true;
						global.uistate.is_video_paused = true;
					}
				}

				// Sleep to get the right frame rate.
				double cur_frame_time = get_wall_time();
				double interval = (cur_frame_time-last_frame_time);
				VLOG(3) << "cur_frame_time " << (cur_frame_time);
				VLOG(3) << "last_frame_time " << (last_frame_time);
				VLOG(3) << "cur-last_frame_time " << (cur_frame_time - last_frame_time);
				VLOG(3) << "Video target frametime " << 1.0/target_frame_time
								<< " read frametime " << 1.0/interval;
				if (interval<target_frame_time) {
					VLOG(3) << "Sleeping for " << (target_frame_time-interval)*1000.0;
					usleep((target_frame_time-interval)*1000.0*1000.0);
					cur_frame_time = get_wall_time();
				}
				last_frame_time = cur_frame_time;
			}	else {
				// From camera, just increase counter.
				if (global.uistate.is_video_paused) {
					image_uchar = image_uchar_prev;
				}
				image_uchar_prev = image_uchar;
				frame_counter++;
			}


		resize(image_uchar, image_uchar, Size(origin_width, origin_height), 0, 0, CV_INTER_AREA);

  		//char imgname[50];
		//sprintf(imgname, "../dome/%03d.jpg", global_counter);
		//sprintf(imgname, "../frame/frame%04d.png", global_counter);
		//image_uchar = imread(imgname, 1);
		//sprintf(imgname, "../Ian/hd%08d_00_00.png", global_counter*2 + 3500); //2700

		// sprintf(imgname, "../domeHD/hd%08d_00_07.png", global_counter + 2700); //2700
		// image_uchar = imread(imgname, 1);
		// resize(image_uchar, image_uchar, Size(960, 540), 0, 0, CV_INTER_AREA);
		// //LOG(ERROR) << "global_counter " << global_counter;

		// if(global_counter>=600){  //449 //250
		// 	imshow("here",image_uchar);
		// 	waitKey();
		// }

		//waitKey(10); //120 //400
		if( image_uchar.empty() ) continue;

		Frame f;
		f.index = global_counter++;
		f.video_frame_number = global.uistate.current_frame;
		f.data_for_wrap = new unsigned char [origin_height * origin_width * 3]; //fill after process
		f.data_for_mat = new float [origin_height * origin_width * 3];
		process_and_pad_image(f.data_for_mat, image_uchar, origin_width, origin_height, 0);

		//resize
		// int target_rows = fixed_scale_height; //some fixed number that is multiplier of 8
		// int target_cols = float(fixed_scale_height) / image_uchar.rows * image_uchar.cols;

		// if(target_cols % 8 != 0) {
		// 	target_cols = 8 * (target_cols / 8 + 1);
		// }


		// if(INIT_PERSON_NET_WIDTH != target_cols || INIT_PERSON_NET_HEIGHT != target_rows){
		// 	LOG(ERROR) << "Size not match: " << INIT_PERSON_NET_WIDTH << "  " << target_cols;
		// 	continue;
		// }

		//pad and transform to float
		int offset = 3 * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
		f.data = new float [batch_size * offset];
		int target_width, target_height;
		Mat image_temp;
		//LOG(ERROR) << "f.index: " << f.index;
		for(int i=0; i < batch_size; i++){
			float scale = start_scale - i*scale_gap;
			target_width = 16 * ceil(INIT_PERSON_NET_WIDTH * scale /16);
			target_height = 16 * ceil(INIT_PERSON_NET_HEIGHT * scale /16);
			//LOG(ERROR) << "target_size[0][0]: " << target_width << " target_size[0][1] " << target_height;

			// int padw, padh;
			// padw = (INIT_PERSON_NET_WIDTH - target_width)/2;
			// padh = (INIT_PERSON_NET_HEIGHT - target_height)/2;
			// LOG(ERROR) << "padw " << padw << " padh " << padh;
			CHECK_LE(target_width, INIT_PERSON_NET_WIDTH);
			CHECK_LE(target_height, INIT_PERSON_NET_HEIGHT);

			resize(image_uchar, image_temp, Size(target_width, target_height), 0, 0, CV_INTER_AREA);
			process_and_pad_image(f.data + i * offset, image_temp, INIT_PERSON_NET_WIDTH, INIT_PERSON_NET_HEIGHT, 1);
		}
		f.commit_time = get_wall_time();
		f.preprocessed_time = get_wall_time();

		// check the first channel
		// int tw = INIT_PERSON_NET_WIDTH;
		// int th = INIT_PERSON_NET_HEIGHT;
		// for(int i=0; i < batch_size; i++){
		// 	Mat test(th, tw, CV_8UC1);
		// 	for(int y = 0; y < th; y++){
		// 		for(int x = 0; x < tw; x++){
		// 			test.data[y * tw + x] = (unsigned int)((f.data[i * tw *th *3 + y * tw + x] + 0.5) * 256);
		// 		}
		// 	}
		// 	char imgname[50];
		// 	sprintf(imgname, "validate%02d.jpg", i);
		// 	cv::imwrite(imgname, test);
		// }

		global.input_queue.push(f);
		//LOG(ERROR) << "Frame " << f.index << " committed with init_time " << fixed << f.commit_time;
		//LOG(ERROR) << "pushing frame " << index << " to input_queue, now size " << global.input_queue.size();
		//printGlobal("prepareFrame    ");
		//if(counter == 3) break;
    }
    return nullptr;
}

int connectLimbs(
	vector< vector<double>> &subset,
	vector< vector< vector<double> > > &connection,
	const float *heatmap_pointer,
	const float *peaks,
	int max_peaks,
	float *joints,
	ModelDescriptor *modeldesc) {
		/* Parts Connection ---------------------------------------*/
		//limbSeq = [15 2; 2 1; 2 3; 3 4; 4 5; 2 6; 6 7; 7 8; 15 12; 12 13; 13 14; 15 9; 9 10; 10 11];
		//int limbSeq[28] = {14,1, 1,0, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 14,11, 11,12, 12,13, 14,8, 8,9, 9,10};
		//int mapIdx[14] = {27, 16, 17, 18, 19, 20, 21, 22, 15, 25, 26, 14, 23, 24};

		const int NUM_PARTS = modeldesc->num_parts();
		const int *limbSeq = modeldesc->get_limb_seq();
		const int *mapIdx = modeldesc->get_map_idx();
		const int num_limb_seq = modeldesc->num_limb_seq();

		int SUBSET_CNT = NUM_PARTS+2;
		int SUBSET_SCORE = NUM_PARTS+1;
		int SUBSET_SIZE = NUM_PARTS+3;

		CHECK_EQ(NUM_PARTS, 15);
		CHECK_EQ(num_limb_seq, 14);

		int peaks_offset = 3*(max_peaks+1);
		subset.clear();
		connection.clear();

		for(int k = 0; k < num_limb_seq; k++){
			//float* score_mid = heatmap_pointer + mapIdx[k] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
			const float* map_x = heatmap_pointer + mapIdx[2*k] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
			const float* map_y = heatmap_pointer + mapIdx[2*k+1] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;

			const float* candA = peaks + limbSeq[2*k]*peaks_offset;
			const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;
			//debug
			// for(int i = 0; i < 33; i++){
			//    	cout << candA[i] << " ";
			// }
			// cout << endl;
			// for(int i = 0; i < 33; i++){
			//    	cout << candB[i] << " ";
			// }
			// cout << endl;

			vector< vector<double> > connection_k;
			int nA = candA[0];
			int nB = candB[0];

			// add parts into the subset in special case
			if(nA ==0 && nB ==0){
				continue;
			}
			else if(nA ==0){
				for(int i = 1; i <= nB; i++){
					vector<double> row_vec(SUBSET_SIZE, 0);
					row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
					row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
					row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
					subset.push_back(row_vec);
				}
				continue;
			}
			else if(nB ==0){
				for(int i = 1; i <= nA; i++){
					vector<double> row_vec(SUBSET_SIZE, 0);
					row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
					row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
					row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
					subset.push_back(row_vec);
				}
				continue;
			}

			vector< vector<double>> temp;
			const int num_inter = 10;

			for(int i = 1; i <= nA; i++){
				for(int j = 1; j <= nB; j++){
					/*//midPoint = round((candA(i,1:2) + candB(j,1:2))/2);
					int mid_x = round((candA[i*3] + candB[j*3])/2);
					int mid_y = round((candA[i*3+1] + candB[j*3+1])/2);
					float dist = sqrt(pow((candA[i*3]-candB[j*3]),2)+pow((candA[i*3+1]-candB[j*3+1]),2));
					//float score = score_mid[ mid_y * INIT_PERSON_NET_WIDTH + mid_x] + std::min((150/dist-1),0.f);

					float sum = 0;
					int count = 0;
					for(int dh=-5; dh < 5; dh++){
					for(int dw=-5; dw < 5; dw++){
					int my = mid_y + dh;
					int mx = mid_x + dw;
					if(mx>=0 && mx < INIT_PERSON_NET_WIDTH && my>=0 && my < INIT_PERSON_NET_HEIGHT ){
					sum = sum + score_mid[ my * INIT_PERSON_NET_WIDTH + mx];
					count ++;
				}
			}
		}
		*/
		float s_x = candA[i*3];
		float s_y = candA[i*3+1];
		float d_x = candB[j*3] - candA[i*3];
		float d_y = candB[j*3+1] - candA[i*3+1];
		float norm_vec = sqrt( pow(d_x,2) + pow(d_y,2) );
		if (norm_vec<1e-6) {
			continue;
		}
		float vec_x = d_x/norm_vec;
		float vec_y = d_y/norm_vec;

		float sum = 0;
		int count = 0;

		for(int lm=0; lm < num_inter; lm++){
			int my = round(s_y + lm*d_y/num_inter);
			int mx = round(s_x + lm*d_x/num_inter);
			int idx = my * INIT_PERSON_NET_WIDTH + mx;
			float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
			if(score > global.connect_inter_threshold){
				sum = sum + score;
				count ++;
			}
		}
		//float score = sum / count; // + std::min((130/dist-1),0.f)

		if(count > global.connect_inter_min_above_threshold){//num_inter*0.8){ //thre/2
			// parts score + cpnnection score
			vector<double> row_vec(4, 0);
			row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
			row_vec[2] = sum/count;
			row_vec[0] = i;
			row_vec[1] = j;
			temp.push_back(row_vec);
		}
	}
}

//** select the top num connection, assuming that each part occur only once
// sort rows in descending order based on parts + connection score
if(temp.size() > 0)
sort(temp.begin(), temp.end(), ColumnCompare());

int num = min(nA, nB);
int cnt = 0;
vector<int> occurA(nA, 0);
vector<int> occurB(nB, 0);

//debug
// 	if(k==3){
//  cout << "connection before" << endl;
// 	for(int i = 0; i < temp.size(); i++){
// 	   	for(int j = 0; j < temp[0].size(); j++){
// 	        cout << temp[i][j] << " ";
// 	    }
// 	    cout << endl;
// 	}
// 	//cout << "debug" << score_mid[ 216 * INIT_PERSON_NET_WIDTH + 184] << endl;
// }

//cout << num << endl;
for(int row =0; row < temp.size(); row++){
	if(cnt==num){
		break;
	}
	else{
		int i = int(temp[row][0]);
		int j = int(temp[row][1]);
		float score = temp[row][2];
		if ( occurA[i-1] == 0 && occurB[j-1] == 0 ){ // && score> (1+thre)
			vector<double> row_vec(3, 0);
			row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
			row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
			row_vec[2] = score;
			connection_k.push_back(row_vec);
			cnt = cnt+1;
			//cout << "cnt: " << connection_k.size() << endl;
			occurA[i-1] = 1;
			occurB[j-1] = 1;
		}
	}
}
//   if(k==0){
//     cout << "connection" << endl;
//     for(int i = 0; i < connection_k.size(); i++){
// 	   	for(int j = 0; j < connection_k[0].size(); j++){
// 	        cout << connection_k[i][j] << " ";
// 	    }
// 	    cout << endl;
// 	}
// }
//connection.push_back(connection_k);


//** cluster all the joints candidates into subset based on the part connection
// initialize first body part connection 15&16
//cout << connection_k.size() << endl;
if(k==0){
	vector<double> row_vec(NUM_PARTS+3, 0);
	for(int i = 0; i < connection_k.size(); i++){
		double indexA = connection_k[i][0];
		double indexB = connection_k[i][1];
		row_vec[limbSeq[0]] = indexA;
		row_vec[limbSeq[1]] = indexB;
		row_vec[SUBSET_CNT] = 2;
		// add the score of parts and the connection
		row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
		subset.push_back(row_vec);
	}
}
else{
	if(connection_k.size()==0){
		continue;
	}
	// A is already in the subset, find its connection B
	for(int i = 0; i < connection_k.size(); i++){
		int num = 0;
		double indexA = connection_k[i][0];
		double indexB = connection_k[i][1];

		for(int j = 0; j < subset.size(); j++){
			if(subset[j][limbSeq[2*k]] == indexA){
				subset[j][limbSeq[2*k+1]] = indexB;
				num = num+1;
				subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
				subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
			}
		}
		// if can not find partA in the subset, create a new subset
		if(num==0){
			vector<double> row_vec(SUBSET_SIZE, 0);
			row_vec[limbSeq[2*k]] = indexA;
			row_vec[limbSeq[2*k+1]] = indexB;
			row_vec[SUBSET_CNT] = 2;
			row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
			subset.push_back(row_vec);
		}
	}
}
//cout << nA << " ";
}

//debug
// cout << " subset " << endl;
// for(int i = 0; i < subset.size(); i++){
//    	for(int j = 0; j < subset[0].size(); j++){
//         cout << subset[i][j] << " ";
//     }
//     cout << endl;
// }

//** joints by deleteing some rows of subset which has few parts occur
//cout << " joints " << endl;
int cnt = 0;
for(int i = 0; i < subset.size(); i++){
	//cout << "score: " << i << " " << subset[i][16]/subset[i][17];
	if (subset[i][SUBSET_CNT]>=global.connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>global.connect_min_subset_score){
		for(int j = 0; j < NUM_PARTS; j++){
			int idx = int(subset[i][j]);
			if(idx){
				joints[cnt*NUM_PARTS*3 + j*3 +2] = peaks[idx];
				joints[cnt*NUM_PARTS*3 + j*3 +1] = peaks[idx-1]* origin_height/ (float)INIT_PERSON_NET_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
				joints[cnt*NUM_PARTS*3 + j*3] = peaks[idx-2]* origin_width/ (float)INIT_PERSON_NET_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
				//cout << peaks[idx-2] << " " << peaks[idx-1] << " " << peaks[idx] << endl;
			}
			else{
				joints[cnt*NUM_PARTS*3 + j*3 +2] = 0;
				joints[cnt*NUM_PARTS*3 + j*3 +1] = 0;
				joints[cnt*NUM_PARTS*3 + j*3] = 0;
				//cout << 0 << " " << 0 << " " << 0 << endl;
			}
			//cout << joints[cnt*45 + j*3] << " " << joints[cnt*45 + j*3 +1] << " " << joints[cnt*45 + j*3 +2] << endl;
		}
		cnt++;
		if (cnt==MAX_PEOPLE) break;
	}
	//cout << endl;
}
return cnt;
}
int distanceThresholdPeaks(const float *in_peaks, int max_peaks,
	float *peaks, ModelDescriptor *modeldesc) {
	// Post-process peaks to remove those which are within sqrt(dist_threshold2)
	// of each other.

	const int NUM_PARTS = modeldesc->num_parts();
	const float dist_threshold2 = 6*6;
	int peaks_offset = 3*(max_peaks+1);

	int total_peaks = 0;
	for(int p = 0; p < NUM_PARTS; p++){
		const float *pipeaks = in_peaks + p*peaks_offset;
		float *popeaks = peaks + p*peaks_offset;
		int num_in_peaks = int(pipeaks[0]);
		int num_out_peaks = 0; // Actual number of peak count
		for (int c1=0;c1<num_in_peaks;c1++) {
			float x1 = pipeaks[(c1+1)*3+0];
			float y1 = pipeaks[(c1+1)*3+1];
			float s1 = pipeaks[(c1+1)*3+2];
			bool keep = true;
			for (int c2=0;c2<num_out_peaks;c2++) {
				float x2 = popeaks[(c2+1)*3+0];
				float y2 = popeaks[(c2+1)*3+1];
				float s2 = popeaks[(c2+1)*3+2];
				float dist2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
				if (dist2<dist_threshold2) {
					// This peak is too close to a peak already in the output buffer
					// so don't add it.
					keep = false;
					if (s1>s2) {
						// It's better than the one in the output buffer
						// so we swap it.
						popeaks[(c2+1)*3+0] = x1;
						popeaks[(c2+1)*3+1] = y1;
						popeaks[(c2+1)*3+2] = s1;
					}
				}
			}
			if (keep && num_out_peaks<max_peaks) {
				// We don't already have a better peak within the threshold distance
				popeaks[(num_out_peaks+1)*3+0] = x1;
				popeaks[(num_out_peaks+1)*3+1] = y1;
				popeaks[(num_out_peaks+1)*3+2] = s1;
				num_out_peaks++;
			}
		}
		// if (num_in_peaks!=num_out_peaks) {
			//LOG(INFO) << "Part: " << p << " in peaks: "<< num_in_peaks << " out: " << num_out_peaks;
		// }
		popeaks[0] = float(num_out_peaks);
		total_peaks += num_out_peaks;
	}
	return total_peaks;
}

int connectLimbsCOCO(
	vector< vector<double>> &subset,
	vector< vector< vector<double> > > &connection,
	const float *heatmap_pointer,
	const float *in_peaks,
	int max_peaks,
	float *joints,
	ModelDescriptor *modeldesc) {
		/* Parts Connection ---------------------------------------*/
		const int NUM_PARTS = modeldesc->num_parts();
		const int *limbSeq = modeldesc->get_limb_seq();
		const int *mapIdx = modeldesc->get_map_idx();
		const int num_limb_seq = modeldesc->num_limb_seq();

		CHECK_EQ(NUM_PARTS, 18) << "Wrong connection function for model";
		CHECK_EQ(num_limb_seq, 19) << "Wrong connection function for model";

		int SUBSET_CNT = NUM_PARTS+2;
		int SUBSET_SCORE = NUM_PARTS+1;
		int SUBSET_SIZE = NUM_PARTS+3;

		const int peaks_offset = 3*(max_peaks+1);

		const float *peaks = in_peaks;
		//float peaks[(MAX_MAX_PEAKS+1)*MAX_NUM_PARTS*3]={0};
		//distanceThresholdPeaks(in_peaks, max_peaks, peaks, modeldesc);
		subset.clear();
		connection.clear();

		for(int k = 0; k < num_limb_seq; k++){
			//float* score_mid = heatmap_pointer + mapIdx[k] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
			const float* map_x = heatmap_pointer + mapIdx[2*k] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
			const float* map_y = heatmap_pointer + mapIdx[2*k+1] * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;

			const float* candA = peaks + limbSeq[2*k]*peaks_offset;
			const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;
			//debug
			// for(int i = 0; i < 33; i++){
			//    	cout << candA[i] << " ";
			// }
			// cout << endl;
			// for(int i = 0; i < 33; i++){
			//    	cout << candB[i] << " ";
			// }
			// cout << endl;

			vector< vector<double> > connection_k;
			int nA = candA[0];
			int nB = candB[0];

			// add parts into the subset in special case
			if(nA ==0 && nB ==0){
				continue;
			} else if(nA ==0){
				for(int i = 1; i <= nB; i++){
					int num = 0;
					int indexB = limbSeq[2*k+1];
					for(int j = 0; j < subset.size(); j++){
							int off = limbSeq[2*k+1]*peaks_offset + i*3 + 2;
							if (subset[j][indexB] == off) {
									num = num+1;
									continue;
							}
					}
					if (num!=0) {
						//LOG(INFO) << " else if(nA==0) shouldn't have any nB already assigned?";
					} else {
						vector<double> row_vec(SUBSET_SIZE, 0);
						row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
						row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
						row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
						subset.push_back(row_vec);
					}
					//LOG(INFO) << "nA==0 New subset on part " << k << " subsets: " << subset.size();
				}
				continue;
			} else if(nB ==0){
				for(int i = 1; i <= nA; i++){
					int num = 0;
					int indexA = limbSeq[2*k];
					for(int j = 0; j < subset.size(); j++){
							int off = limbSeq[2*k]*peaks_offset + i*3 + 2;
							if (subset[j][indexA] == off) {
									num = num+1;
									continue;
							}
					}
					if (num==0) {
						vector<double> row_vec(SUBSET_SIZE, 0);
						row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
						row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
						row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
						subset.push_back(row_vec);
						//LOG(INFO) << "nB==0 New subset on part " << k << " subsets: " << subset.size();
					} else {
						//LOG(INFO) << "nB==0 discarded would have added";
					}
				}
				continue;
			}

			vector< vector<double>> temp;
			const int num_inter = 10;

			for(int i = 1; i <= nA; i++){
				for(int j = 1; j <= nB; j++){
					float s_x = candA[i*3];
					float s_y = candA[i*3+1];
					float d_x = candB[j*3] - candA[i*3];
					float d_y = candB[j*3+1] - candA[i*3+1];
					float norm_vec = sqrt( d_x*d_x + d_y*d_y );
					if (norm_vec<1e-6) {
						// The peaks are coincident. Don't connect them.
						continue;
					}
					float vec_x = d_x/norm_vec;
					float vec_y = d_y/norm_vec;

					float sum = 0;
					int count = 0;

					for(int lm=0; lm < num_inter; lm++){
						int my = round(s_y + lm*d_y/num_inter);
						int mx = round(s_x + lm*d_x/num_inter);
						int idx = my * INIT_PERSON_NET_WIDTH + mx;
						float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
						if(score > global.connect_inter_threshold){
							sum = sum + score;
							count ++;
						}
					}
					//float score = sum / count; // + std::min((130/dist-1),0.f)

					if(count > global.connect_inter_min_above_threshold){//num_inter*0.8){ //thre/2
						// parts score + cpnnection score
						vector<double> row_vec(4, 0);
						row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
						row_vec[2] = sum/count;
						row_vec[0] = i;
						row_vec[1] = j;
						temp.push_back(row_vec);
					}
				}
			}

			//** select the top num connection, assuming that each part occur only once
			// sort rows in descending order based on parts + connection score
			if(temp.size() > 0)
			sort(temp.begin(), temp.end(), ColumnCompare());

			int num = min(nA, nB);
			int cnt = 0;
			vector<int> occurA(nA, 0);
			vector<int> occurB(nB, 0);

			//debug
			// 	if(k==3){
			//  cout << "connection before" << endl;
			// 	for(int i = 0; i < temp.size(); i++){
			// 	   	for(int j = 0; j < temp[0].size(); j++){
			// 	        cout << temp[i][j] << " ";
			// 	    }
			// 	    cout << endl;
			// 	}
			// 	//cout << "debug" << score_mid[ 216 * INIT_PERSON_NET_WIDTH + 184] << endl;
			// }

			//cout << num << endl;
			for(int row =0; row < temp.size(); row++){
				if(cnt==num){
					break;
				}
				else{
					int i = int(temp[row][0]);
					int j = int(temp[row][1]);
					float score = temp[row][2];
					if ( occurA[i-1] == 0 && occurB[j-1] == 0 ){ // && score> (1+thre)
						vector<double> row_vec(3, 0);
						row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
						row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
						row_vec[2] = score;
						connection_k.push_back(row_vec);
						cnt = cnt+1;
						//cout << "cnt: " << connection_k.size() << endl;
						occurA[i-1] = 1;
						occurB[j-1] = 1;
					}
				}
			}
			//   if(k==0){
			//     cout << "connection" << endl;
			//     for(int i = 0; i < connection_k.size(); i++){
			// 	   	for(int j = 0; j < connection_k[0].size(); j++){
			// 	        cout << connection_k[i][j] << " ";
			// 	    }
			// 	    cout << endl;
			// 	}
			// }
			//connection.push_back(connection_k);


			//** cluster all the joints candidates into subset based on the part connection
			// initialize first body part connection 15&16
			//cout << connection_k.size() << endl;
			if(k==0){
				vector<double> row_vec(NUM_PARTS+3, 0);
				for(int i = 0; i < connection_k.size(); i++){
					double indexB = connection_k[i][1];
					double indexA = connection_k[i][0];
					row_vec[limbSeq[0]] = indexA;
					row_vec[limbSeq[1]] = indexB;
					row_vec[SUBSET_CNT] = 2;
					// add the score of parts and the connection
					row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
					//LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
					subset.push_back(row_vec);
				}
			}/* else if(k==17 || k==18){ // TODO: Check k numbers?
				//   %add 15 16 connection
				for(int i = 0; i < connection_k.size(); i++){
					double indexA = connection_k[i][0];
					double indexB = connection_k[i][1];

					for(int j = 0; j < subset.size(); j++){
					// if subset(j, indexA) == partA(i) && subset(j, indexB) == 0
					// 		subset(j, indexB) = partB(i);
					// elseif subset(j, indexB) == partB(i) && subset(j, indexA) == 0
					// 		subset(j, indexA) = partA(i);
					// end
						if(subset[j][limbSeq[2*k]] == indexA && subset[j][limbSeq[2*k+1]]==0){
							subset[j][limbSeq[2*k+1]] = indexB;
						} else if (subset[j][limbSeq[2*k+1]] == indexB && subset[j][limbSeq[2*k]]==0){
							subset[j][limbSeq[2*k]] = indexA;
						}
				}
				continue;
			}
		}*/ else{
			if(connection_k.size()==0){
				continue;
			}
// A is already in the subset, find its connection B
for(int i = 0; i < connection_k.size(); i++){
	int num = 0;
	double indexA = connection_k[i][0];
	double indexB = connection_k[i][1];

	for(int j = 0; j < subset.size(); j++){
		if(subset[j][limbSeq[2*k]] == indexA){
			subset[j][limbSeq[2*k+1]] = indexB;
			num = num+1;
			subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
			subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
		}
	}
	// if can not find partA in the subset, create a new subset
	if(num==0){
		//LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
		vector<double> row_vec(SUBSET_SIZE, 0);
		row_vec[limbSeq[2*k]] = indexA;
		row_vec[limbSeq[2*k+1]] = indexB;
		row_vec[SUBSET_CNT] = 2;
		row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
		subset.push_back(row_vec);
	}
}
}
//cout << nA << " ";
}

//debug
// cout << " subset " << endl;
// for(int i = 0; i < subset.size(); i++){
//    	for(int j = 0; j < subset[0].size(); j++){
//         cout << subset[i][j] << " ";
//     }
//     cout << endl;
// }

//** joints by deleteing some rows of subset which has few parts occur
//cout << " joints " << endl;
int cnt = 0;
for(int i = 0; i < subset.size(); i++){
	//cout << "score: " << i << " " << subset[i][16]/subset[i][17];
	if (subset[i][SUBSET_CNT]<1) {
		LOG(INFO) << "BAD SUBSET_CNT";
	}
	if (subset[i][SUBSET_CNT]>=global.connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>global.connect_min_subset_score){
		for(int j = 0; j < NUM_PARTS; j++){
			int idx = int(subset[i][j]);
			if(idx){
				joints[cnt*NUM_PARTS*3 + j*3 +2] = peaks[idx];
				joints[cnt*NUM_PARTS*3 + j*3 +1] = peaks[idx-1]* origin_height/ (float)INIT_PERSON_NET_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
				joints[cnt*NUM_PARTS*3 + j*3] = peaks[idx-2]* origin_width/ (float)INIT_PERSON_NET_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
				//cout << peaks[idx-2] << " " << peaks[idx-1] << " " << peaks[idx] << endl;
			}
			else{
				joints[cnt*NUM_PARTS*3 + j*3 +2] = 0;
				joints[cnt*NUM_PARTS*3 + j*3 +1] = 0;
				joints[cnt*NUM_PARTS*3 + j*3] = 0;
				//cout << 0 << " " << 0 << " " << 0 << endl;
			}
			//cout << joints[cnt*45 + j*3] << " " << joints[cnt*45 + j*3 +1] << " " << joints[cnt*45 + j*3 +2] << endl;
		}
		cnt++;
		if (cnt==MAX_PEOPLE) break;
	}
	//cout << endl;
}
return cnt;
}


void* processFrame(void *i){
	int tid = *((int *) i);
	warmup(tid);
	LOG(INFO) << "GPU " << tid << " is ready";
	Frame f;

	int offset = INIT_PERSON_NET_WIDTH * INIT_PERSON_NET_HEIGHT * 3;
	//bool empty = false;

	Frame frame_batch[batch_size];

	vector< vector<double>> subset;
	vector< vector< vector<double> > > connection;

	const boost::shared_ptr< caffe::Blob< float > > heatmap_blob = nc[tid].person_net->blob_by_name("resized_map");
	const boost::shared_ptr< caffe::Blob< float > > joints_blob = nc[tid].person_net->blob_by_name("joints");

	caffe::NmsLayer<float> *nms_layer =
		(caffe::NmsLayer<float>*)nc[tid].person_net->layer_by_name("nms").get();

	//while(!empty){
	while(1){
		if (global.quit_threads) break;

		//LOG(ERROR) << "start";
		int valid_data = 0;
		//for(int n = 0; n < batch_size; n++){
		while(valid_data<1){
			if(global.input_queue.try_pop(&f)) {
				//consider dropping it
				f.gpu_fetched_time = get_wall_time();
				double elaspsed_time = f.gpu_fetched_time - f.commit_time;
				//LOG(ERROR) << "frame " << f.index << " is copied to GPU after " << elaspsed_time << " sec";
				if(elaspsed_time > 0.1
				   && !FLAGS_no_frame_drops) {//0.1*batch_size){ //0.1*batch_size
					//drop frame
					VLOG(1) << "skip frame " << f.index;
					delete [] f.data;
					delete [] f.data_for_mat;
					delete [] f.data_for_wrap;
					//n--;
					global.mtx.lock();
					global.dropped_index.push(f.index);
					global.mtx.unlock();
					continue;
				}
				//double tic1  = get_wall_time();

				cudaMemcpy(nc[tid].canvas, f.data_for_mat, origin_width * origin_height * 3 * sizeof(float), cudaMemcpyHostToDevice);

				frame_batch[0] = f;
				//LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
				float* pointer = nc[tid].person_net->blobs()[0]->mutable_gpu_data();

				cudaMemcpy(pointer + 0 * offset, frame_batch[0].data, batch_size * offset * sizeof(float), cudaMemcpyHostToDevice);
				valid_data++;
				//VLOG(2) << "Host->device " << (get_wall_time()-tic1)*1000.0 << " ms.";
			}
			else {
				//empty = true;
				break;
			}
		}
		if(valid_data == 0) continue;

		if (global.uistate.select_stage>=0) {
			caffe::SwitchLayer<float> *layer_ptr =
				(caffe::SwitchLayer<float>*)nc[tid].person_net->layer_by_name("Switch_L1").get();
				if (layer_ptr!=NULL) {
					LOG(INFO) << "Selecting stage " << global.uistate.select_stage;
					layer_ptr->SelectSwitch(global.uistate.select_stage);
					layer_ptr->switch_select_ = global.uistate.select_stage;
				}
				(caffe::SwitchLayer<float>*)nc[tid].person_net->layer_by_name("Switch_L2").get();
				if (layer_ptr!=NULL) {
					LOG(INFO) << "Selecting stage " << global.uistate.select_stage;
					layer_ptr->SelectSwitch(global.uistate.select_stage);
					layer_ptr->switch_select_ = global.uistate.select_stage;
				}
		}
		//timer.Stop();
		//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		//timer.Start();
		//LOG(ERROR) << "GPU " << tid << ": Running forward person_net";
		//nc.person_net->ForwardFromTo(0,nlayer-1);

		nms_layer->SetThreshold(global.nms_threshold);
		nc[tid].person_net->ForwardFrom(0);
		VLOG(2) << "CNN time " << (get_wall_time()-f.gpu_fetched_time)*1000.0 << " ms.";
				//cudaDeviceSynchronize();
		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";
		float* heatmap_pointer = heatmap_blob->mutable_cpu_data();
		const float* peaks = joints_blob->mutable_cpu_data();

		float joints[MAX_NUM_PARTS*3*MAX_PEOPLE]; //10*15*3

		int cnt = 0;
		// CHECK_EQ(nc[tid].nms_num_parts, 15);
		double tic = get_wall_time();
		if (nc[tid].nms_num_parts==15) {
			cnt = connectLimbs(subset, connection,
												 heatmap_pointer, peaks,
												 nc[tid].nms_max_peaks, joints, nc[tid].modeldesc);
    } else {
			cnt = connectLimbsCOCO(subset, connection,
												 heatmap_pointer, peaks,
												 nc[tid].nms_max_peaks, joints, nc[tid].modeldesc);
		}

		VLOG(2) << "CNT: " << cnt << " Connect time " << (get_wall_time()-tic)*1000.0 << " ms.";

		/* debug ---------------------------------------*/
		// //vector<int> bottom_shape = nc[tid].person_net->blobs()[nc[tid].nblob_person-3]->shape();
		// //cout << bottom_shape[0] << " " << bottom_shape[1] << " " << bottom_shape[2] << " " << bottom_shape[3] << " " << endl;

		// //float* heatmap_ptr = nc[tid].person_net->blobs()[nc[tid].nblob_person-3]->mutable_cpu_data();
		// //Mat heatmap(INIT_PERSON_NET_HEIGHT/8, INIT_PERSON_NET_WIDTH/8, CV_8UC1);
		// // for(int c = 0; c < 60; c++){
		// // 	float* jointsmap = heatmap_ptr + c * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH / 64;
		// // 	if(c==14)
		// // 		float* jointsmap = heatmap_ptr + 28 * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH / 64;

		// // 	for (int y = 0; y < INIT_PERSON_NET_HEIGHT/8; y++){
		// // 		for (int x = 0; x < INIT_PERSON_NET_WIDTH/8; x++){
		// // 			float num = jointsmap[ y * INIT_PERSON_NET_WIDTH/8 + x]; //0 ~ 1;
		// // 			num = (num > 1 ? 1 : (num < 0 ? 0 : num)); //prevent overflow for uchar
		// // 			heatmap.data[(y * INIT_PERSON_NET_WIDTH/8 + x)] = (unsigned char)(num * 255);
		// // 		}
		// // 	}

		// Mat heatmap(INIT_PERSON_NET_HEIGHT, INIT_PERSON_NET_WIDTH, CV_8UC1);
		// char filename [100];
		// for(int c = 0; c < 15; c++){  //30
		// 	float* jointsmap = heatmap_pointer + c * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;
		// 	if(c==14)
		// 		float* jointsmap = heatmap_pointer + 28 * INIT_PERSON_NET_HEIGHT * INIT_PERSON_NET_WIDTH;

		//     sprintf(filename, "map%02d.txt", c);
		//     ofstream fout(filename);

		// 	for (int y = 0; y < INIT_PERSON_NET_HEIGHT; y++){
		// 		for (int x = 0; x < INIT_PERSON_NET_WIDTH; x++){
		// 			float num = jointsmap[ y * INIT_PERSON_NET_WIDTH + x]; //0 ~ 1;
		// 			fout << num << "\t";
		// 			num = (num > 1 ? 1 : (num < 0 ? 0 : num)); //prevent overflow for uchar
		// 			heatmap.data[(y * INIT_PERSON_NET_WIDTH + x)] = (unsigned char)(num * 255);
		// 		}
		// 		fout<<endl;
		// 	}
		// 	fout.close();

		// 	for(int i = 0; i < 33; i++){
		// 		cout << peaks[c*33+i] << " ";
		// 	}
		// 	cout << endl;

		// 	for(int i = 0; i < peaks[c*33]; i++){
		// 		circle(heatmap, Point2f(peaks[c*33+3*(i+1)], peaks[c*33+3*(i+1)+1]), 2, Scalar(0,0,0), -1);
		// 		//cout << jointsmap[ int(peaks[c*33+3*(i+1)+1] * INIT_PERSON_NET_WIDTH + peaks[c*33+3*(i+1)]) ] << " ";
		// 	}

		// 	//cout << "here!" << endl;
		// 	//imshow("person_map", heatmap);
		// 	//waitKey();
		// 	char imgname[50];
		// 	sprintf(imgname, "map%02d.jpg", c);
		// 	cv::imwrite(imgname, heatmap);
		// }

		nc[tid].num_people[0] = cnt;
		VLOG(2) << "num_people[i] = " << cnt;


		cudaMemcpy(nc[tid].joints, joints,
			MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float),
			cudaMemcpyHostToDevice);

		// debug, wrong!
		// float *h_out = (float *) malloc(450 * sizeof(float) );
		// cudaMemcpy(h_out, nc[tid].joints, 450 * sizeof(float) , cudaMemcpyDeviceToHost);
		// for(int i = 0; i < cnt; i++){
		//     for(int j = 0; j < 15; j++){
		// 	    cout << h_out[i*45 + j*3] << " " << h_out[cnt*45 + j*3 +1] << " " << h_out[cnt*45 + j*3 +2] << endl;
		// 	}
		//     cout << endl;
		// }

		if(subset.size() != 0){
			// timer.Start();
			//LOG(ERROR) << "Rendering";
			render(tid, heatmap_pointer); //only support batch size = 1!!!!
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
			render(tid, heatmap_pointer);
			//frame_batch[n].data should revert to 0-255
			for(int n = 0; n < valid_data; n++){
				frame_batch[n].numPeople = 0;
				frame_batch[n].gpu_computed_time = get_wall_time();
				cudaMemcpy(frame_batch[n].data_for_mat, nc[tid].canvas, origin_height * origin_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
				global.output_queue.push(frame_batch[n]);
			}
		}

		// //copy data to pose_net
		// timer.Start();
		//LOG(ERROR) << "GPU " << tid << ": copy to posenet and reshape";

		//copy_to_posenet_and_reshape(tid);

		// timer.Stop();
		// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		// if(nc[tid].total_num_people != 0){
		// 	nc[tid].pose_net->ForwardFrom(0);
		// 	render(tid); //only support batch size = 1!!!!

		// 	for(int n = 0; n < valid_data; n++){
		// 		frame_batch[n].numPeople = nc[tid].num_people[n];
		// 		frame_batch[n].gpu_computed_time = get_wall_time();
		// 		cudaMemcpy(frame_batch[n].data_for_mat, nc[tid].canvas, origin_height * origin_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		// 		global.output_queue.push(frame_batch[n]);
		// 		//LOG(ERROR) << "Pushing data " << index_array[n] << " to output_queue, now size " << global.output_queue.size();
		// 		//printGlobal("processFrame   ");
		// 	}
		// }
		// else {
		// 	render(tid);
		// 	//frame_batch[n].data should revert to 0-255
		// 	for(int n = 0; n < valid_data; n++){
		// 		frame_batch[n].numPeople = 0;
		// 		frame_batch[n].gpu_computed_time = get_wall_time();
		// 		cudaMemcpy(frame_batch[n].data_for_mat, nc[tid].canvas, origin_height * origin_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		// 		global.output_queue.push(frame_batch[n]);
		// 	}
		// }
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
		if (global.quit_threads) break;
		bool success = global.output_queue_mated.try_pop(&f);
		f.buffer_start_time = get_wall_time();
		if(success){
			VLOG(4) << "buffer getting " << f.index << ", waiting for " << frame_waited;
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
		if (global.quit_threads) break;

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
	int counter = 1;
	double last_time = get_wall_time();
  double this_time;
  float FPS = 0;
	char tmp_str[256];
	while(1) {
		if (global.quit_threads) break;

		f = global.output_queue_ordered.pop();
		double tic = get_wall_time();
		Mat wrap_frame(origin_height, origin_width, CV_8UC3, f.data_for_wrap);

		if (FLAGS_write_frames.empty()) {
			snprintf(tmp_str, 256, "%4.1f fps", FPS);
		} else {
			snprintf(tmp_str, 256, "%4.2f s/gpu", FLAGS_num_gpu*1.0/FPS);
		}
		if (1) {
		cv::putText(wrap_frame, tmp_str, cv::Point(25,35),
			cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);

		snprintf(tmp_str, 256, "%4d", f.numPeople);
		cv::putText(wrap_frame, tmp_str, cv::Point(origin_width-100+2, 35+2),
			cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
		cv::putText(wrap_frame, tmp_str, cv::Point(origin_width-100, 35),
			cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
		}
		if (global.part_to_show!=0) {
			if (global.part_to_show-1<=model_descriptor->num_parts()) {
				snprintf(tmp_str, 256, "%10s", model_descriptor->get_part_name(global.part_to_show-1).c_str());
			} else {
				int aff_part = ((global.part_to_show-1)-model_descriptor->num_parts()-1)*2;
				if (aff_part==0) {
					snprintf(tmp_str, 256, "%10s", "PAFs");
				} else {
					aff_part = aff_part-2;
					aff_part += 1+model_descriptor->num_parts();
					std::string uvname = model_descriptor->get_part_name(aff_part);
					std::string conn = uvname.substr(0, uvname.find("("));
					snprintf(tmp_str, 256, "%10s", conn.c_str());
				}
			}
			cv::putText(wrap_frame, tmp_str, cv::Point(origin_width-175+1, 55+1),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
		}
		if (!FLAGS_video.empty() && FLAGS_write_frames.empty()) {
			snprintf(tmp_str, 256, "Frame %6d", global.uistate.current_frame);
			// cv::putText(wrap_frame, tmp_str, cv::Point(27,37),
			// 	cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);
			cv::putText(wrap_frame, tmp_str, cv::Point(25,55),
				cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,255,255), 1);
		}

		imshow("video", wrap_frame);
		if (!FLAGS_write_frames.empty()) {
			double a = get_wall_time();
			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
			compression_params.push_back(98);
			char fname[256];
			sprintf(fname, "%s%06d.jpg", FLAGS_write_frames.c_str(), f.video_frame_number);
			cv::imwrite(fname, wrap_frame, compression_params);
			last_time += get_wall_time()-a;
		}

		counter++;

		if(counter % 30 == 0){
			this_time = get_wall_time();
			FPS = 30.0f / (this_time - last_time);
			global.uistate.fps = FPS;
				//LOG(ERROR) << frame.cols << "  " << frame.rows;
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
			LOG(INFO) << msg;
		}

		delete [] f.data_for_mat;
		delete [] f.data_for_wrap;
		delete [] f.data;

		//LOG(ERROR) << msg;
		int key = waitKey(1);
		if (!handleKey(key)) {
			// TODO: sync issues?
			break;
		}

		VLOG(2) << "Display time " << (get_wall_time()-tic)*1000.0 << " ms.";
		//LOG(ERROR) << "showed_and_waited";

		// char filename[256];
		// sprintf(filename, "../result/frame%04d.jpg", f.index); //counter
		// imwrite(filename, wrap_frame);
		// LOG(ERROR) << "Saved output " << counter;
	}
	return nullptr;
}

int rtcpm() {
	caffe::CPUTimer timer;
	timer.Start();

	pthread_t gpu_threads_pool[NUM_GPU];
	for(int gpu = 0; gpu < NUM_GPU; gpu++){
		int *arg = new int[1];
		*arg = gpu+FLAGS_start_device;
		int rc = pthread_create(&gpu_threads_pool[gpu], NULL, processFrame, (void *) arg);
		if(rc){
			LOG(ERROR) << "Error:unable to create thread," << rc << "\n";
			exit(-1);
	    }
	}
	LOG(ERROR) << "Finish spawning " << NUM_GPU << " threads. now waiting." << "\n";


	usleep(10 * 1e6);
	if (FLAGS_fullscreen) {
		cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::resizeWindow("video", 1920, 1080);
		cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		global.uistate.is_fullscreen = true;
	} else {
		cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
		cv::resizeWindow("video", origin_width, origin_height);
		global.uistate.is_fullscreen = false;
	}
	/* push all frames for now (single thread)
	for(int i = 1; i <= 449; i++){
	 	char name[256];
	 	sprintf(name, "../frame/frame%04d.png", i);
	 	string name_to_push(name);
	 	global.frame_file_queue.push(make_pair(i,name_to_push));
	}
	LOG(ERROR) << "Finished pushing all frames" << "\n"; */


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
	VLOG(3) << "Finish spawning " << thread_pool_size << " threads. now waiting." << "\n";


	/* threads handling outputs */
	int thread_pool_size_out = NUM_GPU;
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
	VLOG(3) << "Finish spawning " << thread_pool_size_out << " threads. now waiting." << "\n";

	/* thread for buffer and ordering frame */
	pthread_t threads_order;
	int *arg = new int[1];
	int rc = pthread_create(&threads_order, NULL, buffer_and_order, (void *) arg);
	if(rc){
		LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
		exit(-1);
    }
	VLOG(3) << "Finish spawning the thread for ordering. now waiting." << "\n";

	/* display */
	pthread_t thread_display;
	rc = pthread_create(&thread_display, NULL, displayFrame, (void *) arg);
	if(rc){
		LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
		exit(-1);
    }
	VLOG(3) << "Finish spawning the thread for display. now waiting." << "\n";

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

// int main(int argc, char** argv) {
// 	NUM_GPU = atoi(argv[1]); // 4
// 	batch_size = atoi(argv[2]); //1

int main(int argc, char *argv[]) {
	::google::InitGoogleLogging("rtcpm");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	// Parse requested resolution string
	{
		int nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d",
											 &origin_width, &origin_height);
		CHECK_EQ(nRead,2) << "Error, resolution format ("
											<<  FLAGS_resolution
											<< ") invalid, should be e.g., 960x540 ";
		nRead = sscanf(FLAGS_camera_resolution.c_str(), "%dx%d",
											 &camera_frame_width, &camera_frame_height);
		CHECK_EQ(nRead,2) << "Error, camera resolution format ("
											<<  FLAGS_camera_resolution
											<< ") invalid, should be e.g., 1280x720";
		nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d",
											 &init_person_net_width, &init_person_net_height);
		CHECK_EQ(nRead,2) << "Error, net resolution format ("
											<<  FLAGS_net_resolution
											<< ") invalid, should be e.g., 656x368 (multiples of 16)";

	}

	NUM_GPU = FLAGS_num_gpu;


	global.part_to_show = FLAGS_part_to_show;
	/* server warming up */
	set_nets();
	//warmup();
	rtcpm();
	//rt29parts(); //without multiple thread


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

bool handleKey(int c) {
	const std::string key2part = "0123456789qwertyuiopas";
	const std::string key2stage = "zxcvbn";
	VLOG(4) << "key: " << (char)c << " code: " << c;
	if (c>=65505) {
		global.uistate.is_shift_down = true;
		c = (char)c;
		c = tolower(c);
		VLOG(4) << "post key: " << (char)c << " code: " << c;
	} else {
		global.uistate.is_shift_down = false;
	}
	VLOG(4) << "shift: " << global.uistate.is_shift_down;
	if (c==27) {
		global.quit_threads = true;
		return false;
	}

	if (c=='g') {
		global.uistate.is_googly_eyes = !global.uistate.is_googly_eyes;
	}

	// Rudimentary seeking in video
	if (c=='l' || c=='k' || c==' ') {
		if (!FLAGS_video.empty()) {
			int cur_frame = global.uistate.current_frame;
			int frame_delta = 30;
			if (global.uistate.is_shift_down) frame_delta = 2;
			if (c=='l') {
				VLOG(4) << "Jump " << frame_delta << " frames to " << cur_frame;
				global.uistate.current_frame+=frame_delta;
				global.uistate.seek_to_frame = 1;
			} else if (c=='k') {
				VLOG(4) << "Rewind " << frame_delta << " frames to " << cur_frame;
				global.uistate.current_frame-=frame_delta;
				global.uistate.seek_to_frame = 1;
			}
		}
		if (c==' ') {
			global.uistate.is_video_paused = !global.uistate.is_video_paused;
		}
	}
	if (c=='f') {
		if (!global.uistate.is_fullscreen) {
			cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
			cv::resizeWindow("video", 1920, 1080);
			cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			global.uistate.is_fullscreen = true;
		} else {
			cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
			cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
			cv::resizeWindow("video", origin_width, origin_height);
			global.uistate.is_fullscreen = false;
		}
	}
	int target = -1;
	int ind = key2part.find(c);
	if (ind!=string::npos){// && !global.uistate.is_shift_down) {
		target = ind;
	}

	if(target >= 0 && target <= 42) {
		global.part_to_show = target;
		LOG(INFO) << "p2s: " << global.part_to_show;
	}

	if (c=='-' || c=='=') {
		if (c=='-') global.nms_threshold -= 0.005;
		if (c=='=') global.nms_threshold += 0.005;
		LOG(INFO) << "nms_threshold: " << global.nms_threshold;
	}
	if (c=='_' || c=='+') {
		if (c=='_') global.connect_min_subset_score -= 0.005;
		if (c=='+') global.connect_min_subset_score += 0.005;
		LOG(INFO) << "connect_min_subset_score: " << global.connect_min_subset_score;
	}
	if (c=='[' || c==']') {
		if (c=='[') global.connect_inter_threshold -= 0.005;
		if (c==']') global.connect_inter_threshold += 0.005;
		LOG(INFO) << "connect_inter_threshold: " << global.connect_inter_threshold;
	}
	if (c=='{' || c=='}') {
		if (c=='{') global.connect_inter_min_above_threshold -= 1;
		if (c=='}') global.connect_inter_min_above_threshold += 1;
		LOG(INFO) << "connect_inter_min_above_threshold: " << global.connect_inter_min_above_threshold;
	}
	if (c==';' || c=='\'') {
		if (c==';') global.connect_min_subset_cnt -= 1;
		if (c=='\'') global.connect_min_subset_cnt += 1;
		LOG(INFO) << "connect_min_subset_cnt: " << global.connect_min_subset_cnt;
	}

	if (c==',' || c=='.') {
		if (c=='.') global.part_to_show++;
		if (c==',') global.part_to_show--;
		if (global.part_to_show<0) {
			global.part_to_show = 42;
		}
		// if (global.part_to_show>42) {
		// 	global.part_to_show = 0;
		// }
		if (global.part_to_show>55) {
			global.part_to_show = 0;
		}
		LOG(INFO) << "p2s: " << global.part_to_show;
	}

	int stage = -1;
	ind = key2stage.find(c);
	if (ind!=string::npos) {
		stage = ind;
		global.uistate.select_stage = stage;
	}


	return true;
}
