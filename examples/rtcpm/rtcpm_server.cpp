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
#include <omp.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"

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

#define boxsize 368
#define batch_size 1
#define fixed_scale_height 368
#define peak_blob_offset 22

//person detector
string person_detector_caffemodel;
string person_detector_proto;

//pose estimator
string pose_estimator_caffemodel;
string pose_estimator_proto;

struct net_copy {
	Net<float>* person_net;
	Net<float>* pose_net;
	vector<int> num_people;
	int total_num_people;
	int net_width;
	int net_height;
	int origin_width;
	int origin_height;
	int nblob_person;
	int nblob_pose;
	float* canvas; //GPU memory, in the size of original image
	//float* enlarged_image; //GPU memory
};

net_copy nc;

void set_nets();
int rtcpm(string filename, string save_folder);
void putGaussianMaps(float* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
void warmup();
void dostuff(int); /* function prototype */
void error(const char *msg);
void process_and_pad_image(float* target, Mat oriImg, int tw, int th);

void set_nets(){
	person_detector_caffemodel = "model/pose_iter_70000.caffemodel"; //"/media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/model/pose_iter_70000.caffemodel";
	person_detector_proto = "model/pose_deploy_copy_4sg_resize.prototxt"; ///media/posenas4b/User/zhe/arch/MPI_exp_caffe/person/pose_deploy_copy_4sg_resize.prototxt";

	pose_estimator_caffemodel = "model/pose_iter_320000.caffemodel"; //  "/media/posenas1/Users/shihenw/caffe_model/MPI/pose_exp78.6_vgg_to4-2/nc0/pose_iter_320000.caffemodel";
	pose_estimator_proto = "model/pose_deploy_resize.prototxt"; // "/media/posenas1/Users/shihenw/caffe_model/MPI/pose_exp78.6_vgg_to4-2/nc0/pose_deploy_resize.prototxt";
}

void warmup(){
	::google::InitGoogleLogging("rtcpm_server");

	LOG(ERROR) << "Setting GPU";
	uint device_id = 0;

	Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
	Caffe::set_mode(Caffe::GPU); //
	LOG(ERROR) << "Setting GPU done.";

	LOG(ERROR) << "Copying to person net";
	nc.person_net = new Net<float>(person_detector_proto, caffe::TEST);
	nc.person_net->CopyTrainedLayersFrom(person_detector_caffemodel);

	nc.nblob_person = nc.person_net->blob_names().size();
	//int nlayer = nc.person_net->layer_names().size();

	// for(int i = 0; i < nc.nblob_person; i++){
	// 	LOG(ERROR) << "person_net Blob: " << i << " " << nc.person_net->blob_names()[i] << " " 
	// 	           << nc.person_net->blobs()[i]->shape(0) << " " 
	// 	           << nc.person_net->blobs()[i]->shape(1) << " " 
	// 	           << nc.person_net->blobs()[i]->shape(2) << " "
	// 	           << nc.person_net->blobs()[i]->shape(3) << " ";
	// }
	// for(int i=0;i<=39;i++){ //cpm_net->layer_names().size();i++){
	// 	LOG(ERROR) << "Layer: " << i << " " << cpm_net->layer_names()[i];
	// }

	LOG(ERROR) << "Copying to pose net";
	nc.pose_net = new Net<float>(pose_estimator_proto, caffe::TEST);
	nc.pose_net->CopyTrainedLayersFrom(pose_estimator_caffemodel);

	nc.nblob_pose = nc.pose_net->blob_names().size();
	//nlayer = nc.person_net->layer_names().size();

	// for(int i = 0; i < nc.nblob_pose; i++) {
	// 	LOG(ERROR) << "pose_net Blob: " << i << " " << nc.pose_net->blob_names()[i] << " " 
	// 	           << nc.pose_net->blobs()[i]->shape(0) << " " 
	// 	           << nc.pose_net->blobs()[i]->shape(1) << " " 
	// 	           << nc.pose_net->blobs()[i]->shape(2) << " "
	// 	           << nc.pose_net->blobs()[i]->shape(3) << " ";
	// }
	
	//dry run
	LOG(INFO) << "Dry running...";
	nc.person_net->ForwardFrom(0);
	nc.pose_net->ForwardFrom(0);
}

void process_and_pad_image(float* target, Mat oriImg, int tw, int th, bool normalize){
	int ow = oriImg.cols;
	int oh = oriImg.rows;
	int offset2_target = tw * th;
	//LOG(ERROR) << tw << " " << th << " " << ow << " " << oh;

	//parallel here
	unsigned char* pointer = (unsigned char*)(oriImg.data);

	for(int c = 0; c < 3; c++){
		#pragma omp parallel for schedule(static) num_threads(4)
		for(int y = 0; y < th; y++){
			for(int x = 0; x < tw; x++){
				if(x < ow && y < oh){
					if(normalize) 
						target[c * offset2_target + y * tw + x] = float(pointer[(y * ow + x) * 3 + c])/256.0f - 0.5f;
					else
						target[c * offset2_target + y * tw + x] = float(pointer[(y * ow + x) * 3 + c]);
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

void render() {
	// LOG(ERROR) << "begin render";
	// LOG(ERROR) << nc.person_net->blob_names()[3] << " " << 
	// 			  nc.person_net->blobs()[3]->shape(0) << " " << 
	//               nc.person_net->blobs()[3]->shape(1) << " " << 
	//               nc.person_net->blobs()[3]->shape(2) << " " <<
	//               nc.person_net->blobs()[3]->shape(3);

	//float* canvas = nc.person_net->blobs()[3]->mutable_gpu_data(); //render layer
	//float* image_ref = nc.person_net->blobs()[0]->mutable_gpu_data();
	float* centers  = nc.person_net->blobs()[nc.nblob_person-1]->mutable_gpu_data();
	float* heatmaps = nc.pose_net->blobs()[nc.nblob_pose-2]->mutable_gpu_data();
	float* poses    = nc.pose_net->blobs()[nc.nblob_pose-1]->mutable_gpu_data(); 
	
	LOG(ERROR) << "begin render_in_cuda";
	caffe::render_in_cuda_website(nc.canvas, nc.origin_width, nc.origin_height, nc.net_width, nc.net_height,
		                          heatmaps, boxsize, 
		                          centers, poses, nc.num_people);

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

void copy_to_posenet_and_reshape(){
	float* image = nc.person_net->blobs()[0]->mutable_gpu_data();
	//float* person_input = nc.pose_net->blobs()[0]->mutable_gpu_data();

	nc.total_num_people = 0;
	float* peak_pointer_cpu = nc.person_net->blobs()[nc.nblob_person-1]->mutable_cpu_data();

	for(int i = 0; i < batch_size; i++){
		nc.num_people[i] = int(peak_pointer_cpu[i * peak_blob_offset]+0.5);
		nc.total_num_people += nc.num_people[i];
	}
	LOG(ERROR) << "total " << nc.total_num_people << " people";
	if(nc.total_num_people == 0) return;

	vector<int> new_shape(4);
	new_shape[0] = nc.total_num_people;
	new_shape[1] = 4;
	new_shape[2] = new_shape[3] = boxsize;
	nc.pose_net->blobs()[0]->Reshape(new_shape);
	nc.pose_net->Reshape();

	float* dst = nc.pose_net->blobs()[0]->mutable_gpu_data();

	float* peak_pointer_gpu = nc.person_net->blobs()[nc.nblob_person-1]->mutable_gpu_data();
	//LOG(ERROR) << "Getting into fill_pose_net";
	caffe::fill_pose_net(image, nc.net_width, nc.net_height, dst, boxsize, peak_pointer_gpu, nc.num_people, 16);
	//src, dst, peak_info, num_people

	// float* pp = nc.pose_net->blobs()[0]->mutable_cpu_data();
	// Mat visualize(boxsize, boxsize, CV_8UC1);
	// for(int i = 0; i < boxsize; i++){
	// 	for(int j = 0; j < boxsize; j++){
	// 		visualize.data[i*boxsize + j] = (unsigned char)(256 * (pp[7*boxsize*boxsize + i*boxsize + j] + 0.5));
	// 	}
	// }
	// imwrite("validate.jpg", visualize);
}

int rtcpm(string filename, string save_folder) {

	nc.num_people.resize(batch_size);
	int return_value;

 	caffe::CPUTimer timer;
	timer.Start();
	
	//(1) load and store original image, both on cpu and gpu
	Mat image_uchar = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if(!image_uchar.data){
		return 1;
	}
	LOG(ERROR) << "Loaded image in size " << image_uchar.cols << " x " << image_uchar.rows;
	if(image_uchar.rows >= 500){
		float ratio = 500 / float(image_uchar.rows);
		resize(image_uchar, image_uchar, Size(0,0), ratio, ratio, INTER_LINEAR);
		LOG(ERROR) << "Too large, resized to " << image_uchar.cols << " x " << image_uchar.rows;
	}
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

	//timer.Start();
	//LOG(ERROR) << "Save on cpu";
	nc.origin_width = image_uchar.cols;
	nc.origin_height = image_uchar.rows;
	float* original_image = new float [nc.origin_width * nc.origin_height * 3];
	process_and_pad_image(original_image, image_uchar, nc.origin_width, nc.origin_height, 0); // don't normalize
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

	//timer.Start();
	//LOG(ERROR) << "Save on gpu";
	cudaMalloc(&nc.canvas, nc.origin_width * nc.origin_height * 3 * 16 * sizeof(float));
	int offset_canvas = nc.origin_width * nc.origin_height * 3;
	for(int i = 0; i < 16; i++){
		cudaMemcpy(nc.canvas + i*offset_canvas, original_image, nc.origin_width * nc.origin_height * 3 * sizeof(float), 
			      cudaMemcpyHostToDevice);
	}
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


	//(2) Transform and pad to 8's multiplier
	//timer.Start();
	//LOG(ERROR) << "Transform and pad to 8's multiplier (should be parallelized over CPUs)";
	//resize
	int net_height = fixed_scale_height; //some fixed number that is multiplier of 8
	int net_width = float(fixed_scale_height) / image_uchar.rows * image_uchar.cols;
	resize(image_uchar, image_uchar, Size(net_width, net_height), 0, 0, INTER_CUBIC);
	
	if(net_width % 8 != 0) {
		net_width = 8 * (net_width / 8 + 1);
	}
	nc.net_width = net_width;
	nc.net_height = net_height;

	//pad and transform to float
	float* processed_and_padded_image = new float [net_width * net_height * 3];
	process_and_pad_image(processed_and_padded_image, image_uchar, net_width, net_height, 1); // need normalize
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

	// (3) reshape person_net and prepare data on gpu blob
	//timer.Start();
	//LOG(ERROR)<< "Reshape person_net to target_dim: " << nc.net_height << " " << nc.net_width;
	vector<int> shape(4); 
	shape[0] = 1;
	shape[1] = 3;
	shape[2] = nc.net_height;
	shape[3] = nc.net_width;
	nc.person_net->blobs()[0]->Reshape(shape);
	nc.person_net->Reshape();
	// LOG(ERROR) << "after reshape";
	// for(int i = 0; i < nblob; i++) {
	// 	LOG(ERROR) << "person_net Blob: " << i << " " << nc.person_net->blob_names()[i] << " " 
	// 	           << nc.person_net->blobs()[i]->shape(0) << " " 
	// 	           << nc.person_net->blobs()[i]->shape(1) << " " 
	// 	           << nc.person_net->blobs()[i]->shape(2) << " "
	// 	           << nc.person_net->blobs()[i]->shape(3) << " ";
	// }
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

	//copy to device
	//timer.Start();
	//LOG(ERROR)<< "Copy to device";
	//int offset = boxsize * boxsize;
	float* d_pointer = nc.person_net->blobs()[0]->mutable_gpu_data();
	//cudaStream_t stream[batch_size];
	for(int n = 0; n < batch_size; n++){
		cudaMemcpy(d_pointer, processed_and_padded_image, nc.net_width * nc.net_height * 3 * sizeof(float), 
			      cudaMemcpyHostToDevice);
	}
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


	// (4) Run person net
	//timer.Start();
	//LOG(ERROR) << "Running forward person_net";
	//nc.person_net->ForwardFromTo(0,nlayer-1);
	nc.person_net->ForwardFrom(0);
	cudaDeviceSynchronize();
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


	// timer.Start();
	// shared_ptr<Blob<float> > b = nc.person_net->blobs()[nc.nblob_person-2];
	// LOG(ERROR)<< "Saving heatmap (1) from shape " << b->shape(2) << "(h) x " << b->shape(3) << "(w)";

	// const float* heatmap_pointer = b->cpu_data();
	
	// Mat heatmap(b->shape(2), b->shape(3), CV_8UC1); //constructor: height first
	// // //offset = b->shape(3) * b->shape(2);

	// for (int y = 0; y < b->shape(2); y++){
	// 	for (int x = 0; x < b->shape(3); x++){
	// 		float num = heatmap_pointer[y * b->shape(3) + x]; //0 ~ 1;
	// 		num = (num > 1 ? 1 : (num < 0 ? 0 : num)); //prevent overflow for uchar
	// 	    heatmap.data[(y * b->shape(3) + x)] = (unsigned char)(num * 255);
	// 	}
	// }

	// b = nc.person_net->blobs()[nblob-1];
	// float* peaks = b->mutable_cpu_data();
	// for(int i = 0; i < 22; i++){
	// 	cout << peaks[i] << " ";
	// }
	// cout << endl;

	// for(int i = 0; i < peaks[0]; i++){
	// 	circle(heatmap, Point2f(peaks[2*(i+1)], peaks[2*(i+1)+1]), 2, Scalar(0,0,0), -1);
	// }

	// cv::imwrite("person_map.jpg", heatmap);
	// timer.Stop();
	// LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";


	//copy data to pose_net
	//timer.Start();
	//LOG(ERROR) << "copy to posenet and reshape";
	copy_to_posenet_and_reshape();
	//timer.Stop();
	//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

	if(nc.total_num_people != 0){
		//timer.Start();
		//LOG(ERROR) << "Forwarding pose net";
		nc.pose_net->ForwardFrom(0);
		// nblob = nc.pose_net->blob_names().size();
		// for(int i = 0; i < nblob; i++) {
		// 	LOG(ERROR) << "pose_net Blob: " << i << " " << nc.pose_net->blob_names()[i] << " " 
		// 	           << nc.pose_net->blobs()[i]->shape(0) << " " 
		// 	           << nc.pose_net->blobs()[i]->shape(1) << " " 
		// 	           << nc.pose_net->blobs()[i]->shape(2) << " "
		// 	           << nc.pose_net->blobs()[i]->shape(3) << " ";
		// }
		cudaDeviceSynchronize();
		//timer.Stop();
		//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";

		//timer.Start();
		//LOG(ERROR) << "Rendering";
		
		render();

		//timer.Stop();
		//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";
		
		//float* out = nc.person_net->blobs()[3]->mutable_cpu_data();
		boost::filesystem::path p(filename);
		string rawname = p.stem().string();

		for(int part = 0; part < 16; part++){
			cudaMemcpy(original_image, nc.canvas + part * offset_canvas, 
				       nc.origin_width * nc.origin_height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
			
			float* out = original_image;
			//timer.Start();
			//LOG(ERROR) << "Saving";
			Mat visualize(nc.origin_height, nc.origin_width, CV_8UC3);
			for(int c = 0; c < 3; c++){
				#pragma omp parallel for schedule(static) num_threads(8)
				for(int i = 0; i < nc.origin_height; i++){
					for(int j = 0; j < nc.origin_width; j++){
						int value = int(out[c * nc.origin_height * nc.origin_width + i * nc.origin_width + j] + 0.5);
						value = value < 0 ? 0 : (value > 255 ? 255 : value);
						visualize.data[3 * (i * nc.origin_width + j) + c] = (unsigned char)(value);
					}
				}
			}
			char filename_to_save[256];
			if(part == 0) {
				sprintf(filename_to_save, "%s/%s_pose.jpg", save_folder.c_str(), rawname.c_str());
			} else {
				sprintf(filename_to_save, "%s/%s_part%02d.jpg", save_folder.c_str(), rawname.c_str(), part);
			}
			LOG(ERROR) << "saving " << filename_to_save;
			imwrite(filename_to_save, visualize);
			//timer.Stop();
			//LOG(ERROR) << "Time: " << timer.MicroSeconds() << "us.";
		}
		return_value = 0;
	} else { // no people
		return_value = -1;
	}

	timer.Stop();
	LOG(ERROR) << "Total Time: " << timer.MicroSeconds() << "us.";

	delete [] processed_and_padded_image;
	delete [] original_image;
	cudaFree(nc.canvas);
	return return_value;
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

	string msg(buffer);
	istringstream iss(msg);
	string filename, save_folder;
	for(int i = 0; i < 2; i++) {
		if(i == 0)
			getline(iss, filename, ';');
		else
			getline(iss, save_folder, ';');
	}

	boost::filesystem::path folder(save_folder);
    if (!exists(folder)){
    	boost::filesystem::create_directory(save_folder);
    	cout << save_folder << " is created.\n";
    }


	int status = rtcpm(filename, save_folder);

	ofstream myfile;
    char msg_file[256];
    sprintf(msg_file, "%s/message", save_folder.c_str());
    myfile.open(msg_file);

	//int status = 0;
	if(status == 0){
		char msg[] = "You image is processed";
		LOG(ERROR) << "Returning: You image is processed";
		myfile << "success" << endl;
		n = write(sock, msg, strlen(msg));
		if (n < 0) LOG(ERROR) << "Error when writing to socket";
	}
	else if(status == 1){
		char msg[] = "Have problems processing your image";
		LOG(ERROR) << "Returning: Have problems processing your image";
		myfile << "fail" << endl << "image cannot be opened" << endl;
		n = write(sock, msg, strlen(msg));
		if (n < 0) LOG(ERROR) << "Error when writing to socket";
	}
	else {
		char msg[] = "No people detected";
		LOG(ERROR) << "Returning: No people detected";
		myfile << "fail" << endl << "no people detected" << endl;
		n = write(sock, msg, strlen(msg));
		if (n < 0) LOG(ERROR) << "Error when writing to socket";
	}

	myfile.close();
}

void error(const char *msg){
    perror(msg);
    exit(1);
}

int main(int argc, char** argv) {

	/* server warming up */
	set_nets();
	warmup();

	/* begin socket programming */
 	int sockfd, newsockfd, portno; /*, pid;*/
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    signal(SIGCHLD, SIG_IGN);


    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = 12345;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) 
              error("ERROR on binding");
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    LOG(ERROR) << "Init ready, waiting for request....";
    while (1) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0) 
            error("ERROR on accept");
        //pid = fork();
        //if (pid < 0)
        //    error("ERROR on fork");
        //if (pid == 0)  {
        //    close(sockfd);
            dostuff(newsockfd);
        //    exit(0);
        //}
        //else close(newsockfd);
    } /* end of while */
    close(sockfd);
    return 0; /* we never get here */
}

