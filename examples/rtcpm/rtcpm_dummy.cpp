#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <filename> <save_folder>" << endl;
        exit(1);
    }
    string filename(argv[1]);
    string save_folder(argv[2]);

    boost::filesystem::path folder(save_folder);
    if (!exists(folder)){
    	//cerr << "not existing.";
    	//exit(-1);
    	boost::filesystem::create_directory(save_folder);
    	cout << save_folder << " is created.\n";
    } 

    Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if(!image.data){
		cerr << "failure. image not found." << endl;
		return 1;
	}
	boost::filesystem::path p(filename);

	//size_t lastindex = filename.find_last_of("."); 
	string rawname = p.stem().string();

	resize(image, image, Size(0,0), 0.5, 0.5, INTER_CUBIC);

	char filename_to_save[256];
	for (int part = 1; part <= 15; part++){
		for(int stage = 1; stage <= 6; stage++){
			sprintf(filename_to_save, "%s/%s_part%02d_sg%d.jpg", save_folder.c_str(), rawname.c_str(), part, stage);
			try{
				bool result = imwrite(filename_to_save, image);
				if(!result){
					cerr << "failure. writing error." << endl;
					return 1;
				}
			}
			catch (runtime_error& ex) {
		        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		        return 1;
		    }
		}
	}
	sprintf(filename_to_save, "%s/%s_pose.jpg", save_folder.c_str(), rawname.c_str());
	try{
		bool result = imwrite(filename_to_save, image);
		if(!result){
			cerr << "failure. writing error." << endl;
			return 1;
		}
	}
	catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 1;
    }

	cout << "success. In: " << save_folder << "/" << rawname << "_part[01-14]_sg[1-6].jpg" << endl;
	return 0;
}