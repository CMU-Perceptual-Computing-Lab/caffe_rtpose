#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}


int main(){

	string input_file;
	ifstream myfile("/home/shihenw/Research/data/eyes/all_png.txt");
	if (myfile){
		while (getline(myfile, input_file)){
			replace(input_file, "./", "/home/shihenw/Research/data/eyes/");
			cout << input_file << endl;

			Mat bgr_image = imread(input_file);

			// namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
		 //    imshow( "Display window", bgr_image );                   // Show our image inside it.
		 //    waitKey(0);                   
			
			cvtColor( bgr_image, bgr_image, CV_BGR2GRAY );
			equalizeHist( bgr_image, bgr_image );
			cvtColor( bgr_image, bgr_image, CV_GRAY2BGR );

			// namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
		 //    imshow( "Display window", bgr_image );                   // Show our image inside it.
		 //    waitKey(0);                   

			string output_file = input_file;
			replace(output_file, "eyes/", "eyes_histeq/");
			cout << output_file << endl;

			imwrite(output_file.c_str(), bgr_image);
		}
		myfile.close();
	}


	

	return 0;
}