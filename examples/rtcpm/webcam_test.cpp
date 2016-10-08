#include "opencv2/opencv.hpp"
using namespace cv;

#include <time.h>
#include <sys/time.h>
#include <cstdio>
#include <ctime>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
    //return (double)time.tv_usec;
}


int main(int argc, char** argv)
{
    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0)){
        printf("no cam.\n");
        return 0;
    }
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,720);

    double last_time = get_wall_time();
    double this_time;
    int count = 0;
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("stream", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC

          count++;

          if(count % 10 == 0){
            this_time = get_wall_time();
            LOG(ERROR) << frame.cols << "  " << frame.rows;
            LOG(ERROR) << "FPS = " << 10.0f / (this_time - last_time);
            last_time = this_time;
          }
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}
