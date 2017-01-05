#include "caffe/cpm/cpm_data_transformer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
using namespace cv;
using namespace std;

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
CpmDataTransformer<Dtype>::CpmDataTransformer(const TransformationParameter& param, Phase phase) :
    DataTransformer<Dtype>(param, phase) {
  np_in_lmdb = this->param_.np_in_lmdb();
  np = this->param_.num_parts();
  is_table_set = false;

  ofs_analysis.open("analysis.log", ofstream::out);
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::DecodeFloats(const string& data, size_t idx, float* pf, size_t len) {
  memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
}

template<typename Dtype>
string CpmDataTransformer<Dtype>::DecodeString(const string& data, size_t idx) {
  string result = "";
  int i = 0;
  while(data[idx+i] != 0){
    result.push_back(char(data[idx+i]));
    i++;
  }
  return result;
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  meta.type = "cpm";
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = Size(width, height);
  // ----------- Validation, nop, counters -----------------
  meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
  meta.numOtherPeople = (int)data[offset3+2*offset1+1];
  meta.people_index = (int)data[offset3+2*offset1+2];
  float annolist_index;
  DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
  meta.annolist_index = (int)annolist_index;
  float write_number;
  DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
  meta.write_number = (int)write_number;
  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(meta.write_number % 1000 == 0){
    LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
        << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number
        << "; meta.total_write_number: " << meta.total_write_number << "; meta.epoch: " << meta.epoch;
  }
  if(this->param_.aug_way() == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }

  // ------------------- objpos -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  meta.objpos -= Point2f(1,1);
  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    meta.joint_self.joints[i] -= Point2f(1,1); //from matlab 1-index to c++ 0-index
    float isVisible;
    DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
    meta.joint_self.isVisible[i] = isVisible;

    //if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
    //   meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
      //if(meta.dataset.find("eyes") == string::npos){
    //    meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
      //} else {
      //  meta.joint_self.isVisible[i] = 3; // 3 means missing point, just for eye dataset for now
      //}
    //}
    //LOG(INFO) << meta.joint_self.joints[i].x << " " << meta.joint_self.joints[i].y << " " << meta.joint_self.isVisible[i];
  }

  //others (8 lines loaded)
  meta.objpos_other.resize(meta.numOtherPeople);
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);
  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    meta.objpos_other[p] -= Point2f(1,1);
    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(8+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(8+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      meta.joint_others[p].joints[i] -= Point2f(1,1);
      float isVisible;
      DecodeFloats(data, offset3+(8+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
      meta.joint_others[p].isVisible[i] = isVisible;
      //if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
      //   meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
      //  meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 1 means occluded by still on image
      //}
    }
  }
  //8 + 4*numOtherPeople lines loaded
  if(this->param_.has_masks()){
    meta.has_teeth_mask = data[offset3 + (8 + 4*meta.numOtherPeople)*offset1];
    //LOG(INFO) << "teeth_mask: " << meta.has_teeth_mask;
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::ReadMetaData_COCO(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB_COCO.py
  meta.type = "detect";
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = Size(width, height);
  // ----------- Validation, nop, counters -----------------
  meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
  meta.numAnns = (int)data[offset3+2*offset1+1];

  float image_id_in_coco;
  DecodeFloats(data, offset3+2*offset1+2, &image_id_in_coco, 1);
  meta.image_id_in_coco = (int)image_id_in_coco;

  float image_id_in_training;
  DecodeFloats(data, offset3+2*offset1+6, &image_id_in_training, 1);
  meta.image_id_in_training = (int)image_id_in_training;

  float write_number;
  DecodeFloats(data, offset3+2*offset1+10, &write_number, 1);
  meta.write_number = (int)write_number;

  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+14, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(meta.write_number % 1000 == 0){
    LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
        << "; meta.numAnns: " << meta.numAnns
        << "; meta.image_id_in_coco: " << meta.image_id_in_coco
        << "; meta.image_id_in_training: " << meta.image_id_in_training
        << "; meta.write_number: " << meta.write_number
        << "; meta.total_write_number: " << meta.total_write_number << "; meta.epoch: " << meta.epoch;
  }

  // ------------ num_keypoints and iscrowd -------------
  //LOG(INFO) << "num_keypoints: ";
  meta.num_keypoints.resize(meta.numAnns);
  for(int a = 0; a < meta.numAnns; a++){
    meta.num_keypoints[a] = int(data[offset3 + 3*offset1 + a]);
    //LOG(INFO) << meta.num_keypoints[a];
  }

  //LOG(INFO) << "is_crowd: ";
  meta.iscrowd.resize(meta.numAnns);
  for(int a = 0; a < meta.numAnns; a++){
    meta.iscrowd[a] = data[offset3 + 4*offset1 + a] == 0 ? false : true;
    //LOG(INFO) << meta.iscrowd[a];
  }

  // ------------ bboxes(1 line), annotations(3 line) --------------
  meta.bboxes.resize(meta.numAnns);
  meta.annotations.resize(meta.numAnns);

  for(int a = 0; a < meta.numAnns; a++){
    float bbox[4];
    DecodeFloats(data, offset3 + (5 + 4*a)*offset1, bbox, 4);
    meta.bboxes[a].left = bbox[0];
    meta.bboxes[a].top = bbox[1];
    meta.bboxes[a].width = bbox[2];
    meta.bboxes[a].height = bbox[3];

    //LOG(INFO) << bbox[0] << "\t" << bbox[1] << "\t" << bbox[2] << "\t" << bbox[3];

    float keypoints_row[3][17];
    DecodeFloats(data, offset3 + (6 + 4*a)*offset1, keypoints_row[0], 17);
    DecodeFloats(data, offset3 + (7 + 4*a)*offset1, keypoints_row[1], 17);
    DecodeFloats(data, offset3 + (8 + 4*a)*offset1, keypoints_row[2], 17);
    for(int j = 0; j < 17; j++){
      meta.annotations[a].joints.resize(17);
      meta.annotations[a].isVisible.resize(17);
      meta.annotations[a].joints[j].x = keypoints_row[0][j];
      meta.annotations[a].joints[j].y = keypoints_row[1][j];
      meta.annotations[a].isVisible[j] = int(keypoints_row[2][j]);
      //cout << meta.annotations[a].joints[j].x << "\t" << meta.annotations[a].joints[j].y << "\t" << meta.annotations[a].isVisible[j] << endl;
    }
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::SetAugTable(int numData){
  aug_degs.resize(numData);
  aug_flips.resize(numData);
  for(int i = 0; i < numData; i++){
    aug_degs[i].resize(this->param_.num_total_augs());
    aug_flips[i].resize(this->param_.num_total_augs());
  }
  //load table files
  char filename[100];
  sprintf(filename, "../../rotate_%d_%d.txt", this->param_.num_total_augs(), numData);
  ifstream rot_file(filename);
  char filename2[100];
  sprintf(filename2, "../../flip_%d_%d.txt", this->param_.num_total_augs(), numData);
  ifstream flip_file(filename2);

  for(int i = 0; i < numData; i++){
    for(int j = 0; j < this->param_.num_total_augs(); j++){
      rot_file >> aug_degs[i][j];
      flip_file >> aug_flips[i][j];
    }
  }
  // for(int i = 0; i < numData; i++){
  //   for(int j = 0; j < this->param_.num_total_augs(); j++){
  //     printf("%d ", (int)aug_degs[i][j]);
  //   }
  //   printf("\n");
  // }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::TransformJoints(Joints& j) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  //MPII R leg: 0(ankle), 1(knee), 2(hip)
  //     L leg: 5(ankle), 4(knee), 3(hip)
  //     R arms: 10(wrist), 11(elbow), 12(shoulder)
  //     L arms: 15(wrist), 14(elbow), 13(shoulder)
  //     6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top
  //LOG(INFO) << "TransformJoints: here np == " << np << " np_lmdb = " << np_in_lmdb << " joints.size() = " << j.joints.size();
  //assert(joints.size() == np_in_lmdb);
  //assert(np == 14 || np == 28);
  Joints jo = j;
  if(np == 14){
    int MPI_to_ours[14] = {9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<np;i++){
      jo.joints[i] = j.joints[MPI_to_ours[i]];
      jo.isVisible[i] = j.isVisible[MPI_to_ours[i]];
    }
  }
  else if(np == 28){
    int MPI_to_ours_1[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             9, 8,12,11, 8,13,14, 2, 1, 3, 4, 6};
                          //17,18,19,20,21,22,23,24,25,26,27,28
    int MPI_to_ours_2[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             8,12,11,10,13,14,15, 1, 0, 4, 5, 7};
                          //17,18,19,20,21,22,23,24,25,26,27,28
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<np;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }
  }
  else if(np == 56){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(18);
    jo.isVisible.resize(18);
    np_in_lmdb = 18; //fake it

    for(int i = 0; i < 18; i++){
      jo.joints[i] = (j.joints[COCO_to_ours_1[i]-1] + j.joints[COCO_to_ours_2[i]-1]) * 0.5;
      if(j.isVisible[COCO_to_ours_1[i]-1]==2 || j.isVisible[COCO_to_ours_2[i]-1]==2){
        jo.isVisible[i] = 2;
      }
      else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i] = j.isVisible[COCO_to_ours_1[i]-1] && j.isVisible[COCO_to_ours_2[i]-1];
      }
    }
  }

  j = jo;
}

template<typename Dtype>
float CpmDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, MetaData& meta, float scale_multiplier) {

  bool change_meta = 0;
  if(scale_multiplier == -1){
    change_meta = 1;
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    //float scale_multiplier;
    //float scale = (this->param_.scale_max() - this->param_.scale_min()) * dice + this->param_.scale_min(); //linear shear into [scale_min, scale_max]
    if(dice > this->param_.scale_prob()) {
      img_temp = img_src.clone();
      scale_multiplier = 1;
    }
    else {
      float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
      scale_multiplier = (this->param_.scale_max() - this->param_.scale_min()) * dice2 + this->param_.scale_min(); //linear shear into [scale_min, scale_max]
    }
  }

  float scale = 1;
  if(meta.type == "cpm") {
    float scale_abs = this->param_.target_dist() / meta.scale_self;
    scale = scale_abs * scale_multiplier;
  } else if(meta.type == "detect") {
    float scale_abs = this->param_.target_dist() / meta.img_size.height; // no self scale here
    scale = scale_abs * scale_multiplier;
  }

  resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);

  //modify meta data
  if(change_meta){
    if(meta.type == "cpm") {
      meta.objpos *= scale;
      for(int i=0; i<np; i++){
        meta.joint_self.joints[i] *= scale;
      }
      for(int p=0; p<meta.numOtherPeople; p++){
        meta.objpos_other[p] *= scale;
        for(int i=0; i<np; i++){
          meta.joint_others[p].joints[i] *= scale;
        }
      }
    } else if(meta.type == "detect") {
      for(int a = 0; a < meta.numAnns; a++){
        meta.bboxes[a].left *= scale;
        meta.bboxes[a].top *= scale;
        meta.bboxes[a].width *= scale;
        meta.bboxes[a].height *= scale;

        for(int j=0; j<np; j++){
          meta.annotations[a].joints[j] *= scale;
        }
      }
    }
  }
  return scale_multiplier;
}

template<typename Dtype>
bool CpmDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

template<typename Dtype>
Size CpmDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, MetaData& meta, Size anchor, bool last_time) {

  int crop_x = this->param_.crop_size_x();
  int crop_y = this->param_.crop_size_y();

  bool change_meta = 0;
  if(anchor.width == -999 && anchor.height == -999){
    change_meta = 1;
    float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]

    anchor.width = int((dice_x - 0.5) * 2 * this->param_.center_perterb_max());
    anchor.height = int((dice_y - 0.5) * 2 * this->param_.center_perterb_max());
  }

  //LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);

  Point2i center;
  if(meta.type == "cpm") {
    center = meta.objpos + Point2f(anchor.width, anchor.height);
  } else if(meta.type == "detect") {
    center = Point2f(img_src.cols/2, img_src.rows/2) + Point2f(anchor.width, anchor.height);
  }

  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);

  if(change_meta)
    img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  else
    img_dst = Mat::zeros(crop_y, crop_x, CV_8U);

  for(int i = 0; i < crop_y; i++){
    for(int j = 0; j < crop_x; j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
        if(change_meta)
          img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
        else
          img_dst.at<uchar>(i,j) = img_src.at<uchar>(coord_y_on_img, coord_x_on_img);
      }
    }
  }

  Point2f offset(offset_left, offset_up);

  if(meta.type == "cpm"){
    //modify meta data
    if(change_meta){
      //meta.objpos += offset; // if you change it now, all following masks are wrong
      for(int i=0; i<np; i++){
        meta.joint_self.joints[i] += offset;
      }
      for(int p=0; p<meta.numOtherPeople; p++){
        meta.objpos_other[p] += offset;
        for(int i=0; i<np; i++){
          meta.joint_others[p].joints[i] += offset;
        }
      }
    }
    if(last_time){
      meta.objpos += offset;
    }
  } else if(meta.type == "detect") {
    if(change_meta){
      for(int a=0; a<meta.numAnns; a++){
        for(int i=0; i<np; i++){
          meta.annotations[a].joints[i] += offset;
        }
        meta.bboxes[a].left += offset_left;
        meta.bboxes[a].top += offset_up;
        // some bbox may be (partially) out of image. We keep it here, but when generate label we change
      }
    }
  }

  return anchor;
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::swapLeftRight(Joints& j) {
  //assert(j.joints.size() == 9 && j.joints.size() == 14 && j.isVisible.size() == 28);
  //MPII R leg: 0(ankle), 1(knee), 2(hip)
  //     L leg: 5(ankle), 4(knee), 3(hip)
  //     R arms: 10(wrist), 11(elbow), 12(shoulder)
  //     L arms: 15(wrist), 14(elbow), 13(shoulder)
  if(np == 9){
    int right[4] = {1,2,3,7};
    int left[4] = {4,5,6,8};
    for(int i=0; i<4; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 14){
    int right[6] = {3,4,5,9,10,11}; //1-index
    int left[6] = {6,7,8,12,13,14}; //1-index
    for(int i=0; i<6; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 27){
    int right[12] = {3,4,5,9,10,11,15,18,19,20,24,25}; //1-index
    int left[12] = {6,7,8,12,13,14,16,21,22,23,26,27}; //1-index
    for(int i=0; i<12; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 28){
    int right[11] = {3,4,5,9,10,11,18,19,20,24,25}; //1-index
    int left[11] = {6,7,8,12,13,14,21,22,23,26,27}; //1-index
    for(int i=0; i<11; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 33){
    int right[15] = {3,4,5, 9,10,11,15,17,20,21,22,26,27,30,32};
    int left[15] =  {6,7,8,12,13,14,16,18,23,24,25,28,29,31,33};
    for(int i=0; i<15; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 34){
    int right[14] = {3,4,5, 9,10,11,15,17,20,21,22,26,27,32};
    int left[14] =  {6,7,8,12,13,14,16,18,23,24,25,28,29,33};
    for(int i=0; i<14; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 36){
    int right[16] = {1,3,5,7,10,11,12,13,14,15,23,24,25,26,32,33};
    int left[16] =  {2,4,6,8,17,16,19,18,21,20,28,27,30,29,35,34};
    for(int i=0; i<16; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 37){
    int right[8] = {3,4,5, 9,10,11,15,17};
    int left[8] =  {6,7,8,12,13,14,16,18};
    for(int i=0; i<8; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 43){
    int right[18] = {3,4,5,9,10,11,18,19,20,21,22,23,32,33,34,35,36,37};
    int left[18] = {6,7,8,12,13,14,24,25,26,27,28,29,38,39,40,41,42,43};
    for(int i=0; i<18; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 52){
    int right[24] = {3,4,5, 9,10,11,15,17,19,20,21,22,23,24,31,32,33,34,35,36,37,38,49,51};
    int left[24] =  {6,7,8,12,13,14,16,18,25,26,27,28,29,30,39,40,41,42,43,44,45,46,50,52};
    for(int i=0; i<24; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 56){
    //int right[26] = {3,4,5, 9,10,11,15,17,19,20,21,22,23,24,31,32,33,34,35,36,37,38,49,50,53,55};
    //int left[26] =  {6,7,8,12,13,14,16,18,25,26,27,28,29,30,39,40,41,42,43,44,45,46,51,52,54,56};
    //for(int i=0; i<26; i++){
    int right[8] = {3,4,5, 9,10,11,15,17};
    int left[8] =  {6,7,8,12,13,14,16,18};
    for(int i=0; i<8; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 75){
    int right[35] = {3,4,5, 9,10,11,15,17,19,20,21,22,23,24,25,26,27,37,38,39,40,41,42,43,44,45,46,47,48,64,65,66,70,71,72};
    int left[35] =  {6,7,8,12,13,14,16,18,28,29,30,31,32,33,34,35,36,49,50,51,52,53,54,55,56,57,58,59,60,67,68,69,73,74,75};
    for(int i=0; i<35; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 78){
    int right[36] = {1,3,5,7, 9,11,14,15,16,17,18,19,20,21,22,23,35,36,37,38,39,40,41,42,52,53,54,55,56,57,65,66,67,68,74,75};
    int left[36] =  {2,4,6,8,10,12,25,24,27,26,29,28,31,30,33,32,44,43,46,45,48,47,50,49,59,58,61,60,63,62,70,69,72,71,77,76};
    for(int i=0; i<36; i++){
      int ri = right[i] - 1;
      int li = left[i] - 1;
      Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
}

template<typename Dtype>
int CpmDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, MetaData& meta, int doflip) {
  bool change_meta = 0;
  if(doflip == -1){ //-1:not set, 0:no, 1: yes
    change_meta = 1;
    if(this->param_.aug_way() == "rand"){
      float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      doflip = (dice <= this->param_.flip_prob());
    }
    else if(this->param_.aug_way() == "table"){
      doflip = (aug_flips[meta.write_number][meta.epoch % this->param_.num_total_augs()] == 1);
    }
    else {
      doflip = 0;
      LOG(INFO) << "Unhandled exception!!!!!!";
    }
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;

    if(change_meta){
      if(meta.type == "cpm"){
        meta.objpos.x = w - 1 - meta.objpos.x;
        for(int i=0; i<np; i++){
          meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
        }
        if(this->param_.transform_body_joint())
          swapLeftRight(meta.joint_self);

        for(int p=0; p<meta.numOtherPeople; p++){
          meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
          for(int i=0; i<np; i++){
            meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
          }
          if(this->param_.transform_body_joint())
            swapLeftRight(meta.joint_others[p]);
        }
      } else if(meta.type == "detect") {
        for(int a = 0; a < meta.numAnns; a++){
          for(int i=0; i<np; i++){
            meta.annotations[a].joints[i].x = w - 1 - meta.annotations[a].joints[i].x;
          }
          if(this->param_.transform_body_joint())
            swapLeftRight(meta.annotations[a]);
          meta.bboxes[a].left = w - 1 - (meta.bboxes[a].left + meta.bboxes[a].width);
        }
      }
    }
  } else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat R){
  Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
float CpmDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, MetaData& meta, float degree) {

  bool change_meta = 0;
  if(degree == -999){
    change_meta = 1;
    if(this->param_.aug_way() == "rand"){
      float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      degree = (dice - 0.5) * 2 * this->param_.max_rotate_degree();
    }
    else if(this->param_.aug_way() == "table"){
      degree = aug_degs[meta.write_number][meta.epoch % this->param_.num_total_augs()];
    }
    else {
      degree = 0;
      LOG(INFO) << "Unhandled exception!!!!!!";
    }
  }

  Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";"
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  if(change_meta)
    warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  else
    warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC);

  //adjust meta data
  if(change_meta){
    if(meta.type == "cpm") {
      RotatePoint(meta.objpos, R);
      for(int i=0; i<np; i++){
        RotatePoint(meta.joint_self.joints[i], R);
      }
      for(int p=0; p<meta.numOtherPeople; p++){
        RotatePoint(meta.objpos_other[p], R);
        for(int i=0; i<np; i++){
          RotatePoint(meta.joint_others[p].joints[i], R);
        }
      }
    } else if(meta.type == "detect") {
      for(int a = 0; a < meta.numAnns; a++){
        for(int i = 0; i < np; i++){
          RotatePoint(meta.annotations[a].joints[i], R);
        }
        //bounding box..
        float area = meta.bboxes[a].width * meta.bboxes[a].height;
        Point2f lt(meta.bboxes[a].left                       , meta.bboxes[a].top);
        Point2f lb(meta.bboxes[a].left                       , meta.bboxes[a].top + meta.bboxes[a].height);
        Point2f rt(meta.bboxes[a].left + meta.bboxes[a].width, meta.bboxes[a].top);
        Point2f rb(meta.bboxes[a].left + meta.bboxes[a].width, meta.bboxes[a].top + meta.bboxes[a].height);
        //LOG(INFO) << lt << lb << rt << rb;
        RotatePoint(lt, R);
        RotatePoint(lb, R);
        RotatePoint(rt, R);
        RotatePoint(rb, R);
        //LOG(INFO) << lt << lb << rt << rb;
        meta.bboxes[a].left = min(min(lt.x, lb.x), min(rt.x, rb.x));
        meta.bboxes[a].top = min(min(lt.y, lb.y), min(rt.y, rb.y));
        meta.bboxes[a].width = max(max(lt.x, lb.x), max(rt.x, rb.x)) - meta.bboxes[a].left;
        meta.bboxes[a].height = max(max(lt.y, lb.y), max(rt.y, rb.y)) - meta.bboxes[a].top;
        //LOG(INFO) << meta.bboxes[a].left << " " << meta.bboxes[a].top << " " << meta.bboxes[a].width << " " << meta.bboxes[a].height;
        //LOG(INFO) << " ";
        // shrink bbox to keep constant area
        float new_area = meta.bboxes[a].width * meta.bboxes[a].height;
        float shrink_ratio = sqrt(new_area / area); // > 1
        Point2f bc(meta.bboxes[a].left + meta.bboxes[a].width/2, meta.bboxes[a].top + meta.bboxes[a].height/2);
        meta.bboxes[a].width /= shrink_ratio;
        meta.bboxes[a].height /= shrink_ratio;
        meta.bboxes[a].left = bc.x - meta.bboxes[a].width/2;
        meta.bboxes[a].top = bc.y - meta.bboxes[a].height/2;
      }
    }
  }
  return degree;
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma, float peak_ratio){
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
      entry[g_y*grid_x + g_x] += peak_ratio * exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1)
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Bbox b, int stride, int grid_x, int grid_y, float sigma_0){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  //fix bbox first
  float right = b.left + b.width;
  float bottom = b.top + b.height;
  if(b.left >= grid_x * stride || right < 0 || b.top >= grid_y * stride || bottom < 0) return;

  if(b.left < 0){
    b.left = 0;
    b.width = right;
  }
  if(right >= stride * grid_x){
    b.width = stride * grid_x - b.left;
  }
  if(b.top < 0){
    b.top = 0;
    b.height = bottom;
  }
  if(bottom >= stride * grid_y){
    b.height = stride * grid_y - b.top;
  }

  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...

  Point2f center(b.left+b.width/2, b.top+b.height/2);

  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float sigma_x = sigma_0 * b.width;
      float sigma_y = sigma_0 * b.height;
      float exponent = (x-center.x)*(x-center.x)/2.0/sigma_x + (y-center.y)*(y-center.y)/2.0/sigma_y;
      //float exponent = d2 / 2.0 / sigma_0 / sigma_0;
      if(exponent > 6.9078){ //ln(1000) = -ln(0.1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      //if(entry[g_y*grid_x + g_x] > 1)
      //  entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::putScaleMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma, float scale_normalize){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent < 2.3026){ // -ln(10%)
        entry[g_y*grid_x + g_x] = scale_normalize;
      }
    }
  }
}

void setLabel(Mat& im, const std::string label, const Point& org) {
    int fontface = FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, org + Point(0, baseline), org + Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
}


template<typename Dtype>
void CpmDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat& img_aug, MetaData meta, Dtype* mask, Mat& teeth_mask) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = this->param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;
  int mask_channel = (this->param_.has_masks() ? 1 : 0);

  // clear out transformed_label, it may remain things for last batch
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      if(meta.type == "cpm"){
        for (int i = 0; i < 2*(np+1+mask_channel); i++){
          transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
        }
      } else if(meta.type == "detect") {
        for (int i = 0; i < np+1+mask_channel; i++){
          transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
        }
      }
    }
  }
  //LOG(INFO) << "label cleaned";

  if(meta.type == "cpm"){
    // foreground parts
    for (int i = 0; i < np; i++){
      //LOG(INFO) << i << meta.numOtherPeople;
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + i*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma()); //self
        putGaussianMaps(transformed_label + (i+np+1+mask_channel)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma()); //self
      }
      //LOG(INFO) << "label put for" << i;
      //plot others
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+1+mask_channel)*channelOffset, center, this->param_.stride(),
                          grid_x, grid_y, this->param_.sigma());
        }
      }
    }
    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++) {
      for (int g_x = 0; g_x < grid_x; g_x++) {
        float maximum = 0;
        for (int i = 0; i < np; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[np*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
        //second background channel
        maximum = 0;
        for (int i = np+1+mask_channel; i < 2*np+1+mask_channel; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1+mask_channel)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
    //put mask channel
    if(this->param_.has_masks()){
      float one_over_stride = 1.0/stride;
      resize(teeth_mask, teeth_mask, Size(), one_over_stride, one_over_stride, INTER_CUBIC);
      //assert(teeth_mask.cols == g_x && teeth_mask.rows == g_y);

      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          transformed_label[(np+1)*channelOffset + g_y*grid_x + g_x] = float(teeth_mask.at<uchar>(g_y,g_x))/255;
        }
      }

      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          transformed_label[(2*np+3)*channelOffset + g_y*grid_x + g_x] = float(teeth_mask.at<uchar>(g_y,g_x))/255;
        }
      }
    }
  } else if(meta.type == "detect") {
    for(int a = 0; a < meta.numAnns; a++){
      if(!meta.iscrowd[a]){
        putGaussianMaps(transformed_label + 0*channelOffset, meta.bboxes[a], this->param_.stride(),
                          grid_x, grid_y, this->param_.sigma()); //overloading this function
      }
    }
  }


  //visualize
  Mat label_map;
  if(1 && this->param_.visualize() && meta.type == "cpm"){
    for(int i = 0; i < 2*(np+1+mask_channel); i++){
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      //int MPI_index = MPI_to_ours[i];
      //Point2f center = meta.joint_self.joints[MPI_index];
      for (int g_y = 0; g_y < grid_y; g_y++){
        //printf("\n");
        for (int g_x = 0; g_x < grid_x; g_x++){
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
          //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

      stringstream ss1;
      if(i < np+1+mask_channel){
        if(mask[i]) {
          ss1 << "valid";
        } else {
          ss1 << "missed";
        }
      } else {
        if(mask[i-np-1-mask_channel]) {
          ss1 << "valid";
        } else {
          ss1 << "missed";
        }
      }
      string str_info = ss1.str();
      setLabel(label_map, str_info, Point(0, 20));

      //center = center * (1.0/(float)this->param_.stride());
      //circle(label_map, center, 3, CV_RGB(255,0,255), -1);
      char imagename [100];
      sprintf(imagename, "augment_writeNum_%04d_label_part_%02d.jpg", meta.write_number, i);
      imwrite(imagename, label_map);
    }

    // label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
    // for (int g_y = 0; g_y < grid_y; g_y++){
    //   //printf("\n");
    //   for (int g_x = 0; g_x < grid_x; g_x++){
    //     label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[np*channelOffset + g_y*grid_x + g_x]*255);
    //     //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
    //   }
    // }
    // resize(label_map, label_map, Size(), stride, stride, INTER_CUBIC);
    // applyColorMap(label_map, label_map, COLORMAP_JET);
    // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

    // for(int i=0;i<np;i++){
    //   Point2f center = meta.joint_self.joints[i];// * (1.0/this->param_.stride());
    //   circle(label_map, center, 3, CV_RGB(100,100,100), -1);
    // }
    // char imagename [100];
    // sprintf(imagename, "augment_%04d_label_part_back.jpg", counter);
    // //LOG(INFO) << "filename is " << imagename;
    // imwrite(imagename, label_map);
  } else if (1 && this->param_.visualize() && meta.type == "detect") {
    label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
    for (int g_y = 0; g_y < grid_y; g_y++){
      //printf("\n");
      for (int g_x = 0; g_x < grid_x; g_x++){
        label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[0*channelOffset + g_y*grid_x + g_x]*255)/2;
        //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
      }
    }
    resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
    applyColorMap(label_map, label_map, COLORMAP_JET);
    addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

    for(int a = 0; a < meta.numAnns; a++){
      if(meta.iscrowd[a]){
        rectangle(label_map, Point2f(meta.bboxes[a].left, meta.bboxes[a].top),
                             Point2f(meta.bboxes[a].left + meta.bboxes[a].width, meta.bboxes[a].top + meta.bboxes[a].height),
                             CV_RGB(255,0,0), 3);
      }
    }

    char imagename [100];
    sprintf(imagename, "augment_writeNum_%04d_epoch_%04d_label.jpg", meta.write_number, meta.epoch);
    imwrite(imagename, label_map);
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::visualize(Mat& img, MetaData meta, AugmentSelection as, Dtype* mask, Mat& teeth_mask) {
  //Mat img_vis = Mat::zeros(img.rows*2, img.cols, CV_8UC3);
  //copy image content
  // for (int i = 0; i < img.rows; ++i) {
  //   for (int j = 0; j < img.cols; ++j) {
  //     Vec3b& rgb = img.at<Vec3b>(i, j);
  //     Vec3b& rgb_vis_upper = img_vis.at<Vec3b>(i, j);
  //     rgb_vis_upper = rgb;
  //   }
  // }
  // for (int i = 0; i < img_aug.rows; ++i) {
  //   for (int j = 0; j < img_aug.cols; ++j) {
  //     Vec3b& rgb_aug = img_aug.at<Vec3b>(i, j);
  //     Vec3b& rgb_vis_lower = img_vis.at<Vec3b>(i + img.rows, j);
  //     rgb_vis_lower = rgb_aug;
  //   }
  // }
  Mat img_vis = img.clone();
  if(this->param_.has_masks()){
    Vec3b hlcolor(0,255,255);
    for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
        uchar m = teeth_mask.at<uchar>(i,j);
        if(m > 128){
          //cout << int(m) << " ";
          img_vis.at<Vec3b>(i,j) = hlcolor;
        }
      }
    }
  }

  static int counter = 0;

  if(meta.type == "cpm"){
    rectangle(img_vis, meta.objpos-Point2f(3,3), meta.objpos+Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
    for(int i=0;i<np;i++){
      //LOG(INFO) << "drawing part " << i << ": ";
      //LOG(INFO) << meta.joint_self.joints.size();
      //LOG(INFO) << meta.joint_self.joints[i];
      //if(meta.joint_self.isVisible[i])
      if(np == 21){ // hand case
        if(i < 4)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
        else if(i < 6 || i == 12 || i == 13)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else if(i < 8 || i == 14 || i == 15)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
        else if(i < 10|| i == 16 || i == 17)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,0), -1);
        else if(i < 12|| i == 18 || i == 19)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
        else
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,100,100), -1);
      }
      else if(np == 9){
        if(i==0 || i==1 || i==2 || i==6)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
        else if(i==3 || i==4 || i==5 || i==7)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
      }
      else if(np == 14 || np == 28) {//body case
        if(i < 14){
          if(i==2 || i==3 || i==4 || i==8 || i==9 || i==10)
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
          else if(i==5 || i==6 || i==7 || i==11 || i==12 || i==13)
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
          else
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
        }
        else if(i < 16)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,255,0), -1);
        else {
          if(i==17 || i==18 || i==19 || i==23 || i==24)
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
          else if(i==20 || i==21 || i==22 || i==25 || i==26)
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
          else
            circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,200,200), -1);
        }
      }
      else if(np == 24) { //eye
        if(i < 16){
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
        }
        else {
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,255,0), -1);
        }
      }
    }
    //line(img_vis, meta.objpos+Point2f(-368/2,-368/2), meta.objpos+Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
    //line(img_vis, meta.objpos+Point2f(368/2,-368/2), meta.objpos+Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
    //line(img_vis, meta.objpos+Point2f(368/2,368/2), meta.objpos+Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
    //line(img_vis, meta.objpos+Point2f(-368/2,368/2), meta.objpos+Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);

    for(int p=0;p<meta.numOtherPeople;p++){
      rectangle(img_vis, meta.objpos_other[p]-Point2f(3,3), meta.objpos_other[p]+Point2f(3,3), CV_RGB(0,255,255), CV_FILLED);
      for(int i=0;i<np;i++){
        // if(meta.joint_others[p].isVisible[i])
        //   circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,0,255), -1);
        // else
        //   circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,255,255), -1);

        //MPII R leg: 0(ankle), 1(knee), 2(hip)
        //     L leg: 5(ankle), 4(knee), 3(hip)
        //     R arms: 10(wrist), 11(elbow), 12(shoulder)
        //     L arms: 13(wrist), 14(elbow), 15(shoulder)
        //if(i==0 || i==1 || i==2 || i==10 || i==11 || i==12)
        circle(img_vis, meta.joint_others[p].joints[i], 2, CV_RGB(0,0,0), -1);
        //else if(i==5 || i==4 || i==3 || i==13 || i==14 || i==15)
          //circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,255,255), -1);
        //else
          //circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(255,255,0), -1);
      }
    }
  } else if(meta.type == "detect"){
    for(int a = 0; a < meta.numAnns; a++) {
      //Scalar clr(rand() % 255, rand() % 255, rand() % 255);
      Scalar color(255, 0, 0);

      Point lr(meta.bboxes[a].left, meta.bboxes[a].top);
      Point rb = lr + Point(meta.bboxes[a].width, meta.bboxes[a].height);
      rectangle(img_vis, lr, rb, color, 3);

      for(int i = 0; i < np; i++){
        circle(img_vis, meta.annotations[a].joints[i], 2, color, 3);
      }
    }
  }



  // draw text
  if(this->phase_ == TRAIN){
    std::stringstream ss;
    // ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: "
    //    << as.crop.height << "," << as.crop.width;
    ss << meta.dataset << " " << meta.write_number << " index:" << meta.annolist_index << "; p:" << meta.people_index
       << "; o_scale: " << meta.scale_self;
    string str_info = ss.str();
    setLabel(img_vis, str_info, Point(0, 20));

    stringstream ss2;
    ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
    str_info = ss2.str();
    setLabel(img_vis, str_info, Point(0, 40));

    int mask_legnth = meta.joint_self.isVisible.size() + 1;
    if(this->param_.has_masks())
      mask_legnth++;

    if(this->param_.has_masks()){
      stringstream ss3;
      if(mask[mask_legnth-1]) {
        ss3 << "teeth mask";
      } else {
        ss3 << "no mask";
      }
      str_info = ss3.str();
      setLabel(img_vis, str_info, Point(0, 60));
    }

    // stringstream ss4;
    // for(int i=mask_legnth/2;i<mask_legnth;i++)
    //   ss4 << mask[i] << " ";
    // str_info = ss4.str();
    // setLabel(img_vis, str_info, Point(0, 80));

    rectangle(img_vis, Point(0, 0+img_vis.rows), Point(this->param_.crop_size_x(), this->param_.crop_size_y()+img_vis.rows), Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "augment_writeNum_%04d_epoch_%04d_%04d.jpg", meta.write_number, meta.epoch, counter);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);

    // sprintf(imagename, "augment_%04d_epoch_%d_writenum_%d_mask.jpg", counter, meta.epoch, meta.write_number);
    // imwrite(imagename, teeth_mask);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, Point(0, 20));

    char imagename [100];
    sprintf(imagename, "augment_%04d.jpg", counter);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  counter++;
}

template <typename Dtype>
void CpmDataTransformer<Dtype>::clahe(Mat& bgr_image, int tileSize, int clipLimit) {
  // Mat lab_image;
  // cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // // Extract the L channel
  // vector<Mat> lab_planes(3);
  // split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // // apply the CLAHE algorithm to the L channel
  // Ptr<CLAHE> clahe = createCLAHE(clipLimit, Size(tileSize, tileSize));
  // //clahe->setClipLimit(4);
  // Mat dst;
  // clahe->apply(lab_planes[0], dst);

  // // Merge the the color planes back into an Lab image
  // dst.copyTo(lab_planes[0]);
  // merge(lab_planes, lab_image);

  // // convert back to RGB
  // Mat image_clahe;
  // cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  // bgr_image = image_clahe.clone();

  cvtColor( bgr_image, bgr_image, CV_BGR2GRAY );
  equalizeHist( bgr_image, bgr_image );
  cvtColor( bgr_image, bgr_image, CV_GRAY2BGR );
}

template <typename Dtype>
void CpmDataTransformer<Dtype>::dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData meta){

  char filename[100];
  sprintf(filename, "transformed_data_%04d_%02d", meta.annolist_index, meta.people_index);
  ofstream myfile;
  myfile.open(filename);
  int data_length = this->param_.crop_size_y() * this->param_.crop_size_x() * 4;

  //LOG(INFO) << "before copy data: " << filename << "  " << data_length;
  for(int i = 0; i<data_length; i++){
    myfile << transformed_data[i] << " ";
  }
  //LOG(INFO) << "after copy data: " << filename << "  " << data_length;
  myfile.close();

  sprintf(filename, "transformed_label_%04d_%02d", meta.annolist_index, meta.people_index);
  myfile.open(filename);
  int label_length = this->param_.crop_size_y() * this->param_.crop_size_x() / this->param_.stride() / this->param_.stride() * (this->param_.num_parts()+1);
  for(int i = 0; i<label_length; i++){
    myfile << transformed_label[i] << " ";
  }
  myfile.close();
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_CPM(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, Blob<Dtype>* mask, int cnt) {
  //std::cout << "Function 2 is used"; std::cout.flush();
  const int datum_channels = datum.channels();
  //const int datum_height = datum.height();
  //const int datum_width = datum.width();

  const int im_channels = transformed_data->channels();
  //const int im_height = transformed_data->height();
  //const int im_width = transformed_data->width();
  const int im_num = transformed_data->num();

  //const int lb_channels = transformed_label->channels();
  //const int lb_height = transformed_label->height();
  //const int lb_width = transformed_label->width();
  const int lb_num = transformed_label->num();

  //LOG(INFO) << "image shape: " << transformed_data->num() << " " << transformed_data->channels() << " "
  //                             << transformed_data->height() << " " << transformed_data->width();
  //LOG(INFO) << "label shape: " << transformed_label->num() << " " << transformed_label->channels() << " "
  //                             << transformed_label->height() << " " << transformed_label->width();

  if(this->param_.has_masks())
    CHECK_EQ(datum_channels, 5);
  else
    CHECK_EQ(datum_channels, 4);

  if(this->param_.put_gaussian())
    CHECK_EQ(im_channels, 4);
  else
    CHECK_EQ(im_channels, 3);

  CHECK_EQ(im_num, lb_num);
  //CHECK_LE(im_height, datum_height);
  //CHECK_LE(im_width, datum_width);
  CHECK_GE(im_num, 1);

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
  Dtype* mask_pointer = mask->mutable_cpu_data();

  Transform_CPM(datum, transformed_data_pointer, transformed_label_pointer, mask_pointer, cnt); //call function 1
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_CPM(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, Dtype* mask, int cnt) {
  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = this->param_.clahe_tile_size();
  int clahe_clipLimit = this->param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    Size(),
    0,
  };
  MetaData meta;

  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = this->param_.crop_size_x();
  int crop_y = this->param_.crop_size_y();

  CHECK_GT(datum_channels, 0);

  //before any transformation, get the image from datum
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }
    }
  }
  Mat teeth_mask, teeth_mask_temp1, teeth_mask_temp2, teeth_mask_temp3, teeth_mask_aug;
  if(this->param_.has_masks()){
    teeth_mask = Mat::zeros(datum_height, datum_width, CV_8UC1);
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        uchar& m = teeth_mask.at<uchar>(i, j);
        dindex = 4*offset + i*img.cols + j; //4 channel ahead is r, g, b, and metadata
        if (has_uint8)
          m = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          m = datum.float_data(dindex);
      }
    }
  }

  //color, contract
  if(this->param_.do_clahe())
    clahe(img, clahe_tileSize, clahe_clipLimit);
  if(this->param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  ReadMetaData(meta, data, offset3, offset1);
  if(this->param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta);

  //fill mask
  int all_present = 1;
  for(int i=0;i<meta.joint_self.isVisible.size();i++) {
    if(meta.joint_self.isVisible[i] <= 2){ //missed labels are 3
      mask[i] = 1;
    }
    else {
      mask[i] = 0;
      all_present = 0;
    }
  }
  mask[meta.joint_self.isVisible.size()] = all_present; //background

  if(this->param_.has_masks()){
    mask[meta.joint_self.isVisible.size()+1] = meta.has_teeth_mask;
  }

  //visualize original
  if(0 && this->param_.visualize())
    visualize(img, meta, as, mask, teeth_mask);

  //Start transforming
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  Mat img_temp, img_temp2, img_temp3; //size determined by scale
  // We only do random transform as augmentation when training.
  if (this->phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp, meta, -1);
    if(this->param_.has_masks())
      augmentation_scale(teeth_mask, teeth_mask_temp1, meta, as.scale);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp, meta, as, mask, teeth_mask_temp1);
    as.degree = augmentation_rotate(img_temp, img_temp2, meta, -999);
    if(this->param_.has_masks()){
      augmentation_rotate(teeth_mask_temp1, teeth_mask_temp2, meta, as.degree);
    }

    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp2, meta, as, mask, teeth_mask_temp2);
    if(this->param_.has_masks()){
      as.crop = augmentation_croppad(img_temp2, img_temp3, meta, Size(-999,-999), 0);
      augmentation_croppad(teeth_mask_temp2, teeth_mask_temp3, meta, as.crop, 1);
    }
    else {
      as.crop = augmentation_croppad(img_temp2, img_temp3, meta, Size(-999,-999), 1);
    }

    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp3, meta, as, mask, teeth_mask_temp3);
    as.flip = augmentation_flip(img_temp3, img_aug, meta, -1);
    if(this->param_.has_masks()){
      augmentation_flip(teeth_mask_temp3, teeth_mask_aug, meta, as.flip);
    }
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];

    if(this->param_.visualize())
      visualize(img_aug, meta, as, mask, teeth_mask_aug);
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = Size();
    as.flip = 0;
    as.degree = 0;
  }
  //LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height
  //          << "); flip:" << as.flip << "; degree: " << as.degree;

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
      transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
      transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
      if(this->param_.put_gaussian()){
        transformed_data[3*offset + i*img_aug.cols + j] = 0; //zero 4-th channel
      }
    }
  }

  if(this->param_.put_gaussian()){
    putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, this->param_.sigma_center());
    LOG(INFO) << "image transformation done!";
  }

  generateLabelMap(transformed_label, img_aug, meta, mask, teeth_mask_aug);

  //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
  //if(this->param_.visualize()){
    //dumpEverything(transformed_data, transformed_label, meta);
  //}
//}
}
// **************************** CPM Above *********************************

template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_COCO(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, Blob<Dtype>* mask, int cnt) {
  //std::cout << "Function 2 is used"; std::cout.flush();
  const int datum_channels = datum.channels();
  //const int datum_height = datum.height();
  //const int datum_width = datum.width();

  const int im_channels = transformed_data->channels();
  //const int im_height = transformed_data->height();
  //const int im_width = transformed_data->width();
  const int im_num = transformed_data->num();

  //const int lb_channels = transformed_label->channels();
  //const int lb_height = transformed_label->height();
  //const int lb_width = transformed_label->width();
  const int lb_num = transformed_label->num();

  //LOG(INFO) << "image shape: " << transformed_data->num() << " " << transformed_data->channels() << " "
  //                             << transformed_data->height() << " " << transformed_data->width();
  //LOG(INFO) << "label shape: " << transformed_label->num() << " " << transformed_label->channels() << " "
  //                             << transformed_label->height() << " " << transformed_label->width();

  if(this->param_.has_masks())
    CHECK_EQ(datum_channels, 5);
  else
    CHECK_EQ(datum_channels, 4);

  if(this->param_.put_gaussian())
    CHECK_EQ(im_channels, 4);
  else
    CHECK_EQ(im_channels, 3);

  CHECK_EQ(im_num, lb_num);
  //CHECK_LE(im_height, datum_height);
  //CHECK_LE(im_width, datum_width);
  CHECK_GE(im_num, 1);

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
  Dtype* mask_pointer = mask->mutable_cpu_data();

  Transform_COCO(datum, transformed_data_pointer, transformed_label_pointer, mask_pointer, cnt); //call function 1
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_COCO(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, Dtype* mask, int cnt) {
  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = this->param_.clahe_tile_size();
  int clahe_clipLimit = this->param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    Size(),
    0,
  };
  MetaData meta;

  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = this->param_.crop_size_x();
  int crop_y = this->param_.crop_size_y();

  CHECK_GT(datum_channels, 0);

  //before any transformation, get the image from datum
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }
    }
  }
  Mat teeth_mask, teeth_mask_temp1, teeth_mask_temp2, teeth_mask_temp3, teeth_mask_aug;
  if(this->param_.has_masks()){
    teeth_mask = Mat::zeros(datum_height, datum_width, CV_8UC1);
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
        uchar& m = teeth_mask.at<uchar>(i, j);
        dindex = 4*offset + i*img.cols + j; //4 channel ahead is r, g, b, and metadata
        if (has_uint8)
          m = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          m = datum.float_data(dindex);
      }
    }
  }

  //color, contract
  if(this->param_.do_clahe())
    clahe(img, clahe_tileSize, clahe_clipLimit);
  if(this->param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  ReadMetaData_COCO(meta, data, offset3, offset1);
  if(this->param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta);

  //fill mask
  int all_present = 1;
  for(int i=0;i<meta.joint_self.isVisible.size();i++) {
    if(meta.joint_self.isVisible[i] <= 2){ //missed labels are 3
      mask[i] = 1;
    }
    else {
      mask[i] = 0;
      all_present = 0;
    }
  }
  mask[meta.joint_self.isVisible.size()] = all_present; //background

  if(this->param_.has_masks()){
    mask[meta.joint_self.isVisible.size()+1] = meta.has_teeth_mask;
  }

  //visualize original
  if(1 && this->param_.visualize())
    visualize(img, meta, as, mask, teeth_mask);

  //Start transforming
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  Mat img_temp, img_temp2, img_temp3; //size determined by scale
  // We only do random transform as augmentation when training.
  if (this->phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp, meta, -1);
    if(this->param_.has_masks())
      augmentation_scale(teeth_mask, teeth_mask_temp1, meta, as.scale);
    //LOG(INFO) << "scale done";
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp, meta, as, mask, teeth_mask_temp1);
    as.degree = augmentation_rotate(img_temp, img_temp2, meta, -999);
    if(this->param_.has_masks()){
      augmentation_rotate(teeth_mask_temp1, teeth_mask_temp2, meta, as.degree);
    }

    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp2, meta, as, mask, teeth_mask_temp2);
    if(this->param_.has_masks()){
      as.crop = augmentation_croppad(img_temp2, img_temp3, meta, Size(-999,-999), 0);
      augmentation_croppad(teeth_mask_temp2, teeth_mask_temp3, meta, as.crop, 1);
    }
    else {
      as.crop = augmentation_croppad(img_temp2, img_temp3, meta, Size(-999,-999), 1);
    }

    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && this->param_.visualize())
      visualize(img_temp3, meta, as, mask, teeth_mask_temp3);
    as.flip = augmentation_flip(img_temp3, img_aug, meta, -1);
    if(this->param_.has_masks()){
      augmentation_flip(teeth_mask_temp3, teeth_mask_aug, meta, as.flip);
    }
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];

    if(this->param_.visualize())
      visualize(img_aug, meta, as, mask, teeth_mask_aug);
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = Size();
    as.flip = 0;
    as.degree = 0;
  }
  //LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height
  //          << "); flip:" << as.flip << "; degree: " << as.degree;

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
      transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
      transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
      if(this->param_.put_gaussian()){
        transformed_data[3*offset + i*img_aug.cols + j] = 0; //zero 4-th channel
      }
    }
  }

  if(this->param_.put_gaussian()){
    putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, this->param_.sigma_center());
    //LOG(INFO) << "image transformation done!";
  }

  generateLabelMap(transformed_label, img_aug, meta, mask, teeth_mask_aug);

  //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
  //if(this->param_.visualize()){
    //dumpEverything(transformed_data, transformed_label, meta);
  //}
//}
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_bottomup(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt) {
  //std::cout << "Function 2 is used"; std::cout.flush();
  // int offset = datum.height()*datum.width();
  // int offset3 = 3 * offset;
  // int offset1 = datum.width();
  // MetaData meta;
  // ReadMetaData(meta, datum.data(), offset3, offset1);
  // LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
  //       << "; meta.annolist_index: " << meta.annolist_index;

  const int datum_channels = datum.channels();
  //LOG(INFO) << datum.channels();
  //const int datum_height = datum.height();
  //const int datum_width = datum.width();

  const int im_channels = transformed_data->channels();
  //LOG(INFO) << im_channels;
  //const int im_height = transformed_data->height();
  //const int im_width = transformed_data->width();
  const int im_num = transformed_data->num();

  //const int lb_channels = transformed_label->channels();
  //const int lb_height = transformed_label->height();
  //const int lb_width = transformed_label->width();
  //const int lb_num = transformed_label->num();

  //LOG(INFO) << "image shape: " << transformed_data->num() << " " << transformed_data->channels() << " "
  //                             << transformed_data->height() << " " << transformed_data->width();
  //LOG(INFO) << "label shape: " << transformed_label->num() << " " << transformed_label->channels() << " "
  //                             << transformed_label->height() << " " << transformed_label->width();

  CHECK_EQ(datum_channels, 6);
  CHECK_EQ(im_channels, 3);
  //CHECK_EQ(im_channels, 4);
  //CHECK_EQ(datum_channels, 4);
  //CHECK_EQ(im_channels, 5);
  //CHECK_EQ(datum_channels, 5);

  //CHECK_EQ(im_num, lb_num);
  //CHECK_LE(im_height, datum_height);
  //CHECK_LE(im_width, datum_width);
  CHECK_GE(im_num, 1);

  //const int crop_size = this->param_.crop_size();

  // if (crop_size) {
  //   CHECK_EQ(crop_size, im_height);
  //   CHECK_EQ(crop_size, im_width);
  // } else {
  //   CHECK_EQ(datum_height, im_height);
  //   CHECK_EQ(datum_width, im_width);
  // }

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();

  Transform_bottomup(datum, transformed_data_pointer, transformed_label_pointer, cnt); //call function 1
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::ReadMetaData_bottomup(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = Size(width, height);
  float image_id;
  DecodeFloats(data, offset3+offset1+8, &image_id, 1);
  meta.image_id = (int) image_id;

  // ----------- Validation, nop, counters -----------------
  meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
  meta.numOtherPeople = (int)data[offset3+2*offset1+1];
  meta.people_index = (int)data[offset3+2*offset1+2];
  float annolist_index;
  DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
  meta.annolist_index = (int)annolist_index;
  float write_number;
  DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
  meta.write_number = (int)write_number;
  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;
  meta.num_keypoints_self = (int)data[offset3+2*offset1+15];
  DecodeFloats(data, offset3+2*offset1+16, &(meta.segmentation_area), 1);

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(meta.write_number % 1000 == 0){
    LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size <<  "; image_id: " << meta.image_id
        << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number
        << "; meta.total_write_number: " << meta.total_write_number
        << "; meta.num_keypoints_self: " << meta.num_keypoints_self << "; meta.segmentation_area: " << meta.segmentation_area
        << "; meta.epoch: " << meta.epoch;
  }
  //LOG(INFO) << "np_in_lmdb" << np_in_lmdb;
  if(this->param_.aug_way() == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }

  // ------------------- objpos and bbox -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  //meta.objpos -= Point2f(1,1);
  DecodeFloats(data, offset3+3*offset1+8, &meta.bbox.left, 1);
  DecodeFloats(data, offset3+3*offset1+12, &meta.bbox.top, 1);
  DecodeFloats(data, offset3+3*offset1+16, &meta.bbox.width, 1);
  DecodeFloats(data, offset3+3*offset1+20, &meta.bbox.height, 1);
  //meta.bbox.left -= 1;
  //meta.bbox.top -= 1;

  //LOG(INFO) << "objpos: " << meta.objpos << "; bbox: " << meta.bbox.left << "," << meta.bbox.top << ","
  //                                                     << meta.bbox.width << "," << meta.bbox.height;

  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    meta.joint_self.joints[i] -= Point2f(1,1); //from matlab 1-index to c++ 0-index
    float isVisible;
    DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
    if (isVisible == 3){
      meta.joint_self.isVisible[i] = 3;
    }
    else{
      meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;
      if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
         meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
      }
    }
    //LOG(INFO) << meta.joint_self.joints[i].x << " " << meta.joint_self.joints[i].y << " " << meta.joint_self.isVisible[i];
  }

  //others (7 lines loaded)
  meta.objpos_other.resize(meta.numOtherPeople);
  meta.bboxes_other.resize(meta.numOtherPeople);
  meta.num_keypoints_others.resize(meta.numOtherPeople);
  meta.segmentation_area_others.resize(meta.numOtherPeople);
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);

  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    //meta.objpos_other[p] -= Point2f(1,1);
    DecodeFloats(data, offset3+(8+p)*offset1+8, &meta.bboxes_other[p].left, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+12, &meta.bboxes_other[p].top, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+16, &meta.bboxes_other[p].width, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+20, &meta.bboxes_other[p].height, 1);
    //meta.bboxes_other[p].left -= 1;
    //meta.bboxes_other[p].top -= 1;

    meta.num_keypoints_others[p] = int(data[offset3+(8+p)*offset1+24]);
    DecodeFloats(data, offset3+(8+p)*offset1+25, &meta.segmentation_area_others[p], 1);

    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);

    //LOG(INFO) << "other " << p << ": objpos: " << meta.objpos_other[p];
    //LOG(INFO) << "other " << p << ": bbox: " << meta.bboxes_other[p].left << "," << meta.bboxes_other[p].top << ","
    //                                         << meta.bboxes_other[p].width << "," << meta.bboxes_other[p].height;
    //LOG(INFO) << "other " << p << ": num_keypoints: " << meta.num_keypoints_others[p];
    //LOG(INFO) << "other " << p << ": segmentation_area: " << meta.segmentation_area_others[p];
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      meta.joint_others[p].joints[i] -= Point2f(1,1);
      float isVisible;
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
      meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
      if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
         meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
        meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 1 means occluded by still on image
      }
      //LOG(INFO) << meta.joint_others[p].joints[i].x << " " << meta.joint_others[p].joints[i].y << " " << meta.joint_others[p].isVisible[i];
    }
  }

  //LOG(INFO) << "Meta read done.";
}





template<typename Dtype>
void CpmDataTransformer<Dtype>::Transform_bottomup(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt) {

  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = this->param_.clahe_tile_size();
  int clahe_clipLimit = this->param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    Size(),
    0,
  };
  MetaData meta;

  const string& data = datum.data();
  const int datum_channels = datum.channels();
  //LOG(INFO) << datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // To do: make this a parameter in caffe.proto
  // const int mode = 6; //related to datum.channels();

  //const int crop_size = this->param_.crop_size();
  //const Dtype scale = this->param_.scale();
  //const bool do_mirror = this->param_.mirror() && Rand(2);
  //const bool has_mean_file = this->param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = this->param_.crop_size_x();
  int crop_y = this->param_.crop_size_y();

  CHECK_GT(datum_channels, 0);
  //CHECK_GE(datum_height, crop_size);
  //CHECK_GE(datum_width, crop_size);

  //before any transformation, get the image from datum
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  Mat mask_all, mask_miss;
  //if(mode >= 5){
    mask_miss = Mat::ones(datum_height, datum_width, CV_8UC1);
  //}
  //if(mode == 6){
    mask_all = Mat::zeros(datum_height, datum_width, CV_8UC1);
  //}

  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }

      //if(mode >= 5){
        dindex = 4*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        // if (round(d_element/255)!=1 && round(d_element/255)!=0){
        //   cout << d_element << " " << round(d_element/255) << endl;
        // }
        mask_miss.at<uchar>(i, j) = d_element; //round(d_element/255);
      //}

      //if(mode == 6){
        dindex = 5*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        mask_all.at<uchar>(i, j) = d_element;
      //}
    }
  }

  //testing image
  //imshow("mask_miss",mask_miss);
  //imshow("mask_all",mask_all);
  // if(mode >= 5){
  //   Mat erosion_dst;
  //   int erosion_size = 1;
  //   mask_miss = 1.0/255 *mask_miss;
  //   Mat element = getStructuringElement( MORPH_ELLIPSE,
  //                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
  //                                    Point( erosion_size, erosion_size ) );
  //   erode( mask_miss, erosion_dst, element );
  //   erosion_dst = 255 *erosion_dst;
  //   imshow( "Erosion Demo", erosion_dst );
  // }


  //color, contract
  if(this->param_.do_clahe())
    clahe(img, clahe_tileSize, clahe_clipLimit);
  if(this->param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  int stride = this->param_.stride();
  ReadMetaData_bottomup(meta, data, offset3, offset1);

  if(this->param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta); //when np = 56, np_in_lmdb becomes 18 from 17 here

  //visualize original
  if(1 && this->param_.visualize()) {
    visualize(img, meta, as, mask_all);
  }

  //Start transforming
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  Mat mask_miss_aug;
  Mat mask_all_aug;

  Mat img_temp1, img_temp2, img_temp3; //size determined by scale

  // We only do random transform as augmentation when training.
  if (this->phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp1, mask_miss, mask_all, meta);
    if(0 && this->param_.visualize())
      visualize(img_temp1, meta, as, mask_all);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    as.degree = augmentation_rotate(img_temp1, img_temp2, mask_miss, mask_all, meta);
    if(0 && this->param_.visualize())
      visualize(img_temp2, meta, as, mask_all);

    as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, mask_all, mask_all_aug, meta);
    if(0 && this->param_.visualize())
      visualize(img_temp3, meta, as, mask_all_aug);

    as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, mask_all_aug, meta);
    if(1 && this->param_.visualize())
      visualize(img_aug, meta, as, mask_all_aug);

    // imshow("img_aug", img_aug);
    // Mat label_map = mask_miss_aug;
    // applyColorMap(label_map, label_map, COLORMAP_JET);
    // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
    // imshow("mask_miss_aug", label_map);

    // if (mode > 4){
    if(!this->param_.analysis()){
      resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
      resize(mask_all_aug, mask_all_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
    // }
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = Size();
    as.flip = 0;
    as.degree = 0;
  }

  if(this->param_.analysis())
    writeAugAnalysis(meta);

  //LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height
  //          << "); flip:" << as.flip << "; degree: " << as.degree;

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  if(!this->param_.analysis()){

    for (int i = 0; i < img_aug.rows; ++i) {
      for (int j = 0; j < img_aug.cols; ++j) {
        Vec3b& rgb = img_aug.at<Vec3b>(i, j);
        transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
        transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
        transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
      }
    }

    // label size is image size/ stride
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        for (int i = 0; i < np+1; i++){ // for first np+1 (=56) channels, fill all the same mask_miss in
          // To do
          // if (mode = 4){
          //   transformed_label[i*channelOffset + g_y*grid_x + g_x] = 1;
          // }
          //if(mode > 4){
            float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) / 255; //mask_miss_aug.at<uchar>(i, j);
            if (meta.joint_self.isVisible[i] != 3){
              transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
            }
            else {
              transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0; //truncated part?
            }
          //}
        }
        // background channel
        //To do: if (mode = 4){
        //if(mode == 5){
        //  transformed_label[np*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255;
       // }
        //if(mode > 5){
        //  transformed_label[np*channelOffset + g_y*grid_x + g_x] = 1;
        //  transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = float(mask_all_aug.at<uchar>(g_y, g_x)) /255;
       // }
      }
    }

    //LOG(INFO) << "Before generating label map";
    //putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, this->param_.sigma_center());
    //LOG(INFO) << "image transformation done!";
    generateLabelMap(transformed_label, img_aug, meta);

    //LOG(INFO) << "After generating label map";
    //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
    //if(this->param_.visualize()){
      //dumpEverything(transformed_data, transformed_label, meta);
    //}
  }
}

// include mask_miss
template<typename Dtype>
float CpmDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp,
                                                 Mat& mask_miss, Mat& mask_all,
                                                 MetaData& meta) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  //float scale = (this->param_.scale_max() - this->param_.scale_min()) * dice + this->param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice > this->param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (this->param_.scale_max() - this->param_.scale_min()) * dice2 + this->param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs;
  if(!this->param_.use_segmentation_scale()){
    //scale_abs = this->param_.target_dist()/meta.scale_self;
    meta.scale_self = max(meta.bbox.width, meta.bbox.height) / 368;
    scale_abs = this->param_.target_dist() / meta.scale_self;
  } else {
    scale_abs = this->param_.target_dist() / sqrt(meta.segmentation_area); //bad.
  }

  float scale = scale_abs * scale_multiplier;

  if(!this->param_.analysis()){
    resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);

    // if(mode>4){
    //   resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);
    // }
    // if(mode>5){
    //   resize(mask_all, mask_all, Size(), scale, scale, INTER_CUBIC);
    // }
    resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);
    resize(mask_all, mask_all, Size(), scale, scale, INTER_CUBIC);
  }

  //modify meta data
  meta.objpos *= scale;
  for(int i = 0; i < np_in_lmdb; i++){
    meta.joint_self.joints[i] *= scale;
  }
  meta.segmentation_area *= (scale * scale);
  meta.scale_self *= scale;

  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i < np_in_lmdb; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
    meta.segmentation_area_others[p] *= (scale * scale);
    meta.scale_other[p] *= scale;
  }
  return scale_multiplier;
}

template<typename Dtype>
Size CpmDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = this->param_.crop_size_x();
  int crop_y = this->param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * this->param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * this->param_.center_perterb_max());

  //LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);

  if(!this->param_.analysis()){
    img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
    mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(255); //Scalar(1);
    mask_all_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
    for(int i=0;i<crop_y;i++){
      for(int j=0;j<crop_x;j++){ //i,j on cropped
        int coord_x_on_img = center.x - crop_x/2 + j;
        int coord_y_on_img = center.y - crop_y/2 + i;
        if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
          img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
          //if(mode>4){
            mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
          //}
          //if(mode>5){
            mask_all_aug.at<uchar>(i,j) = mask_all.at<uchar>(coord_y_on_img, coord_x_on_img);
          //}
        }
      }
    }
  }

  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i = 0; i < np_in_lmdb; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i = 0; i < np_in_lmdb; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return Size(x_offset, y_offset);
}

template<typename Dtype>
bool CpmDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta) {
  bool doflip;
  if(this->param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= this->param_.flip_prob());
  }
  else if(this->param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % this->param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  if(doflip){

    if(!this->param_.analysis()){
      flip(img_src, img_aug, 1);

    //if(mode>4){
      flip(mask_miss, mask_miss, 1);
    //}
    //if(mode>5){
      flip(mask_all, mask_all, 1);
    //}
    }

    int w = img_src.cols;
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i = 0; i<np_in_lmdb; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(this->param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p = 0; p < meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i = 0; i < np_in_lmdb; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(this->param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
float CpmDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_all, MetaData& meta) {

  float degree;
  if(this->param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * this->param_.max_rotate_degree();
  }
  else if(this->param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % this->param_.num_total_augs()];
  }
  else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";"
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  if(!this->param_.analysis()){
    warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  //if(mode >4){
    warpAffine(mask_miss, mask_miss, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255)); //Scalar(1));
  //}
  //if(mode >5){
    warpAffine(mask_all, mask_all, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
  //}
  }

  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np_in_lmdb; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np_in_lmdb; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}
// end here


template<typename Dtype>
void CpmDataTransformer<Dtype>::putVecPeaks(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB * 0.125;
  centerA = centerA * 0.125;
  Point2f bc = centerB - centerA;
  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x / norm_bc;
  bc.y = bc.y / norm_bc;

  for(int j=0;j<3;j++){
    //Point2f center = centerB*0.5 + centerA*0.5;
    Point2f center = centerB*0.5*j + centerA*0.5*(2-j);

    int min_x = std::max( int(floor(center.x-thre)), 0);
    int max_x = std::min( int(ceil(center.x+thre)), grid_x);

    int min_y = std::max( int(floor(center.y-thre)), 0);
    int max_y = std::min( int(ceil(center.y+thre)), grid_y);

    for (int g_y = min_y; g_y < max_y; g_y++){
      for (int g_x = min_x; g_x < max_x; g_x++){
        float dist = (g_x-center.x)*(g_x-center.x) + (g_y-center.y)*(g_y-center.y);
        if(dist <= thre){
          int cnt = count.at<uchar>(g_y, g_x);
          //LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
          if (cnt == 0){
            entryX[g_y*grid_x + g_x] = bc.x;
            entryY[g_y*grid_x + g_x] = bc.y;
          }
          else{
            entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
            entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
            count.at<uchar>(g_y, g_x) = cnt + 1;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre, float peak_ratio){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  Point2f bc = centerB - centerA;
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

  int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x / norm_bc;
  bc.y = bc.y / norm_bc;

  // float x_p = (centerA.x + centerB.x) / 2;
  // float y_p = (centerA.y + centerB.y) / 2;
  // float angle = atan2f(centerB.y - centerA.y, centerB.x - centerA.x);
  // float sine = sinf(angle);
  // float cosine = cosf(angle);
  // float a_sqrt = (centerA.x - x_p) * (centerA.x - x_p) + (centerA.y - y_p) * (centerA.y - y_p);
  // float b_sqrt = 10; //fixed

  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      float dist = std::abs(ba.x*bc.y - ba.y*bc.x);

      // float A = cosine * (g_x - x_p) + sine * (g_y - y_p);
      // float B = sine * (g_x - x_p) - cosine * (g_y - y_p);
      // float judge = A * A / a_sqrt + B * B / b_sqrt;

      if(dist <= thre){
      //if(judge <= 1){
        int cnt = count.at<uchar>(g_y, g_x);
        //LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
        if (cnt == 0){
          entryX[g_y*grid_x + g_x] = bc.x * peak_ratio;
          entryY[g_y*grid_x + g_x] = bc.y * peak_ratio;
        }
        else{
          entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
          entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
      }
    }
  }
}


template<typename Dtype>
void CpmDataTransformer<Dtype>::visualize(Mat& img, MetaData meta, AugmentSelection as, Mat& coco_mask) {

  Mat img_vis = img.clone();
  static int counter = 0;

  rectangle(img_vis, meta.objpos-Point2f(2,2), meta.objpos+Point2f(2,2), CV_RGB(255,255,0), CV_FILLED);
  //rectangle(img_vis, Point2f(meta.bbox.left, meta.bbox.right),
  //                   Point2f(meta.bbox.left, meta.bbox.right) + Point2f(meta.bbox.width, meta.bbox.height),
  //                   CV_RGB(255,255,255), 1);

  for(int i = 0;i < np; i++){
    if(i < 17 && meta.joint_self.isVisible[i] >= 1) {
      circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
    }
  }

  line(img_vis, meta.objpos+Point2f(-368/2,-368/2), meta.objpos+Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(368/2,-368/2), meta.objpos+Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(368/2,368/2), meta.objpos+Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(-368/2,368/2), meta.objpos+Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);

  for(int p = 0; p < meta.numOtherPeople; p++){
    rectangle(img_vis, meta.objpos_other[p]-Point2f(2,2), meta.objpos_other[p]+Point2f(2,2), CV_RGB(0,255,255), CV_FILLED);
    for(int i = 0; i < 17; i++) {
      if(meta.joint_others[p].isVisible[i] <= 1) {
        circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,0,255), -1);
      }
    }
  }

  // draw text
  if(this->phase_ == TRAIN){
    std::stringstream ss;
    // ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: "
    //    << as.crop.height << "," << as.crop.width;
    ss << meta.dataset << " " << meta.write_number << " index:" << meta.annolist_index << "; p:" << meta.people_index
       << "; o_scale: " << meta.scale_self;
    string str_info = ss.str();
    setLabel(img_vis, str_info, Point(0, 20));

    stringstream ss2;
    ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
    str_info = ss2.str();
    setLabel(img_vis, str_info, Point(0, 40));

    rectangle(img_vis, Point(0, 0+img_vis.rows), Point(this->param_.crop_size_x(), this->param_.crop_size_y()+img_vis.rows), Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "augment_%04d_epoch_%d_writenum_%d.jpg", counter, meta.epoch, meta.write_number);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);

    //char imagename [100];
    sprintf(imagename, "augment_%04d_epoch_%d_writenum_%d_mask.jpg", counter, meta.epoch, meta.write_number);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, coco_mask);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, Point(0, 20));

    char imagename [100];
    sprintf(imagename, "augment_%04d.jpg", counter);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  counter++;
}


template<typename Dtype>
void CpmDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = this->param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;
  int mode = 6; // TO DO: make this as a parameter

  // TO DO: in transform_nv, generate the weight Map for MPI images
  // clear out transformed_label, it may remain things for last batch
  // for (int g_y = 0; g_y < grid_y; g_y++){
  //   for (int g_x = 0; g_x < grid_x; g_x++){
  //     for (int i = 0; i < np; i++){
  //       if (meta.joint_self.isVisible[i] == 3){
  //         transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
  //       }
  //       else{
  //         transformed_label[i*channelOffset + g_y*grid_x + g_x] = 1;
  //       }
  //     }
  //     //background channel weight map
  //     if (meta.joint_self.isVisible[0] == 3){
  //       transformed_label[np*channelOffset + g_y*grid_x + g_x] = 0;
  //     }
  //     else{
  //       transformed_label[np*channelOffset + g_y*grid_x + g_x] = 1;
  //     }
  //   }
  // }

  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = np+1; i < 2*(np+1); i++){
        //if (mode == 6 && i == (2*np + 1))
          //continue;
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }

  //LOG(INFO) << "label cleaned";

  if (np == 37){
    for (int i = 0; i < 18; i++){
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma()); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, this->param_.stride(),
                          grid_x, grid_y, this->param_.sigma());
        }
      }
    }

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};

    for(int i=0;i<19;i++){
      for (int j=1;j<=3;j++){
        Joints jo = meta.joint_self;
        if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
          Point2f center = jo.joints[mid_1[i]-1]*(1-j*0.25) + jo.joints[mid_2[i]-1]*j*0.25;
          putGaussianMaps(transformed_label + (np+19+i)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma()); //self
        }

        for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
          Joints jo2 = meta.joint_others[j];
          if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
            Point2f center = jo2.joints[mid_1[i]-1]*(1-j*0.25) + jo2.joints[mid_2[i]-1]*j*0.25;
            putGaussianMaps(transformed_label + (np+19+i)*channelOffset, center, this->param_.stride(),
                            grid_x, grid_y, this->param_.sigma());
          }
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+1; i < 2*np+1; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
  }
  else if (np == 56){ // fill last 57 of the 114 channels here: (np+1) channels are already filled
                      //(19*2 x,y vec + 18+1 parts), but 114 channel to fill?

    // vec maps
    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    int thre = 1;

    for(int i = 0; i < 19; i++){
      // if (i>14){
      //   thre = 1;
      // }
      float peak_ratio = 1;
      if(this->param_.selective_scale()){
        peak_ratio = exp(-(meta.scale_self/this->param_.target_dist() - 1) * (meta.scale_self/this->param_.target_dist() - 1) / 0.1086);
      }

      Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
        //putVecPeaks
        putVecMaps(transformed_label + (np+1+2*i)*channelOffset, transformed_label + (np+1+(2*i+1))*channelOffset,
                  count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], this->param_.stride(), grid_x, grid_y, this->param_.sigma(), thre,
                  peak_ratio); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        peak_ratio = 1;
        if(this->param_.selective_scale()){
          peak_ratio = exp(-(meta.scale_other[j]/this->param_.target_dist() - 1) * (meta.scale_other[j]/this->param_.target_dist() - 1) / 0.1086);
        }

        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
          //putVecPeaks
          putVecMaps(transformed_label + (np+1+2*i)*channelOffset, transformed_label + (np+1+(2*i+1))*channelOffset,
                  count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], this->param_.stride(), grid_x, grid_y, this->param_.sigma(), thre,
                  peak_ratio); //self
        }
      }
    }



    for (int i = 0; i < 18; i++){
      Point2f center = meta.joint_self.joints[i];
      float peak_ratio = 1;
      if(this->param_.selective_scale()){
        peak_ratio = exp(-(meta.scale_self/this->param_.target_dist() - 1) * (meta.scale_self/this->param_.target_dist() - 1) / 0.1086);
      }
      if(meta.joint_self.isVisible[i] <= 1) {
        putGaussianMaps(transformed_label + (np+1+38+i)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma(), peak_ratio); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        peak_ratio = 1;
        if(this->param_.selective_scale()){
          peak_ratio = exp(-(meta.scale_other[j]/this->param_.target_dist() - 1) * (meta.scale_other[j]/this->param_.target_dist() - 1) / 0.1086);
          //LOG(INFO) << meta.scale_other[j] << "   " << this->param_.target_dist() << "   " << peak_ratio;
        }

        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1) {
          putGaussianMaps(transformed_label + (np+1+38+i)*channelOffset, center, this->param_.stride(),
                          grid_x, grid_y, this->param_.sigma(), peak_ratio);
        }
      }
    }

    if(!this->param_.per_part_scale()){
      //put background channel
      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          float maximum = 0;
          for (int i = np+1+38; i < np+1+38+18; i++){
            maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
          }
          transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0); // last ch
        }
      }
    } else {
      for(int i = 0; i < 18; i++){
        Point2f center = meta.joint_self.joints[i];
        float scale_normalize = meta.scale_self - 0.6; // scale_self will be around 0
        if(meta.joint_self.isVisible[i] <= 1) {
          putScaleMaps(transformed_label + (np+1+38+18)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma(), scale_normalize); //self
        }

        for(int j = 0; j < meta.numOtherPeople; j++){
          center = meta.joint_others[j].joints[i];
          scale_normalize = meta.scale_other[j] - 0.6;
          if(meta.joint_others[j].isVisible[i] <= 1){
            putScaleMaps(transformed_label + (np+1+38+18)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma(), scale_normalize); //self
          }
        }

      }
    }
    //LOG(INFO) << "background put";
  }
  else{
    for (int i = 0; i < np; i++){
      //LOG(INFO) << i << meta.numOtherPeople;
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, this->param_.stride(),
                        grid_x, grid_y, this->param_.sigma()); //self
      }
      //LOG(INFO) << "label put for" << i;
      //plot others
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, this->param_.stride(),
                          grid_x, grid_y, this->param_.sigma());
        }
      }
    }

    //put background channel
    if (mode != 6){ // mode = 6, use the mask_all as the background
      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          if (meta.joint_self.isVisible[0] == 3){
            transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = 0;
          }
          else{
            float maximum = 0;
            //second background channel
            for (int i = np+1; i < 2*np+1; i++){
              maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
            }
            transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = maximum; //max(1.0-maximum, 0.0);
          }
        }
      }
    }
    //LOG(INFO) << "background put";
  }

  //visualize
  if(1 && this->param_.visualize()){
    Mat label_map;
    for(int i = 2*(np+1)-1; i < 2*(np+1); i++){
    //for(int i = 0; i < 2*(np+1); i++){
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      //int MPI_index = MPI_to_ours[i];
      //Point2f center = meta.joint_self.joints[MPI_index];
      for (int g_y = 0; g_y < grid_y; g_y++){
        //printf("\n");
        for (int g_x = 0; g_x < grid_x; g_x++){
          //label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
          label_map.at<uchar>(g_y,g_x) = (int)((0.6+transformed_label[i*channelOffset + g_y*grid_x + g_x])*100);
          //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_CUBIC);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

      //center = center * (1.0/(float)this->param_.stride());
      //circle(label_map, center, 3, CV_RGB(255,0,255), -1);
      char imagename [100];
      sprintf(imagename, "augment_%04d_label_part_%02d.jpg", meta.write_number, i);
      //LOG(INFO) << "filename is " << imagename;
      imwrite(imagename, label_map);
    }

    // label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
    // for (int g_y = 0; g_y < grid_y; g_y++){
    //   //printf("\n");
    //   for (int g_x = 0; g_x < grid_x; g_x++){
    //     label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[np*channelOffset + g_y*grid_x + g_x]*255);
    //     //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
    //   }
    // }
    // resize(label_map, label_map, Size(), stride, stride, INTER_CUBIC);
    // applyColorMap(label_map, label_map, COLORMAP_JET);
    // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

    // for(int i=0;i<np;i++){
    //   Point2f center = meta.joint_self.joints[i];// * (1.0/this->param_.stride());
    //   circle(label_map, center, 3, CV_RGB(100,100,100), -1);
    // }
    // char imagename [100];
    // sprintf(imagename, "augment_%04d_label_part_back.jpg", counter);
    // //LOG(INFO) << "filename is " << imagename;
    // imwrite(imagename, label_map);
  }
}

template<typename Dtype>
void CpmDataTransformer<Dtype>::writeAugAnalysis(MetaData& meta){
  //self
  int num_keypoints = 0;
  for(int i = 0;i < 18; i++){
    //LOG(INFO) << meta.joint_self.joints[i] << " " << meta.joint_self.isVisible[i];
    if(onPlane(meta.joint_self.joints[i], Size(this->param_.crop_size_x(), this->param_.crop_size_y())) &&
       meta.joint_self.isVisible[i] >= 1) {
      num_keypoints++;
    }
  }
  if(num_keypoints != 0){
    ofs_analysis << "0" << "\t" << meta.segmentation_area << "\t"
                 << meta.scale_self << "\t"
                 << num_keypoints << endl;
  }

  //others
  for(int p=0;p<meta.numOtherPeople; p++){
    num_keypoints = 0;
    for(int i = 0;i < 18; i++){
      if(onPlane(meta.joint_others[p].joints[i], Size(this->param_.crop_size_x(), this->param_.crop_size_y())) &&
         meta.joint_others[p].isVisible[i] >= 1) {
        num_keypoints++;
      }
    }
    if(num_keypoints != 0){
        ofs_analysis << "1" << "\t" << meta.segmentation_area_others[p] << "\t"
                     << meta.scale_other[p] << "\t"
                     << num_keypoints << endl;
    }
  }
}

INSTANTIATE_CLASS(CpmDataTransformer);

}  // namespace caffe
