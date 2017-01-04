#ifndef CAFFE_CPM_CPM_DATA_TRANSFORMER_HPP
#define CAFFE_CPM_CPM_DATA_TRANSFORMER_HPP

#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/data_transformer.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class CpmDataTransformer : public DataTransformer<Dtype> {
 public:
  explicit CpmDataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~CpmDataTransformer() {ofs_analysis.close();}

  void Transform_CPM(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, Blob<Dtype>* mask, int cnt); //image and label
  void Transform_COCO(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, Blob<Dtype>* mask, int cnt); //image and label
  void Transform_bottomup(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt);

protected:
  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  struct Joints {
    vector<Point2f> joints;
    vector<int> isVisible;
  };

  struct Bbox {
    float left;
    float top;
    float width;
    float height;
  };

  struct MetaData {
    string type; //"cpm" or "detect"

    string dataset;
    Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)
    bool has_teeth_mask;
    int image_id;
    int num_keypoints_self;
    float segmentation_area;
    Bbox bbox;

    vector<Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople
    vector<Bbox> bboxes_other;
    vector<int> num_keypoints_others;
    vector<float> segmentation_area_others;

    // uniquely used by coco data (detect)
    int numAnns;
    int image_id_in_coco;
    int image_id_in_training;
    vector<int> num_keypoints;
    vector<bool> iscrowd;
    vector<Bbox> bboxes;
    vector<Joints> annotations;
  };

  int np_in_lmdb;
  int np;
  bool is_table_set;
  vector<vector<float> > aug_degs;
  vector<vector<int> > aug_flips;
  std::ofstream ofs_analysis;

  void Transform_CPM(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, Dtype* mask, int cnt);
  void Transform_COCO(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, Dtype* mask, int cnt);
  void ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1);
  void ReadMetaData_COCO(MetaData& meta, const string& data, size_t offset3, size_t offset1);
  void TransformMetaJoints(MetaData& meta);
  void TransformJoints(Joints& joints);
  void clahe(Mat& img, int, int);
  void putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma, float peak_ratio=1.0);
  void putGaussianMaps(Dtype* entry, Bbox b, int stride, int grid_x, int grid_y, float sigma_0);
  void dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData);
  void generateLabelMap(Dtype*, Mat&, MetaData meta, Dtype* mask, Mat&);
  void visualize(Mat& img, MetaData meta, AugmentSelection as, Dtype* mask, Mat& teeth_mask);
  void visualize(Mat& img, MetaData meta, AugmentSelection as, Mat& coco_mask);

  void putScaleMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma, float scale_normalize);

  int augmentation_flip(Mat& img, Mat& img_aug, MetaData& meta, int);
  float augmentation_rotate(Mat& img_src, Mat& img_aug, MetaData& meta, float);
  float augmentation_scale(Mat& img, Mat& img_temp, MetaData& meta, float);
  Size augmentation_croppad(Mat& img_temp, Mat& img_aug, MetaData& meta, Size, bool);
  void RotatePoint(Point2f& p, Mat R);
  bool onPlane(Point p, Size img_size);
  void swapLeftRight(Joints& j);
  void SetAugTable(int numData);

  //for cpmbottomup layer
  void Transform_bottomup(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt);
  void ReadMetaData_bottomup(MetaData& meta, const string& data, size_t offset3, size_t offset1);
  //overloading for bottomup layer
  bool augmentation_flip(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta);
  float augmentation_rotate(Mat& img_src, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta);
  float augmentation_scale(Mat& img, Mat& img_temp, Mat&, Mat&, MetaData& meta);
  Size augmentation_croppad(Mat& img_temp, Mat& img_aug, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta);
  void generateLabelMap(Dtype*, Mat&, MetaData meta);

  void putVecMaps(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre, float peak_ratio=1.0);
  void putVecPeaks(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);

  //utility
  void DecodeFloats(const string& data, size_t idx, float* pf, size_t len);
  string DecodeString(const string& data, size_t idx);
  void writeAugAnalysis(MetaData& meta);
};

}  // namespace caffe

#endif  // CAFFE_CPM_CPM_DATA_TRANSFORMER_HPP_
