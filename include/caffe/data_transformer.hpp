#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {ofs_analysis.close();}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
  void Transform_CPM(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, Blob<Dtype>* mask, int cnt); //image and label
  void Transform_COCO(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, Blob<Dtype>* mask, int cnt); //image and label
  void Transform_bottomup(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt);
  
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

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

  void Transform(const Datum& datum, Dtype* transformed_data);
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
  std::ofstream ofs_analysis;

  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
