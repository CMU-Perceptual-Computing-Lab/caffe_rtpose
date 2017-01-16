#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

// CPM extra code: extra includes
#include <opencv2/opencv.hpp>

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

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
   * transform_param block to a vector of cv::Mat.
   *
   * @param mat_vector
   *    A vector of cv::Mat containing the data to be transformed.
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
   *    A vector of cv::Mat containing the data to be transformed.
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

  void Transform(const Datum& datum, Dtype* transformed_data);
  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;

// CPM extra code: public and protected methods/structs/etc below
public:
  void Transform_nv(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob, int cnt); //image and label

  struct AugmentSelection {
    bool flip;
    float degree;
    cv::Size crop;
    float scale;
  };

  struct Joints {
    vector<cv::Point2f> joints;
    vector<int> isVisible;
  };

  struct MetaData {
    string dataset;
    cv::Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    cv::Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)

    vector<cv::Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople
  };

  void generateLabelMap(Dtype*, cv::Mat&, MetaData meta);
  void visualize(cv::Mat& img, MetaData meta, AugmentSelection as);

  bool augmentation_flip(cv::Mat& img, cv::Mat& img_aug, MetaData& meta);
  float augmentation_rotate(cv::Mat& img_src, cv::Mat& img_aug, MetaData& meta);
  float augmentation_scale(cv::Mat& img, cv::Mat& img_temp, MetaData& meta);
  cv::Size augmentation_croppad(cv::Mat& img_temp, cv::Mat& img_aug, MetaData& meta);

  bool augmentation_flip(cv::Mat& img, cv::Mat& img_aug, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode);
  float augmentation_rotate(cv::Mat& img_src, cv::Mat& img_aug, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode);
  float augmentation_scale(cv::Mat& img, cv::Mat& img_temp, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode);
  cv::Size augmentation_croppad(cv::Mat& img_temp, cv::Mat& img_aug, cv::Mat& mask_miss, cv::Mat& mask_miss_aug, cv::Mat& mask_all, cv::Mat& mask_all_aug, MetaData& meta, int mode);

  void RotatePoint(cv::Point2f& p, cv::Mat R);
  bool onPlane(cv::Point p, cv::Size img_size);
  void swapLeftRight(Joints& j);
  void SetAugTable(int numData);

  int np_in_lmdb;
  int np;
  bool is_table_set;
  vector<vector<float> > aug_degs;
  vector<vector<int> > aug_flips;

protected:
  void Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt);
  void ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1);
  void TransformMetaJoints(MetaData& meta);
  void TransformJoints(Joints& joints);
  void clahe(cv::Mat& img, int, int);
  void putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);
  void putVecPeaks(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);
  void dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData);
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
