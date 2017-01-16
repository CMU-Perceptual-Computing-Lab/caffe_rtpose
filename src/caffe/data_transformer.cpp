#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// CPM extra code: extra includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
// CPM end extra code

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  // CPM extra code:
  LOG(INFO) << "DataTransformer constructor done.";
  np_in_lmdb = param_.np_in_lmdb();
  LOG(INFO) << "np_in_lmdb" << np_in_lmdb;
  np = param_.num_parts();
  is_table_set = false;
  // CPM end extra code
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);





// CPM extra code: public and protected methods/structs/etc below
template<typename Dtype>
void DecodeFloats(const string& data, size_t idx, Dtype* pf, size_t len) {
  memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
}

string DecodeString(const string& data, size_t idx) {
  string result = "";
  int i = 0;
  while(data[idx+i] != 0){
    result.push_back(char(data[idx+i]));
    i++;
  }
  return result;
}

template<typename Dtype>
void DataTransformer<Dtype>::ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = cv::Size(width, height);
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
  //LOG(INFO) << "np_in_lmdb" << np_in_lmdb;
  if(param_.aug_way() == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }

  // ------------------- objpos -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  meta.objpos -= cv::Point2f(1,1);
  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    meta.joint_self.joints[i] -= cv::Point2f(1,1); //from matlab 1-index to c++ 0-index
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
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);
  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    meta.objpos_other[p] -= cv::Point2f(1,1);
    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      meta.joint_others[p].joints[i] -= cv::Point2f(1,1);
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
}

template<typename Dtype>
void DataTransformer<Dtype>::SetAugTable(int numData){
  aug_degs.resize(numData);     
  aug_flips.resize(numData);  
  for(int i = 0; i < numData; i++){
    aug_degs[i].resize(param_.num_total_augs());
    aug_flips[i].resize(param_.num_total_augs());
  }
  //load table files
  char filename[100];
  sprintf(filename, "../../rotate_%d_%d.txt", param_.num_total_augs(), numData);
  std::ifstream rot_file(filename);
  char filename2[100];
  sprintf(filename2, "../../flip_%d_%d.txt", param_.num_total_augs(), numData);
  std::ifstream flip_file(filename2);

  for(int i = 0; i < numData; i++){
    for(int j = 0; j < param_.num_total_augs(); j++){
      rot_file >> aug_degs[i][j];
      flip_file >> aug_flips[i][j];
    }
  }
  //debug
  // for(int i = 0; i < numData; i++){
  //   for(int j = 0; j < param_.num_total_augs(); j++){
  //     printf("%d ", (int)aug_degs[i][j]);
  //   }
  //   printf("\n");
  // }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformJoints(Joints& j) {
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
  else if(np == 27){
    int MPI_to_ours_1[27] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 8, 8, \
                             9, 8,12,11, 8,13,14, 2, 1, 3, 4};
                          //17,18,19,20,21,22,23,24,25,26,27
    int MPI_to_ours_2[27] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 2, 3, \
                             8,12,11,10,13,14,15, 1, 0, 4, 5};
                          //17,18,19,20,21,22,23,24,25,26,27
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
  else if(np == 29){
    int MPI_to_ours_1[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             9, 8,12,11, 8,13,14, 2, 1, 3, 4, 6};
    int MPI_to_ours_2[28] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7, 6, \
                             8,12,11,10,13,14,15, 1, 0, 4, 5, 7};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<np-1;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }

    jo.joints[28] = jo.joints[27];
    jo.isVisible[28] = jo.isVisible[27];
    int corr_1[3] = {8, 11, 1};
    int change[3] = {14, 15, 27};
    for(int i=0;i<3;i++){
      jo.joints[change[i]] = (jo.joints[corr_1[i]] + jo.joints[28]) * 0.5; 
      if(jo.isVisible[corr_1[i]]==2 || jo.isVisible[28]==2){
        jo.isVisible[change[i]] = 2;
      }
      else {
        jo.isVisible[change[i]] = jo.isVisible[corr_1[i]] && jo.isVisible[28];
      }
    } 
  }

  else if(np == 33){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;

    int mid_1[15] = {1, 3,  3, 4, 6,  6, 7, 9,  10, 12, 13, 3, 6,  15, 16};
    int mid_2[15] = {2, 17, 4, 5, 18, 7, 8, 10, 11, 13, 14, 9, 12, 17, 18};

    for(int i=0;i<15;i++){
      if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
        jo.isVisible[i + 18] = 2;
      }
      else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i + 18] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
      }

      jo.joints[i + 18] = jo.joints[mid_1[i]-1]*0.5 + jo.joints[mid_2[i]-1]*0.5;
    } 
  }

  else if(np == 34){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;

    int mid_1[16] = {15, 3,  3, 4, 6,  6, 7, 9,  10, 12, 13, 9,  2,  15, 16, 1};
    int mid_2[16] = {16, 17, 4, 5, 18, 7, 8, 10, 11, 13, 14, 12, 30, 17, 18, 2};

    for(int i=0;i<16;i++){
      if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
        jo.isVisible[i + 18] = 2;
      }
      else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i + 18] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
      }

      jo.joints[i + 18] = jo.joints[mid_1[i]-1]*0.5 + jo.joints[mid_2[i]-1]*0.5;
    } 
  }

  else if(np == 36){
    int num_kpt = 8;
    int COCO_to_ours[8] = {7,6,9,8,11,10,13,12};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<num_kpt;i++){
      jo.joints[i] = j.joints[COCO_to_ours[i]-1];
      jo.isVisible[i] = j.isVisible[COCO_to_ours[i]-1];
    }
    
    int cnt = num_kpt;
    for(int i=1;i<num_kpt;i++){
      for(int j=i+1;j<=num_kpt;j++){
        if(jo.isVisible[i-1]==2 || jo.isVisible[j-1]==2){
          jo.isVisible[cnt] = 2;
        }
        else if(jo.isVisible[i-1]==3 || jo.isVisible[j-1]==3){
          jo.isVisible[cnt] = 3;
        }
        else {
          jo.isVisible[cnt] = jo.isVisible[i-1] && jo.isVisible[j-1];
        }

        jo.joints[cnt] = jo.joints[i-1]*0.5 + jo.joints[j-1]*0.5;
        cnt = cnt + 1;
        //std::cout << i << " " << j << " " << cnt <<" " << std::endl;
      }
    } 
  }

  else if(np == 37){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;
  }

  else if(np == 43){
    int MPI_to_ours_1[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7};
    int MPI_to_ours_2[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6};
    jo.joints.resize(np);
    jo.isVisible.resize(np);

    for(int i=0;i<15;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else if(j.isVisible[MPI_to_ours_1[i]]==3 || j.isVisible[MPI_to_ours_2[i]]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }

    int mid_1[14] = {0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12};
    int mid_2[14] = {1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13};

    for(int i=0;i<14;i++){
       
      if(jo.isVisible[mid_1[i]]==2 || jo.isVisible[mid_2[i]]==2){
        jo.isVisible[2*i + 15] = 2;
        jo.isVisible[2*i + 16] = 2;
      }
      else if(jo.isVisible[mid_1[i]]==3 || jo.isVisible[mid_2[i]]==3){
        jo.isVisible[2*i + 15] = 3;
        jo.isVisible[2*i + 16] = 3;
      }
      else {
        jo.isVisible[2*i + 15] = jo.isVisible[mid_1[i]] && jo.isVisible[mid_2[i]];
        jo.isVisible[2*i + 16] = jo.isVisible[2*i + 15];
      }

      jo.joints[2*i + 15] = jo.joints[mid_1[i]]*0.6667 + jo.joints[mid_2[i]]*0.3333;
      jo.joints[2*i + 16] = jo.joints[mid_1[i]]*0.3333 + jo.joints[mid_2[i]]*0.6667;
    } 
  }

  else if(np == 52){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    // 19 -- 48
    for(int i=0;i<15;i++){
      if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
        jo.isVisible[2*i + 18] = 2;
        jo.isVisible[2*i + 19] = 2;
      }
      else if(jo.isVisible[mid_1[i]-1]==3 || jo.isVisible[mid_2[i]-1]==3){
        jo.isVisible[2*i + 18] = 3;
        jo.isVisible[2*i + 19] = 3;
      }
      else {
        jo.isVisible[2*i + 18] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
        jo.isVisible[2*i + 19] = jo.isVisible[2*i + 18];
      }

      jo.joints[2*i + 18] = jo.joints[mid_1[i]-1]*0.6667 + jo.joints[mid_2[i]-1]*0.3333;
      jo.joints[2*i + 19] = jo.joints[mid_1[i]-1]*0.3333 + jo.joints[mid_2[i]-1]*0.6667;
    }
    // 49 -- 52
    for(int i=15;i<19;i++){
      if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
        jo.isVisible[i + 33] = 2;
      }
      else {
        jo.isVisible[i + 33] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
      }
      jo.joints[i + 33] = jo.joints[mid_1[i]-1]*0.5 + jo.joints[mid_2[i]-1]*0.5;
    }
  }

  else if(np == 56){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;

    // int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    // int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    // // 19 -- 48
    // for(int i=0;i<19;i++){
    //   if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
    //     jo.isVisible[2*i + 18] = 2;
    //     jo.isVisible[2*i + 19] = 2;
    //   }
    //   else if(jo.isVisible[mid_1[i]-1]==3 || jo.isVisible[mid_2[i]-1]==3){
    //     jo.isVisible[2*i + 18] = 3;
    //     jo.isVisible[2*i + 19] = 3;
    //   }
    //   else {
    //     jo.isVisible[2*i + 18] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
    //     jo.isVisible[2*i + 19] = jo.isVisible[2*i + 18];
    //   }

    //   jo.joints[2*i + 18] = jo.joints[mid_1[i]-1]*0.6667 + jo.joints[mid_2[i]-1]*0.3333;
    //   jo.joints[2*i + 19] = jo.joints[mid_1[i]-1]*0.3333 + jo.joints[mid_2[i]-1]*0.6667;
    // }
  }

  else if(np == 75){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
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
    //LOG(INFO) << "here " ;

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    // 19 -- 48
    for(int i=0;i<19;i++){
      if(jo.isVisible[mid_1[i]-1]==2 || jo.isVisible[mid_2[i]-1]==2){
        jo.isVisible[3*i + 18] = 2;
        jo.isVisible[3*i + 19] = 2;
        jo.isVisible[3*i + 20] = 2;
      }
      else if(jo.isVisible[mid_1[i]-1]==3 || jo.isVisible[mid_2[i]-1]==3){
        jo.isVisible[3*i + 18] = 3;
        jo.isVisible[3*i + 19] = 3;
        jo.isVisible[3*i + 20] = 3;
      }
      else {
        jo.isVisible[3*i + 18] = jo.isVisible[mid_1[i]-1] && jo.isVisible[mid_2[i]-1];
        jo.isVisible[3*i + 19] = jo.isVisible[3*i + 18];
        jo.isVisible[3*i + 20] = jo.isVisible[3*i + 18];
      }

      jo.joints[3*i + 18] = jo.joints[mid_1[i]-1]*0.75 + jo.joints[mid_2[i]-1]*0.25;
      jo.joints[3*i + 19] = jo.joints[mid_1[i]-1]*0.50 + jo.joints[mid_2[i]-1]*0.50;
      jo.joints[3*i + 20] = jo.joints[mid_1[i]-1]*0.25 + jo.joints[mid_2[i]-1]*0.75;
    }
  }

  else if(np == 78){
    int num_kpt = 12;
    int COCO_to_ours[12] = {7,6,9,8,11,10,13,12,15,14,17,16};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<num_kpt;i++){
      jo.joints[i] = j.joints[COCO_to_ours[i]-1];
      if(j.isVisible[COCO_to_ours[i]-1]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[COCO_to_ours[i]-1];
      }
    }
    
    int cnt = num_kpt;
    for(int i=1;i<num_kpt;i++){
      for(int j=i+1;j<=num_kpt;j++){
        if(jo.isVisible[i-1]==2 || jo.isVisible[j-1]==2){
          jo.isVisible[cnt] = 2;
        }
        else {
          jo.isVisible[cnt] = jo.isVisible[i-1] && jo.isVisible[j-1];
        }

        jo.joints[cnt] = jo.joints[i-1]*0.5 + jo.joints[j-1]*0.5;
        cnt = cnt + 1;
        //std::cout << i << " " << j << " " << cnt <<" " << std::endl;
      }
    } 
  }
  j = jo;
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform_nv(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt) {
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
  const int lb_num = transformed_label->num();

  //LOG(INFO) << "image shape: " << transformed_data->num() << " " << transformed_data->channels() << " " 
  //                             << transformed_data->height() << " " << transformed_data->width();
  //LOG(INFO) << "label shape: " << transformed_label->num() << " " << transformed_label->channels() << " " 
  //                             << transformed_label->height() << " " << transformed_label->width();

  CHECK_EQ(datum_channels, 6);
  CHECK_EQ(im_channels, 6);
  //CHECK_EQ(im_channels, 4);
  //CHECK_EQ(datum_channels, 4);
  //CHECK_EQ(im_channels, 5);
  //CHECK_EQ(datum_channels, 5);

  CHECK_EQ(im_num, lb_num);
  //CHECK_LE(im_height, datum_height);
  //CHECK_LE(im_width, datum_width);
  CHECK_GE(im_num, 1);

  //const int crop_size = param_.crop_size();

  // if (crop_size) {
  //   CHECK_EQ(crop_size, im_height);
  //   CHECK_EQ(crop_size, im_width);
  // } else {
  //   CHECK_EQ(datum_height, im_height);
  //   CHECK_EQ(datum_width, im_width);
  // }

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();

  Transform_nv(datum, transformed_data_pointer, transformed_label_pointer, cnt); //call function 1
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt) {
  
  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = param_.clahe_tile_size();
  int clahe_clipLimit = param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    cv::Size(),
    0,
  };
  MetaData meta;
  
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  //LOG(INFO) << datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // To do: make this a parameter in caffe.proto
  const int mode = 5; //related to datum.channels();

  //const int crop_size = param_.crop_size();
  //const Dtype scale = param_.scale();
  //const bool do_mirror = param_.mirror() && Rand(2);
  //const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  CHECK_GT(datum_channels, 0);
  //CHECK_GE(datum_height, crop_size);
  //CHECK_GE(datum_width, crop_size);

  //before any transformation, get the image from datum
  cv::Mat img = cv::Mat::zeros(datum_height, datum_width, CV_8UC3);
  cv::Mat mask_all, mask_miss;
  if(mode >= 5){
    mask_miss = cv::Mat::ones(datum_height, datum_width, CV_8UC1);
  }
  if(mode == 6){
    mask_all = cv::Mat::zeros(datum_height, datum_width, CV_8UC1);
  }

  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      cv::Vec3b& rgb = img.at<cv::Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }

      if(mode >= 5){
        dindex = 4*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        if (round(d_element/255)!=1 && round(d_element/255)!=0){
          std::cout << d_element << " " << round(d_element/255) << std::endl;
        }
        mask_miss.at<uchar>(i, j) = d_element; //round(d_element/255);
      }

      if(mode == 6){
        dindex = 5*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        mask_all.at<uchar>(i, j) = d_element;
      }
    }
  }

  //testing image
  //imshow("mask_miss",mask_miss);
  //imshow("mask_all",mask_all);
  // if(mode >= 5){
  //   cv::Mat erosion_dst;
  //   int erosion_size = 1;
  //   mask_miss = 1.0/255 *mask_miss;
  //   cv::Mat element = getStructuringElement( MORPH_ELLIPSE,
  //                                    cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
  //                                    cv::Point( erosion_size, erosion_size ) );
  //   erode( mask_miss, erosion_dst, element );
  //   erosion_dst = 255 *erosion_dst;
  //   imshow( "Erosion Demo", erosion_dst );
  // }
  

  //color, contract
  if(param_.do_clahe())
    clahe(img, clahe_tileSize, clahe_clipLimit);
  if(param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  int stride = param_.stride();
  ReadMetaData(meta, data, offset3, offset1);
  if(param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta);

  //visualize original
  if(0 && param_.visualize()) 
    visualize(img, meta, as);

  //Start transforming
  cv::Mat img_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC3);
  cv::Mat mask_miss_aug, mask_all_aug ;
  //cv::Mat mask_miss_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC1);
  //cv::Mat mask_all_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC1);
  cv::Mat img_temp, img_temp2, img_temp3; //size determined by scale
  // We only do random transform as augmentation when training.
  if (phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp, mask_miss, mask_all, meta, mode);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, mask_all, meta, mode);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && param_.visualize()) 
      visualize(img_temp2, meta, as);
    as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, mask_all, mask_all_aug, meta, mode);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(0 && param_.visualize()) 
      visualize(img_temp3, meta, as);
    as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, mask_all_aug, meta, mode);
    //LOG(INFO) << meta.joint_self.joints.size();
    //LOG(INFO) << meta.joint_self.joints[0];
    if(param_.visualize()) 
      visualize(img_aug, meta, as);

    // imshow("img_aug", img_aug);
    // cv::Mat label_map = mask_miss_aug;
    // applyColorMap(label_map, label_map, COLORMAP_JET);
    // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
    // imshow("mask_miss_aug", label_map);

    if (mode > 4){
      resize(mask_miss_aug, mask_miss_aug, cv::Size(), 1.0/stride, 1.0/stride, cv::INTER_CUBIC);
      resize(mask_all_aug, mask_all_aug, cv::Size(), 1.0/stride, 1.0/stride, cv::INTER_CUBIC);
    }
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = cv::Size();
    as.flip = 0;
    as.degree = 0;
  }
  //LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height 
  //          << "); flip:" << as.flip << "; degree: " << as.degree;

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      cv::Vec3b& rgb = img_aug.at<cv::Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
      transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
      transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
    }
  }
  
  // label size is image size/ stride
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = 0; i < np; i++){
        // To do
        // if (mode = 4){
        //   transformed_label[i*channelOffset + g_y*grid_x + g_x] = 1;
        // }
        if(mode > 4){
          float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255; //mask_miss_aug.at<uchar>(i, j); 
          if (meta.joint_self.isVisible[i] != 3){
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
          }
          else{
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
          }
        }
      }  
      // background channel
      //To do: if (mode = 4){
      if(mode == 5){
        transformed_label[np*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255;
      }
      if(mode > 5){
        transformed_label[np*channelOffset + g_y*grid_x + g_x] = 1;
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = float(mask_all_aug.at<uchar>(g_y, g_x)) /255;
      }
    }
  }

  //putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, param_.sigma_center());
  //LOG(INFO) << "image transformation done!";
  generateLabelMap(transformed_label, img_aug, meta);

  //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
  //if(param_.visualize()){
    //dumpEverything(transformed_data, transformed_label, meta);
  //}
}

// include mask_miss
template<typename Dtype>
float DataTransformer<Dtype>::augmentation_scale(cv::Mat& img_src, cv::Mat& img_temp, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice > param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist()/meta.scale_self;
  //LOG(INFO) << "scale_abs: " << scale_abs;
  if (scale_abs > 3.0){
    //scale_abs = std::min(scale_abs/2, float(3.0));
    //std::cout << "scale_abs: " << scale_abs << std::endl;
  }

  float scale = scale_abs * scale_multiplier;
  resize(img_src, img_temp, cv::Size(), scale, scale, cv::INTER_CUBIC);
  if(mode>4){
    resize(mask_miss, mask_miss, cv::Size(), scale, scale, cv::INTER_CUBIC);
  }
  if(mode>5){
    resize(mask_all, mask_all, cv::Size(), scale, scale, cv::INTER_CUBIC);
  }

  //modify meta data
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
  return scale_multiplier;
}

template<typename Dtype>
cv::Size DataTransformer<Dtype>::augmentation_croppad(cv::Mat& img_src, cv::Mat& img_dst, cv::Mat& mask_miss, cv::Mat& mask_miss_aug, cv::Mat& mask_all, cv::Mat& mask_all_aug, MetaData& meta, int mode) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  //LOG(INFO) << "cv::Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  cv::Point2i center = meta.objpos + cv::Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = std::max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = std::max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
  
  img_dst = cv::Mat::zeros(crop_y, crop_x, CV_8UC3) + cv::Scalar(128,128,128);
  mask_miss_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC1) + cv::Scalar(255); //cv::Scalar(1);
  mask_all_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC1);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(cv::Point(coord_x_on_img, coord_y_on_img), cv::Size(img_src.cols, img_src.rows))){
        img_dst.at<cv::Vec3b>(i,j) = img_src.at<cv::Vec3b>(coord_y_on_img, coord_x_on_img);
        if(mode>4){
          mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
        if(mode>5){
          mask_all_aug.at<uchar>(i,j) = mask_all.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
      }
    }
  }

  //modify meta data
  cv::Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return cv::Size(x_offset, y_offset);
}

template<typename Dtype>
bool DataTransformer<Dtype>::augmentation_flip(cv::Mat& img_src, cv::Mat& img_aug, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode) {
  bool doflip;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob());
  }
  else if(param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;
    if(mode>4){
      flip(mask_miss, mask_miss, 1);
    }
    if(mode>5){
      flip(mask_all, mask_all, 1);
    }
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_rotate(cv::Mat& img_src, cv::Mat& img_dst, cv::Mat& mask_miss, cv::Mat& mask_all, MetaData& meta, int mode) {
  
  float degree;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  }
  else if(param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
  }
  else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }
  
  cv::Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  cv::Mat R = cv::getRotationMatrix2D(center, degree, 1.0);
  cv::Rect bbox = cv::RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  cv::warpAffine(img_src, img_dst, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
  if(mode >4){
    cv::warpAffine(mask_miss, mask_miss, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255)); //cv::Scalar(1));
  }
  if(mode >5){
    cv::warpAffine(mask_all, mask_all, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));
  }

  //adjust meta data
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
  return degree;
}
// end here

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_scale(cv::Mat& img_src, cv::Mat& img_temp, MetaData& meta) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice > param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist()/meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  resize(img_src, img_temp, cv::Size(), scale, scale, cv::INTER_CUBIC);
  //modify meta data
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
  return scale_multiplier;
}

template<typename Dtype>
bool DataTransformer<Dtype>::onPlane(cv::Point p, cv::Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

template<typename Dtype>
cv::Size DataTransformer<Dtype>::augmentation_croppad(cv::Mat& img_src, cv::Mat& img_dst, MetaData& meta) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  //LOG(INFO) << "cv::Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  cv::Point2i center = meta.objpos + cv::Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = std::max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = std::max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
  
  img_dst = cv::Mat::zeros(crop_y, crop_x, CV_8UC3) + cv::Scalar(128,128,128);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(cv::Point(coord_x_on_img, coord_y_on_img), cv::Size(img_src.cols, img_src.rows))){
        img_dst.at<cv::Vec3b>(i,j) = img_src.at<cv::Vec3b>(coord_y_on_img, coord_x_on_img);
      }
    }
  }

  //modify meta data
  cv::Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return cv::Size(x_offset, y_offset);
}

template<typename Dtype>
void DataTransformer<Dtype>::swapLeftRight(Joints& j) {
  //assert(j.joints.size() == 9 && j.joints.size() == 14 && j.isVisible.size() == 27 && j.isVisible.size() == 28 && j.isVisible.size() == 29 && j.isVisible.size() == 33 && j.isVisible.size() == 34 && j.isVisible.size() == 43);
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
  else if(np == 29){
    int right[12] = {3,4,5,9,10,11,15,18,19,20,24,25}; 
    int left[12] = {6,7,8,12,13,14,16,21,22,23,26,27}; 
    for(int i=0; i<12; i++){   
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
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
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
}

template<typename Dtype>
bool DataTransformer<Dtype>::augmentation_flip(cv::Mat& img_src, cv::Mat& img_aug, MetaData& meta) {
  bool doflip;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob());
  }
  else if(param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;

    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
void DataTransformer<Dtype>::RotatePoint(cv::Point2f& p, cv::Mat R){
  cv::Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  cv::Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
float DataTransformer<Dtype>::augmentation_rotate(cv::Mat& img_src, cv::Mat& img_dst, MetaData& meta) {
  
  float degree;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  }
  else if(param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
  }
  else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }
  
  cv::Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  cv::Mat R = cv::getRotationMatrix2D(center, degree, 1.0);
  cv::Rect bbox = cv::RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  warpAffine(img_src, img_dst, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
  
  //adjust meta data
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
  return degree;
}

template<typename Dtype>
void DataTransformer<Dtype>::putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma){
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

template<typename Dtype>
void DataTransformer<Dtype>::putVecPeaks(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  cv::Point2f bc = centerB - centerA;
  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;

  for(int j=0;j<3;j++){
    //cv::Point2f center = centerB*0.5 + centerA*0.5;
    cv::Point2f center = centerB*0.5*j + centerA*0.5*(2-j);

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
void DataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  cv::Point2f bc = centerB - centerA;
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

  int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;

  // float x_p = (centerA.x + centerB.x) / 2;
  // float y_p = (centerA.y + centerB.y) / 2;
  // float angle = atan2f(centerB.y - centerA.y, centerB.x - centerA.x);
  // float sine = sinf(angle);
  // float cosine = cosf(angle);
  // float a_sqrt = (centerA.x - x_p) * (centerA.x - x_p) + (centerA.y - y_p) * (centerA.y - y_p);
  // float b_sqrt = 10; //fixed

  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      cv::Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      float dist = std::abs(ba.x*bc.y -ba.y*bc.x);

      // float A = cosine * (g_x - x_p) + sine * (g_y - y_p);
      // float B = sine * (g_x - x_p) - cosine * (g_y - y_p);
      // float judge = A * A / a_sqrt + B * B / b_sqrt;

      if(dist <= thre){
      //if(judge <= 1){
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

template<typename Dtype>
void DataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, cv::Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = param_.stride();
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
        if (mode == 6 && i == (2*np + 1))
          continue;
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }

  //LOG(INFO) << "label cleaned";

  if (np == 37){
    for (int i = 0; i < 18; i++){
      cv::Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        cv::Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                          grid_x, grid_y, param_.sigma());
        }
      }
    }

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};

    for(int i=0;i<19;i++){
      for (int j=1;j<=3;j++){
        Joints jo = meta.joint_self;
        if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
          cv::Point2f center = jo.joints[mid_1[i]-1]*(1-j*0.25) + jo.joints[mid_2[i]-1]*j*0.25;
          putGaussianMaps(transformed_label + (np+19+i)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
        }

        for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
          Joints jo2 = meta.joint_others[j];
          if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
            cv::Point2f center = jo2.joints[mid_1[i]-1]*(1-j*0.25) + jo2.joints[mid_2[i]-1]*j*0.25;
            putGaussianMaps(transformed_label + (np+19+i)*channelOffset, center, param_.stride(), 
                            grid_x, grid_y, param_.sigma());
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
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = std::max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
  }
  else if (np == 56){
    for (int i = 0; i < 18; i++){
      cv::Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        cv::Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride(), 
                          grid_x, grid_y, param_.sigma());
        }
      }
    }

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    int thre = 1;

    for(int i=0;i<19;i++){
      // if (i>14){
      //   thre = 1;
      // }
      cv::Mat count = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
        //putVecPeaks
        putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
          //putVecPeaks
          putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+39; i < np+57; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = std::max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
  }
  else{
    for (int i = 0; i < np; i++){
      //LOG(INFO) << i << meta.numOtherPeople;
      cv::Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
      //LOG(INFO) << "label put for" << i;
      //plot others
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        cv::Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+1)*channelOffset, center, param_.stride(), 
                          grid_x, grid_y, param_.sigma());
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
            transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = maximum; //std::max(1.0-maximum, 0.0);
          }
        }
      }
    }
    //LOG(INFO) << "background put";
  }

  //visualize
  if(1 && param_.visualize()){
    cv::Mat label_map;
    for(int i = 0; i < 2*(np+1); i++){      
      label_map = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      //int MPI_index = MPI_to_ours[i];
      //cv::Point2f center = meta.joint_self.joints[MPI_index];
      for (int g_y = 0; g_y < grid_y; g_y++){
        //printf("\n");
        for (int g_x = 0; g_x < grid_x; g_x++){
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
          //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
        }
      }
      resize(label_map, label_map, cv::Size(), stride, stride, cv::INTER_LINEAR);
      applyColorMap(label_map, label_map, cv::COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
      
      //center = center * (1.0/(float)param_.stride());
      //circle(label_map, center, 3, CV_RGB(255,0,255), -1);
      char imagename [100];
      sprintf(imagename, "augment_%04d_label_part_%02d.jpg", meta.write_number, i);
      //LOG(INFO) << "filename is " << imagename;
      imwrite(imagename, label_map);
    }
    
    // label_map = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
    // for (int g_y = 0; g_y < grid_y; g_y++){
    //   //printf("\n");
    //   for (int g_x = 0; g_x < grid_x; g_x++){
    //     label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[np*channelOffset + g_y*grid_x + g_x]*255);
    //     //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
    //   }
    // }
    // resize(label_map, label_map, cv::Size(), stride, stride, cv::INTER_CUBIC);
    // applyColorMap(label_map, label_map, cv::COLORMAP_JET);
    // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

    // for(int i=0;i<np;i++){
    //   cv::Point2f center = meta.joint_self.joints[i];// * (1.0/param_.stride());
    //   circle(label_map, center, 3, CV_RGB(100,100,100), -1);
    // }
    // char imagename [100];
    // sprintf(imagename, "augment_%04d_label_part_back.jpg", counter);
    // //LOG(INFO) << "filename is " << imagename;
    // imwrite(imagename, label_map);
  }
}

void setLabel(cv::Mat& im, const std::string label, const cv::Point& org) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, org + cv::Point(0, baseline), org + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
}

template<typename Dtype>
void DataTransformer<Dtype>::visualize(cv::Mat& img, MetaData meta, AugmentSelection as) {
  //cv::Mat img_vis = cv::Mat::zeros(img.rows*2, img.cols, CV_8UC3);
  //copy image content
  // for (int i = 0; i < img.rows; ++i) {
  //   for (int j = 0; j < img.cols; ++j) {
  //     cv::Vec3b& rgb = img.at<cv::Vec3b>(i, j);
  //     cv::Vec3b& rgb_vis_upper = img_vis.at<cv::Vec3b>(i, j);
  //     rgb_vis_upper = rgb;
  //   }
  // }
  // for (int i = 0; i < img_aug.rows; ++i) {
  //   for (int j = 0; j < img_aug.cols; ++j) {
  //     cv::Vec3b& rgb_aug = img_aug.at<cv::Vec3b>(i, j);
  //     cv::Vec3b& rgb_vis_lower = img_vis.at<cv::Vec3b>(i + img.rows, j);
  //     rgb_vis_lower = rgb_aug;
  //   }
  // }
  cv::Mat img_vis = img.clone();
  static int counter = 0;

  rectangle(img_vis, meta.objpos-cv::Point2f(3,3), meta.objpos+cv::Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
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
    else {
      if(meta.joint_self.isVisible[i] <= 1)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(200,200,255), -1);
    }
  }
  
  line(img_vis, meta.objpos+cv::Point2f(-368/2,-368/2), meta.objpos+cv::Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+cv::Point2f(368/2,-368/2), meta.objpos+cv::Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+cv::Point2f(368/2,368/2), meta.objpos+cv::Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+cv::Point2f(-368/2,368/2), meta.objpos+cv::Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);

  for(int p=0;p<meta.numOtherPeople;p++){
    rectangle(img_vis, meta.objpos_other[p]-cv::Point2f(3,3), meta.objpos_other[p]+cv::Point2f(3,3), CV_RGB(0,255,255), CV_FILLED);
    for(int i=0;i<np;i++){
      if(meta.joint_others[p].isVisible[i] <= 1)
        circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,0,255), -1);
      // else
      //   circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,255,255), -1);
      
      //MPII R leg: 0(ankle), 1(knee), 2(hip)
      //     L leg: 5(ankle), 4(knee), 3(hip)
      //     R arms: 10(wrist), 11(elbow), 12(shoulder)
      //     L arms: 13(wrist), 14(elbow), 15(shoulder)
      //if(i==0 || i==1 || i==2 || i==10 || i==11 || i==12)
      
      //circle(img_vis, meta.joint_others[p].joints[i], 2, CV_RGB(255,0,0), -1);
      
      //else if(i==5 || i==4 || i==3 || i==13 || i==14 || i==15)
        //circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,255,255), -1);
      //else
        //circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(255,255,0), -1);
    }
  }
  
  // draw text
  if(phase_ == TRAIN){
    std::stringstream ss;
    // ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: " 
    //    << as.crop.height << "," << as.crop.width;
    ss << meta.dataset << " " << meta.write_number << " index:" << meta.annolist_index << "; p:" << meta.people_index 
       << "; o_scale: " << meta.scale_self;
    string str_info = ss.str();
    setLabel(img_vis, str_info, cv::Point(0, 20));

    stringstream ss2; 
    ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
    str_info = ss2.str();
    setLabel(img_vis, str_info, cv::Point(0, 40));

    rectangle(img_vis, cv::Point(0, 0+img_vis.rows), cv::Point(param_.crop_size_x(), param_.crop_size_y()+img_vis.rows), cv::Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "augment_%04d_epoch_%d_writenum_%d.jpg", counter, meta.epoch, meta.write_number);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, cv::Point(0, 20));

    char imagename [100];
    sprintf(imagename, "augment_%04d.jpg", counter);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  counter++;
}

template <typename Dtype>
void DataTransformer<Dtype>::clahe(cv::Mat& bgr_image, int tileSize, int clipLimit) {
  cv::Mat lab_image;
  cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  std::vector<cv::Mat> lab_planes(3);
  split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileSize, tileSize));
  //clahe->setClipLimit(4);
  cv::Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  merge(lab_planes, lab_image);

  // convert back to RGB
  cv::Mat image_clahe;
  cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe.clone();
}

template <typename Dtype>
void DataTransformer<Dtype>::dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData meta){
  
  char filename[100];
  sprintf(filename, "transformed_data_%04d_%02d", meta.annolist_index, meta.people_index);
  std::ofstream myfile;
  myfile.open(filename);
  int data_length = param_.crop_size_y() * param_.crop_size_x() * 4;
  
  //LOG(INFO) << "before copy data: " << filename << "  " << data_length;
  for(int i = 0; i<data_length; i++){
    myfile << transformed_data[i] << " ";
  }
  //LOG(INFO) << "after copy data: " << filename << "  " << data_length;
  myfile.close();

  sprintf(filename, "transformed_label_%04d_%02d", meta.annolist_index, meta.people_index);
  myfile.open(filename);
  int label_length = param_.crop_size_y() * param_.crop_size_x() / param_.stride() / param_.stride() * (param_.num_parts()+1);
  for(int i = 0; i<label_length; i++){
    myfile << transformed_label[i] << " ";
  }
  myfile.close();
}
// CPM end extra code





}  // namespace caffe
