#include "caffe/cpm/layers/cpm_data_layer.hpp"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/cpm/cpmdata_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
CPMDataLayer<Dtype>::CPMDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
CPMDataLayer<Dtype>::~CPMDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read a data point, and use it to initialize the top blob.
  // Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // ***CPMData: Infer the shape of transformed_data_  and transformed_label_ from prototxt
  // ***                            top[0]             and top[1]
  // ***                            prefetch_[i].data_ and prefetch_[i].label_

  vector<int> top_shape(4);

  // image
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  const int height = this->layer_param_.transform_param().crop_size_y();
  const int width = this->layer_param_.transform_param().crop_size_x();
  const bool put_gaussian = this->layer_param_.transform_param().put_gaussian();

  if(put_gaussian){
    top[0]->Reshape(batch_size, 4, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, 4, height, width);
    }
    this->transformed_data_.Reshape(1, 4, height, width);
  }
  else {
    top[0]->Reshape(batch_size, 3, height, width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(batch_size, 3, height, width);
    }
    this->transformed_data_.Reshape(1, 3, height, width);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label and missed-part mask
  if (this->output_labels_) {
    const int stride = this->layer_param_.transform_param().stride();
    int num_parts = this->layer_param_.transform_param().num_parts();

    if(this->layer_param_.transform_param().has_masks()) //for mask channel
      num_parts++;

    top[1]->Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride); //plus 1 for background
    top[2]->Reshape(batch_size, 1, 1, num_parts+1); //plus 1 for background
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
      this->prefetch_[i].missing_part_mask_.Reshape(batch_size, 1, 1, num_parts);
    }
    this->transformed_label_.Reshape(1, 2*(num_parts+1), height/stride, width/stride);
    this->missing_part_mask_.Reshape(1, 1, 1, num_parts+1); //plus 1 for background

    LOG(INFO) << "output label size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
    LOG(INFO) << "output mask size: " << top[2]->num() << ","
        << top[2]->channels() << "," << top[2]->height() << ","
        << top[2]->width();
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void CPMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  static int cnt = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();

  // Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  // *** CPMData: no repeated inferring
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  // top_shape[0] = batch_size;
  // batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_mask = NULL;

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
    top_mask = batch->missing_part_mask_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations on both data and label (mirror, scale, crop...)
    const int offset_data = batch->data_.offset(item_id);
    const int offset_label = batch->label_.offset(item_id);
    const int offset_mask = batch->missing_part_mask_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
    this->transformed_label_.set_cpu_data(top_label + offset_label);
    this->missing_part_mask_.set_cpu_data(top_mask + offset_mask);

    this->data_transformer_->Transform_CPM(datum, &(this->transformed_data_), &(this->transformed_label_),
                                                  &(this->missing_part_mask_), cnt);
    ++cnt;

    // Copy label. No need: all done in Transform_CPM
    //if (this->output_labels_) {
    //  top_label[item_id] = datum.label();
    //}

    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(CPMDataLayer);
REGISTER_LAYER_CLASS(CPMData);

}  // namespace caffe
