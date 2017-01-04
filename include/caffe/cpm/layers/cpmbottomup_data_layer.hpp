#ifndef CAFFE_CPM_LAYER_CPMBOTTOMUP_DATA_LAYER_HPP_
#define CAFFE_CPM_LAYER_CPMBOTTOMUP_DATA_LAYER_HPP_

#include "caffe/cpm/cpmbottomup_data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class CPMBottomUpDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CPMBottomUpDataLayer(const LayerParameter& param);
  virtual ~CPMBottomUpDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPMBottomUpDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  CPMBottomUpDataReader reader_;
  Blob<Dtype> transformed_label_; // add another blob
  Blob<Dtype> missing_part_mask_;
};

}  // namespace caffe

#endif  // CAFFE_CPM_LAYER_CPMBOTTOMUP_DATA_LAYER_HPP_
