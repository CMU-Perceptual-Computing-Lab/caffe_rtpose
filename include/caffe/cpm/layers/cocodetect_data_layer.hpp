#ifndef CAFFE_CPM_LAYER_COCODETECTDATA_LAYER_HPP_
#define CAFFE_CPM_LAYER_COCODETECTDATA_LAYER_HPP_

#include "caffe/cpm/cocodetect_data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class CocoDetectDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CocoDetectDataLayer(const LayerParameter& param);
  virtual ~CocoDetectDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CocoDetectDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  CocoDetectDataReader reader_;
  Blob<Dtype> transformed_label_; // add another blob
  Blob<Dtype> missing_part_mask_;
};

}  // namespace caffe

#endif  // CAFFE_CPM_LAYER_COCODETECTDATA_LAYER_HPP_
