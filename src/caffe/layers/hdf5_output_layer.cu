#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {


template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
/*
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  data_blob_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
  label_blob_.Reshape(bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->height(), bottom[1]->width());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->num();
  const int label_datum_dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < bottom[0]->num(); ++i) {
    caffe_copy(data_datum_dim, &bottom[0]->gpu_data()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->gpu_data()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
  */
  CHECK_GE(bottom.size(), 1); //ensuring multiple blobs
  CHECK_EQ(this->layer_param_.bottom_size(), bottom.size());
  for (int i=0; i<bottom.size(); ++i) {
    //saving each blob
    stringstream batch_id;
    batch_id << this->layer_param_.bottom(i) << "_" << current_batch_;
    LOG_FIRST_N(INFO, bottom.size()) << "Saving batch " << batch_id.str()
        << " to HDF5 file " << file_name_;
    hdf5_save_nd_dataset(file_id_, batch_id.str(), *bottom[i]);
  }
  H5Fflush(file_id_, H5F_SCOPE_GLOBAL);
  current_batch_++;
}


template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5OutputLayer);

}  // namespace caffe
