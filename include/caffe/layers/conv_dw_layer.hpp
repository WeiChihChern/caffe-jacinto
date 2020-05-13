//taken and modified from https://github.com/sp2823/caffe
//see also: https://github.com/BVLC/caffe/pull/5665/commits

#ifndef CAFFE_CONV_DW_LAYER_HPP_
#define CAFFE_CONV_DW_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/quantized_layer.hpp"

namespace caffe {

template<typename Ftype, typename Btype>
class ConvolutionDepthwiseLayer : public QuantizedLayer<Ftype,Btype> {
 public:
  explicit ConvolutionDepthwiseLayer(const LayerParameter& param)
      : QuantizedLayer<Ftype,Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "ConvolutionDepthwise"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  unsigned int kernel_h_;
  unsigned int kernel_w_;
  unsigned int stride_h_;
  unsigned int stride_w_;
  unsigned int pad_h_;
  unsigned int pad_w_;
  unsigned int dilation_h_;
  unsigned int dilation_w_;
  TBlob<Btype> weight_buffer_;
  TBlob<Btype> weight_multiplier_;
  TBlob<Btype> bias_buffer_;
  TBlob<Btype> bias_multiplier_;
};

}  // namespace caffe


#endif  // CAFFE_CONV_DW_LAYER_HPP_
