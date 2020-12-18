#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "engine.h"
#include "cuda/decode.h"
#include "cuda/nms.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

vector<at::Tensor> decode(at::Tensor box_head, at::Tensor conf_head, at::Tensor landm_head,
    vector<float> &anchors, int width, int height, float resize, float score_thresh, int top_n) {
    
    CHECK_INPUT(box_head);
    CHECK_INPUT(conf_head);
    CHECK_INPUT(landm_head);

    int batch = conf_head.size(0);
    int f_height = conf_head.size(2);
    int f_width = conf_head.size(3);
    // int num_anchors = anchors.size() / 4;
    int num_anchors = f_height * f_width * 2;
    auto options = conf_head.options();

    auto scores = at::zeros({batch, top_n}, options);
    auto boxes = at::zeros({batch, top_n, 4}, options);
    auto landms = at::zeros({batch, top_n, 10}, options);

    vector<void *> inputs = {conf_head.data_ptr(), box_head.data_ptr(), landm_head.data_ptr()};
    vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), landms.data_ptr()};

    // Create scratch buffer
    int size = retinaface::cuda::decode(batch, nullptr, nullptr, num_anchors, anchors, width, height, f_width, f_height, resize, score_thresh, top_n, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Decode boxes
    retinaface::cuda::decode(batch, inputs.data(), outputs.data(), num_anchors, anchors, width, height, f_width, f_height, resize, score_thresh, top_n,
        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

    return {scores, boxes, landms};
}

vector<at::Tensor> nms(at::Tensor scores, at::Tensor boxes, at::Tensor landms, float nms_thresh, int detections_per_im) {
    CHECK_INPUT(scores);
    CHECK_INPUT(boxes);
    int batch = scores.size(0);
    int count = scores.size(1);
    auto options = scores.options();
    auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
    auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());
    auto nms_landms = at::zeros({batch, detections_per_im, 10}, landms.options());

    vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), landms.data_ptr()};
    vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_landms.data_ptr()};

    // Create scratch buffer
    int size = retinaface::cuda::nms(batch, nullptr, nullptr, count,
        detections_per_im, nms_thresh, nullptr, 0, nullptr);
    auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

    // Perform NMS
    retinaface::cuda::nms(batch, inputs.data(), outputs.data(), count, detections_per_im, 
        nms_thresh, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());
    
    return {nms_scores, nms_boxes, nms_landms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode);
    m.def("nms", &nms);
}
