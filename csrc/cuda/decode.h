#pragma once

#include <vector>

namespace retinaface {
namespace cuda {

int decode(int batch_size,
    const void *const *inputs, void *const *outputs, size_t num_anchors,
    const std::vector<float> &anchors, int width, int height, size_t f_width, size_t f_height, float resize, float score_thresh, int top_n,
    void *workspace, size_t workspace_size, cudaStream_t stream);

}
}