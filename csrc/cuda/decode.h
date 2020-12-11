#pragma once

#include <vector>

namespace retinaface {
namespace cuda {

int decode(int batch_size,
    const void *const *inputs, void *const *outputs, int num_anchors,
    const std::vector<float> &anchors, int width, int height, float resize, float score_thresh, int top_n,
    void *workspace, size_t workspace_size, cudaStream_t stream);

}
}