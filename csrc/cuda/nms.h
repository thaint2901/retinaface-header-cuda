#pragma once

namespace retinaface {
namespace cuda {

int nms(int batch_size,
    const void *const *inputs, void *const *outputs,
    size_t count, int detections_per_im, float nms_thresh,
    void *workspace, size_t workspace_size, cudaStream_t stream);

}
}