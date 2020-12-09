#include "decode.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>
#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

namespace retinaface {
namespace cuda {

int decode(int batch_size,
    const void *const *inputs, void *const *outputs, size_t num_anchors,
    const std::vector<float> &anchors, int width, int height, float resize, float score_thresh, int top_n,
    void *workspace, size_t workspace_size, cudaStream_t stream) {
    
    /* height, width: net_inshape
    resize = 0.5, 1920 / 960
    */
    
    int scores_size = num_anchors;

    if (!workspace || !workspace_size) {
        // scratch space size cub style
        workspace_size  = get_size_aligned<float>(anchors.size()); // anchors
        workspace_size += get_size_aligned<bool>(scores_size);     // flags
        workspace_size += get_size_aligned<int>(scores_size);      // indices
        workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
        workspace_size += get_size_aligned<float>(scores_size);    // scores
        workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

        size_t temp_size_flag = 0;
        cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
        cub::CountingInputIterator<int>(scores_size),
        (bool *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
        size_t temp_size_sort = 0;
        cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
        (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
        workspace_size += std::max(temp_size_flag, temp_size_sort);

        return workspace_size;
    }

    auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
    cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

    auto on_stream = thrust::cuda::par.on(stream);

    auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
    auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
        auto in_boxes = static_cast<const float *>(inputs[1]) + batch * scores_size * 4;
        auto in_landms = static_cast<const float *>(inputs[2]) + batch * scores_size * 10;

        auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
        auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;
        auto out_landms = static_cast<float10 *>(outputs[2]) + batch * top_n;

        // Discard scores below threshold
        thrust::transform(on_stream, in_scores, in_scores + scores_size, flags, thrust::placeholders::_1 > score_thresh);

        int *num_selected = reinterpret_cast<int *>(indices_sorted);
        cub::DeviceSelect::Flagged(workspace, workspace_size,
            cub::CountingInputIterator<int>(0),
            flags, indices, num_selected, scores_size, stream);
        cudaStreamSynchronize(stream);
        int num_detections = *thrust::device_pointer_cast(num_selected);
        
        // Only keep top n scores
        auto indices_filtered = indices;
        if (num_detections > top_n) {
            // lấy score theo indices đã chọn ở trên, sort index theo score, đẩy vào scores
            thrust::gather(on_stream, indices, indices + num_detections, in_scores, scores);
            cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);
            indices_filtered = indices_sorted;
            num_detections = top_n;
        }

        // Gather boxes
        bool has_anchors = !anchors.empty();
        thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_landms)),
            [=] __device__ (int i) {
                float4 box = float4{
                    in_boxes[i * 4 + 0],
                    in_boxes[i * 4 + 1],
                    in_boxes[i * 4 + 2],
                    in_boxes[i * 4 + 3]
                };
                float10 landm = float10{
                    in_landms[i * 10 + 0],
                    in_landms[i * 10 + 1],
                    in_landms[i * 10 + 2],
                    in_landms[i * 10 + 3],
                    in_landms[i * 10 + 4],
                    in_landms[i * 10 + 5],
                    in_landms[i * 10 + 6],
                    in_landms[i * 10 + 7],
                    in_landms[i * 10 + 8],
                    in_landms[i * 10 + 9]
                };

                if (has_anchors) {
                    float *d = anchors_d + 4*i;

                    float x1 = d[0] + box.x * 0.1f * d[2];
                    float y1 = d[1] + box.y * 0.1f * d[3];
                    float x2 = d[2] * exp(box.z * 0.2f);
                    float y2 = d[3] * exp(box.w * 0.2f);
                    box = float4{
                        (x1 - x2 * 0.5f) * width / resize,
                        (y1 - y2 * 0.5f) * height / resize,
                        (x2 + x1) * width / resize,
                        (y2 + y1) * height / resize,
                    };

                    float x11 = d[0] + landm.x1 * 0.1f * d[2];
                    float y11 = d[1] + landm.y1 * 0.1f * d[3];
                    float x12 = d[0] + landm.x2 * 0.1f * d[2];
                    float y12 = d[1] + landm.y2 * 0.1f * d[3];
                    float x3 = d[0] + landm.x3 * 0.1f * d[2];
                    float y3 = d[1] + landm.y3 * 0.1f * d[3];
                    float x4 = d[0] + landm.x4 * 0.1f * d[2];
                    float y4 = d[1] + landm.y4 * 0.1f * d[3];
                    float x5 = d[0] + landm.x5 * 0.1f * d[2];
                    float y5 = d[1] + landm.y5 * 0.1f * d[3];
                    landm = float10{
                        x11 * width / resize,
                        y11 * height / resize,
                        x12 * width / resize,
                        y12 * height / resize,
                        x3 * width / resize,
                        y3 * height / resize,
                        x4 * width / resize,
                        y4 * height / resize,
                        x5 * width / resize,
                        y5 * height / resize
                    };
                }

                return thrust::make_tuple(in_scores[i], box, landm);
            });

        // Zero-out unused scores
        if (num_detections < top_n) {
            thrust::fill(on_stream, out_scores + num_detections,
                out_scores + top_n, 0.0f);
        }
    }

    return 0;

}

}
}
