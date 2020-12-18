// nvcc --expt-extended-lambda -std=c++14 test.cu -o test

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <stdio.h>

using namespace thrust::placeholders;

__global__ void softmax_kernel(float *in_scores, float *out_scores, int num_elem) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx >= num_elem) return;

    printf("%2.0f \n", in_scores[idx]);

    // for (int k = 0; k < 2; ++k) {
    //     float conf1 = in_scores[idx + k * num_elem * 2];
    //     float conf2 = in_scores[idx + k * num_elem * 2 + num_elem];
    //     printf("%.1f", conf2);
    //     out_scores[idx + k * num_elem * 2 + num_elem] = expf(conf2) / (expf(conf1) + expf(conf2));
    //     out_scores[idx + k * num_elem * 2] = expf(conf1) / (expf(conf1) + expf(conf2));
    // }
}

int main(void)
{
    // // --- Input data 
    // float a = 2.0f;
    // float x[4] = { 1, 2, 3, 4 };
    // float y[4];

    // // thrust::device_vector<float> X(x, x + 4);
    // // thrust::device_vector<float> Y(y, y + 4);

    // thrust::transform(x, 
    //                   x + 4,  
    //                   cub::CountingInputIterator<int>(0), 
    //                   y,
    //                   [=] __host__ __device__ (float x, int i) {
    //                     if (i >= 2) {
    //                         return false;
    //                     } else {
    //                         return (x > 1);
    //                     }

    //                 }      // --- Lambda expression 
    // );
    int N = 32640;
    float *in_scores, *out_scores;
    float *d_in_scores, *d_out_scores;
    // Allocate host memory
    in_scores = (float*)malloc(sizeof(float) * N);
    out_scores = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        in_scores[i] = 1.0f;
    }
    
    cudaMalloc((void**)&d_in_scores, N*sizeof(float));
    cudaMalloc((void**)&d_out_scores, N*sizeof(float));

    // Transfer data from host to device memory
    cudaMemcpy(d_in_scores, in_scores, sizeof(float) * N, cudaMemcpyHostToDevice);

    int thread_count;
    const int thread_count_ = 1024;
    int num_elem = 68*120;
    thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
    softmax_kernel<<<(num_elem + thread_count - 1) / thread_count, thread_count>>>(d_in_scores, d_out_scores, num_elem);

    // Transfer data back to host memory
    cudaMemcpy(out_scores, d_out_scores, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < N; i++) std::cout << out_scores[i] << std::endl;

    return 0;
}