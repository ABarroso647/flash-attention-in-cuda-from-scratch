"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = exp(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float row_max = -INFINITY;
    if (i >= rows) return;
    for (int cur_col = 0; cur_col < cols; ++cur_col){
        row_max = max(row_max, matrix[i*cols + cur_col]);
    }
    out[i] = row_max;
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int row = blockIdx.x; 
    float partial = 0.0f;
    for (int cur_col = threadIdx.x; cur_col < cols; cur_col+=blockDim.x){
        partial = partial + matrix[row*cols + cur_col];
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for(int stride = blockDim.x/2; stride>0; stride>>=1)
    {
        if(threadIdx.x < stride){
            sdata[threadIdx.x] += sdata[threadIdx.x+stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float accumulator = 0.0f;
    for (int i = 0; i < n; ++i){
        accumulator = accumulator + a[i] * b[i];  
    }
    return accumulator;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    
    float cur_sum = 0.0f;
    
    for(int l = 0; l<k; ++l){
        cur_sum += a[i*k+l] * b[l*n + j];
    }     
    
    c[i*n+j] = cur_sum; 

}

# Step 8 - transpose (not yet solved)
# TODO: implement

# Step 9 - qk_scores (not yet solved)
# TODO: implement

# Step 10 - softmax_rows (not yet solved)
# TODO: implement

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

