"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = exp(a[i]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
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

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if(r >= rows || c >= cols) return;
    out[c*rows + r] = in[r*cols + c];
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= seq_len || j >= seq_len) return;
  
    float cur_score = 0.0f;
    for (int head = 0; head< head_dim; ++head){
        cur_score+= q[i*head_dim + head] * k[j*head_dim+ head];
    }
    scores[i * seq_len + j] = cur_score * rsqrtf((float)head_dim);

}

# Step 10 - softmax_rows
struct MaxOp{
   __device__ float operator()(float a, float b) const {return max(a,b);}

};
struct SumOp{
  __device__ float operator()(float a, float b) const {return a + b;}
};

template <typename Op>
__device__ float tree_reduction(float* sdata, Op op)
{
   for(int stride = blockDim.x/2; stride>0; stride>>=1){
      if(threadIdx.x < stride){
          sdata[threadIdx.x] = op(sdata[threadIdx.x], sdata[threadIdx.x+stride]);
      }
      __syncthreads();
    }
  float result = sdata[0];
  __syncthreads();
  return result;
}

__global__ void softmax_rows(float* matrix, int rows, int cols) {
  //small guard incase
  if (blockIdx.x >= rows) return;

  //get my row max
  float localmax = -INFINITY;
  for (int stride = threadIdx.x; stride < cols; stride+=blockDim.x){
    localmax = max(localmax, matrix[blockIdx.x*cols + stride]);
  }
  __shared__ float sdata[256];
  sdata[threadIdx.x] = localmax;
  __syncthreads();
  float rowMax = tree_reduction(sdata, MaxOp{});
  //get the exponential summation centered around max
  float localSum = 0.0f;
  for (int stride = threadIdx.x; stride < cols; stride+=blockDim.x){
     matrix[blockIdx.x*cols + stride] = expf(matrix[blockIdx.x*cols + stride] - rowMax);
     localSum += matrix[blockIdx.x*cols + stride]; 
  }
  sdata[threadIdx.x] = localSum;
  __syncthreads();
  float rowSum = tree_reduction(sdata, SumOp{});
  //final softmax computation
  for (int stride = threadIdx.x; stride < cols; stride+=blockDim.x){
     matrix[blockIdx.x*cols + stride] = matrix[blockIdx.x*cols + stride]/rowSum;
  }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;


    if (i >= seq_len || d >= head_dim) return;
    
    float cur_sum = 0.0f;
    
    for(int j = 0; j<seq_len; ++j){
        cur_sum += p[i*seq_len+j] * v[j*head_dim + d];
    }     

    
    out[i*head_dim+d] = cur_sum; 

    



}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {

    dim3 thread_p_block(32, 8);
    dim3 block((seq_len+31)/32,(seq_len+7)/8);
    // qk_scores 
    float* scores;
    cudaMalloc(&scores, seq_len*seq_len *sizeof(float));
    qk_scores<<<block,thread_p_block>>>(d_q, d_k, scores, seq_len, head_dim);
    // softmax_rows
    softmax_rows<<<seq_len,256>>>(scores, seq_len, seq_len);

    // pv_matmul
    pv_matmul<<<block, thread_p_block>>>(scores, d_v, d_out, seq_len, head_dim);
    cudaFree(scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max- new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for(int i = 0; i< head_dim; ++i)
    {
       out_row[i]*= correction; 
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    for(int idx = thread_id; idx < tile_cols*tile_rows; idx+=num_threads)
    {
        const int shift_r = idx/tile_cols + src_row_start;
        const int shift_c = idx%tile_cols + src_col_start; 
        const bool in_bounds = (shift_r < src_rows) && (shift_c < src_cols); 
        shared_dst[idx] = in_bounds ? src[shift_r *src_cols + shift_c] : 0.0f;
        
    }
        
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    const int s_size = tile_q * tile_k;
    for(int idx = thread_id; idx < s_size; idx+=num_threads)
    {   
        const int q_row = idx/tile_k;
        const int k_t_col = idx%tile_k; 

        float dot_prod = 0.0f;
        for(int inner_idx = 0; inner_idx < head_dim; ++inner_idx)
        {
            dot_prod += q_tile[q_row*head_dim + inner_idx] * k_tile[k_t_col*head_dim + inner_idx];
        }
        s_tile[idx] = dot_prod*scale;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, 
int tile_k, int thread_id, int num_threads) {

    for(int idx = thread_id; idx < tile_q; idx+=num_threads){
        float cur_max = -INFINITY;
        for(int col = 0; col<tile_k; ++col){
            cur_max = max(cur_max, s_tile[idx*tile_k+col]);
        }
        row_max_out[idx] = cur_max;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {

    for (int idx = thread_id; idx < tile_q*tile_k; idx+=num_threads){
        const int r = idx/tile_k;
        const int c = idx%tile_k;
        s_tile[r*tile_k + c] = expf(s_tile[r*tile_k + c] - row_max[r]);

    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {

    for(int idx = thread_id; idx< tile_q; idx+=num_threads){
        float cur_sum = 0.0f;
        for(int col = 0; col < tile_k; ++col){
            cur_sum += p_tile[idx*tile_k+ col];
        }
        row_sum_out[idx] = cur_sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, 
int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {

    for(int idx = thread_id; idx< tile_q*head_dim; idx+=num_threads){
        const int r = idx/head_dim;
        const int c = idx%head_dim;
        float dot_product = 0.0f;
        for(int inner_idx = 0; inner_idx < tile_k; ++inner_idx){
            dot_product += p_tile[r*tile_k+inner_idx]*v_tile[inner_idx*head_dim+c];
        }
        out_acc[idx] += dot_product;
    }

}

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

