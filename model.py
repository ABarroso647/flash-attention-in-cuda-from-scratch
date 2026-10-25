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

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float smem[];

    // sizes
    const int q_size  = tile_q * head_dim;
    const int kv_size = tile_k * head_dim;

    // long-lived regions
    float* s_q = &smem[0];
    float* m   = &smem[q_size];
    float* l   = &smem[q_size + tile_q];
    float* o   = &smem[q_size + 2*tile_q];

    // per-iteration regions
    float* s_k      = &smem[2*q_size + 2*tile_q];
    float* s_v      = &smem[2*q_size + 2*tile_q + kv_size];
    float* s_scores = &smem[2*q_size + 2*tile_q + 2*kv_size];
    float* m_block = &smem[2*q_size+2*tile_q+2*kv_size + tile_q*tile_k];
    float* row_sum_out = &smem[2*q_size+3*tile_q+2*kv_size + tile_q*tile_k];
    // load this block's Q tile
    const int q_row = blockIdx.x * tile_q;
    load_tile(q, s_q,
              q_row, 0,
              seq_len, head_dim,
              tile_q, head_dim,
              threadIdx.x, blockDim.x);
    //get the base values
    for(int i = threadIdx.x; i<tile_q; i+=blockDim.x){
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }  
    for(int i = threadIdx.x; i<q_size; i+=blockDim.x){
        o[i] = 0.0f;
    }    
    __syncthreads(); 
    //main loop  
    for(int i = 0; i< seq_len; i+=tile_k){
        load_tile(k,s_k,
        i, 0,
        seq_len, head_dim,
        tile_k, head_dim,
        threadIdx.x, blockDim.x);

        load_tile(v, s_v,
        i, 0,
        seq_len, head_dim,
        tile_k, head_dim,
        threadIdx.x, blockDim.x);

        __syncthreads();

        tile_scores(
            s_q, s_k, s_scores, tile_q, tile_k, head_dim, scale, threadIdx.x, blockDim.x
        );
        __syncthreads();
        for(int cur_score = threadIdx.x; cur_score < tile_q*tile_k; cur_score+=blockDim.x){
            if (i+ (cur_score%tile_k) >= seq_len){
                s_scores[cur_score] = -INFINITY;
            }
        }
        __syncthreads();
        tile_rowmax(s_scores, m_block, tile_q, tile_k, threadIdx.x, blockDim.x);
        __syncthreads();

        for(int max = threadIdx.x; max < tile_q; max+=blockDim.x){
            m_block[max] = online_max(m[max], m_block[max]);
        }
        __syncthreads();

        tile_exp(s_scores, m_block, tile_q, tile_k, threadIdx.x, blockDim.x);
        __syncthreads();
        tile_rowsum(s_scores, row_sum_out, tile_q,tile_k, threadIdx.x, blockDim.x);
        __syncthreads();
        for(int row_idx = threadIdx.x; row_idx < tile_q; row_idx+=blockDim.x){
           float correction = correction_factor(m[row_idx], m_block[row_idx]);
           l[row_idx] = update_running_sum(l[row_idx], correction, row_sum_out[row_idx]);
           rescale_output(&o[row_idx*head_dim],head_dim, correction);
           m[row_idx]=m_block[row_idx];
        }
        __syncthreads();
        accumulate_pv(s_scores, s_v, o, tile_q, tile_k, head_dim, threadIdx.x, blockDim.x);

        __syncthreads();
    }
    for(int i = threadIdx.x; i<q_size; i+=blockDim.x){
        int idx = i + (q_row*head_dim);
        if (idx < seq_len*head_dim){
            out[idx] = o[i] / l[i/head_dim];
        }
    } 
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {

    float scale = 1/sqrtf(head_dim);
    int blocks = (seq_len+(tile_q-1))/tile_q;
    int thread_size = 128;
    size_t shmem = (2 * tile_q * head_dim + 2 * tile_k * head_dim + tile_k* tile_q + 4* tile_q) *sizeof(float);

    flash_attention_kernel<<<blocks, thread_size, shmem>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);

}

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

