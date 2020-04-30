#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

const int BLOCKSIZE = 32;
constexpr float THRESHOLD = 1e-5f;



__global__ void spmv(const int *row_ptr, const int *col_ind, const float *vals, const float *x, float *y) {

    int myThread = threadIdx.x;
    int block = blockDim.x;
    int row = blockIdx.x;

    __shared__ float ret_arr[BLOCKSIZE];

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    

    float temp = 0.f;
    for(int i = start + myThread; i < end; i+=block){
        temp += vals[i] * x[col_ind[i]];
    }

    ret_arr[myThread] = temp;
    for(int stride = block/2; stride > 0; stride >>= 1){

        __syncthreads();
        
        if(myThread < stride){
           ret_arr[myThread] += ret_arr[myThread + stride];
        }
    }
    __syncthreads();

    if(myThread == 0){ 
    y[row] = ret_arr[0];
    }
}



float * serial_implementation(float * sparse_matrix, int * ptr, int * indices, float * dense_vector, int rows) {
    float * output = (float *)malloc(sizeof(float) * rows);
    
    for (int i = 0; i < rows; i++) {
        float accumulator = 0.f;
        for (int j = ptr[i]; j < ptr[i+1]; j++) {
            accumulator += sparse_matrix[j] * dense_vector[indices[j]];
        }
        output[i] = accumulator;
    }
    
    return output;
}

int main(int argc, char ** argv) {
    
    assert(argc == 2);
    
    // input_cpu
    float * sparse_matrix = nullptr; 
    float * dense_vector = nullptr;
    int * ptr = nullptr;
    int * indices = nullptr;
    int values = 0, rows = 0, cols = 0;
    
    
    read_sparse_file(argv[1], &sparse_matrix, &ptr, &indices, &values, &rows, &cols);
    printf("%d %d %d\n", values, rows, cols);
    dense_vector = (float *)malloc(sizeof(float) * cols);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_real_distribution<> dist(-10.0f, 10.0f);
    for (int i = 0; i < cols; i++) {
        dense_vector[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float * h_output = (float *)malloc(sizeof(float) * rows); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    

    // VARIABLE DECLARATIONS
    float * retarray = nullptr;
    float * GPUmatrix = nullptr;
    float * GPUvector = nullptr;
    int * GPUptr = nullptr;
    int * ind_gpu = nullptr;
    
     //allocate memory
    cudaMalloc(&GPUmatrix, sizeof(float) * values);
    cudaMalloc(&GPUvector, sizeof(float) * cols);
    cudaMalloc(&GPUptr, sizeof(int) * (rows + 1));
    cudaMalloc(&ind_gpu, sizeof(int) * values);
    cudaMalloc(&retarray, sizeof(float) * rows);



    //DATA TRANSFER
    cudaMemcpyAsync(GPUmatrix, sparse_matrix, sizeof(float) * values, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(GPUptr, ptr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(GPUvector, dense_vector, sizeof(float) * cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ind_gpu, indices, sizeof(int) * values, cudaMemcpyHostToDevice, stream);

    
    dim3 block(BLOCKSIZE);
    dim3 grid(rows);
    
    // LAUNCH KERNEL
    cudaEventRecord(begin, stream);
    spmv<<<grid,block,0,stream>>>(GPUptr,ind_gpu, GPUmatrix, GPUvector, retarray);
    cudaEventRecord(end, stream);
 

   //DATA TRANSFER
    cudaMemcpyAsync(h_output, retarray, sizeof(float) * rows, cudaMemcpyDeviceToHost, stream);


    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    float * reference_output = serial_implementation(sparse_matrix, ptr, indices, dense_vector, rows);
    for (int i = 0; i < rows; i++) {
        if (fabs(reference_output[i] - h_output[i]) > THRESHOLD) {
            printf("ERROR: %f != %f at index %d\n", reference_output[i], h_output[i], i);
            abort();
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);


    //FREE CUDAFREE
    cudaFree(GPUmatrix);
    cudaFree(GPUvector);
    cudaFree(GPUptr);
    cudaFree(ind_gpu);
    cudaFree(retarray);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(sparse_matrix);
    free(dense_vector);
    free(ptr);
    free(indices);
    free(reference_output);
    free(h_output);

    return 0;
}   

