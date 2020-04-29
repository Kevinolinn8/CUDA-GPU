#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

const int BLOCKSIZE = 32;
constexpr float THRESHOLD = 1e-6f;
// WRITE CUDA KERNEL FOR COUNT HERE
__global__ void spmv(const int *pointer, const int *indices, const float *vals, const float *x, float *y) {

    int thread = threadIdx.x;
    int block = blockDim.x;
    int row = blockIdx.x;

    __shared__ float sum[BLOCKSIZE];

    int begin = pointer[row];
    int end = pointer[row + 1];
    

    float temp = 0.f;
    for(int i = begin + thread; i < end; i+=block){
        temp += vals[i] * x[indices[i]];
    }

    sum[thread] = temp;
    for(int stride = block/2; stride > 0; stride >>= 1){

        __syncthreads();
        
        if(thread < stride){
           sum[thread] += sum[thread + stride];
        }
    }
    __syncthreads();

    if(thread == 0){ 
    y[row] = sum[0];
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

    

    //input_gpu
    float * sparse_matrix_gpu = nullptr;
    float * dense_vector_gpu = nullptr;
    int * ptr_gpu = nullptr;
    int * indices_gpu = nullptr;
    
    
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

    

    // PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    //int * data = read_file(argv[1], &rows, &cols);
    float * returnArray = nullptr;
    
     //allocate memory
    cudaMalloc(&sparse_matrix_gpu, sizeof(float) * values);
    cudaMalloc(&dense_vector_gpu, sizeof(float) * cols);
    cudaMalloc(&ptr_gpu, sizeof(int) * (rows + 1));
    cudaMalloc(&indices_gpu, sizeof(int) * values);
    cudaMalloc(&returnArray, sizeof(float) * rows);



    //PERFORM NECESSARY DATA TRANSFER HERE
    cudaMemcpyAsync(sparse_matrix_gpu, sparse_matrix, sizeof(float) * values, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_gpu, ptr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dense_vector_gpu, dense_vector, sizeof(float) * cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(indices_gpu, indices, sizeof(int) * values, cudaMemcpyHostToDevice, stream);


    // LAUNCH KERNEL HERE
    dim3 block(BLOCKSIZE);
    dim3 grid(rows);
    cudaEventRecord(begin, stream);
    spmv<<<grid,block,0,stream>>>(ptr_gpu,indices_gpu, sparse_matrix_gpu, dense_vector_gpu, returnArray);
    cudaEventRecord(end, stream);
 

   // PERFORM NECESSARY DATA TRANSFER HERE
    cudaMemcpyAsync(h_output, returnArray, sizeof(float) * rows, cudaMemcpyDeviceToHost, stream);


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


    //FREE THE VARIABLES I USED
    free(sparse_matrix_gpu);
    free(dense_vector_gpu);
    free(ptr_gpu);
    free(indices_gpu);
    free(returnArray);
    free(reference_output);
    free(h_output);
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

