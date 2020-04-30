#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

const int BLOCKSIZE = 32;
constexpr float THRESHOLD = 1e-6f;



__global__ void spmv(const int *row_ptr, const int *col_ind, const float *vals, const float *x, float *y) {

    int myThread = threadIdx.x;
    int block = blockDim.x;
    int row = blockIdx.x;

    __shared__ float sum[BLOCKSIZE];

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    

    float temp = 0.f;
    for(int i = start + myThread; i < end; i+=block){
        temp += vals[i] * x[col_ind[i]];
    }

    sum[myThread] = temp;
    for(int stride = block/2; stride > 0; stride >>= 1){

        __syncthreads();
        
        if(myThread < stride){
           sum[myThread] += sum[myThread + stride];
        }
    }
    __syncthreads();

    if(myThread == 0){ 
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
    float * matrix_gpu = nullptr;
    float * vector_gpu = nullptr;
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
    float * retarray = nullptr;
    
     //allocate memory
    cudaMalloc(&matrix_gpu, sizeof(float) * values);
    cudaMalloc(&vector_gpu, sizeof(float) * cols);
    cudaMalloc(&ptr_gpu, sizeof(int) * (rows + 1));
    cudaMalloc(&indices_gpu, sizeof(int) * values);
    cudaMalloc(&retarray, sizeof(float) * rows);



    //PERFORM NECESSARY DATA TRANSFER HERE
    cudaMemcpyAsync(matrix_gpu, sparse_matrix, sizeof(float) * values, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(ptr_gpu, ptr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(vector_gpu, dense_vector, sizeof(float) * cols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(indices_gpu, indices, sizeof(int) * values, cudaMemcpyHostToDevice, stream);


    // LAUNCH KERNEL HERE
    dim3 block(BLOCKSIZE);
    dim3 grid(rows);
    cudaEventRecord(begin, stream);
    spmv<<<grid,block,0,stream>>>(ptr_gpu,indices_gpu, matrix_gpu, vector_gpu, retarray);
    cudaEventRecord(end, stream);
 

   // PERFORM NECESSARY DATA TRANSFER HERE
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


    //FREE THE VARIABLES I USED
    cudaFree(matrix_gpu);
    cudaFree(vector_gpu);
    cudaFree(ptr_gpu);
    cudaFree(indices_gpu);
    cudaFree(retarray);
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

