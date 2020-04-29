#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

/* start by writing device function
    
    takes in two shared memory buffers
    
    use this scan 
    
    use that in two kernels, chunks of an original array, scans and writes out largest values 
    to larger array
    
    Kernel 2 takes in array of largest values and does a scan across those
    
    Final Kernel, takes in scanned chunks and the scanned max chunks
    
    post follow up to lecture on Piazza
    
    algorithm in lecture is inclusive 
    we do exclusive scan
*/


constexpr int BLOCK_SIZE = 1024;
__global__ void scan1(int *data_out, int *data_in, int *max, int N) {
        // data_out: output array
        // data_in: input array
        // N: total values

        __shared__ int partial[BLOCK_SIZE * 2]; 
        
        int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        int thread = threadIdx.x;
        // takes in two shared memory buffers

        int buffer_output = 0;
        int buffer_input = 1;

        if (global_index > 0 && global_index < N) {
         partial[buffer_output * BLOCK_SIZE + thread] = data_out[global_index-1];
        }else{
         partial[buffer_output * BLOCK_SIZE + thread] = 0;
        }
       
       __syncthreads();          

        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2)   {
                buffer_output = 1 - buffer_output;
                buffer_input = 1 - buffer_output;

            if (thread >= offset){
                partial[buffer_output * BLOCK_SIZE + thread] = partial[buffer_input * BLOCK_SIZE + thread - offset] + partial[buffer_input * BLOCK_SIZE + thread];
            }else{
                partial[buffer_output * BLOCK_SIZE + thread] = partial[buffer_input*BLOCK_SIZE+thread];
            }
            __syncthreads();                    

        }
        if (global_index < N) {
            data_out[global_index] = partial[buffer_output * BLOCK_SIZE + thread];
        }
        // Write the max value of each chunk to a separate array   
        if(thread == 0){
            max[blockIdx.x] = partial[buffer_output * BLOCK_SIZE + BLOCK_SIZE -1];
        }
    }
/*
KERNEL 2
Scans the array of max valu to produce the offsets for the the final array
*/
__global__ void scan2(int *data_out, int *data_in, int N) {
        // data_out: output array
        // data_in: input array
        // N: total values


        __shared__ int partial[BLOCK_SIZE * 2]; 
        
        int global_index = blockIdx.x * blockDim.x + threadIdx.x;
        int thread = threadIdx.x;
        // takes in two shared memory buffers

        int buffer_output = 0;
        int buffer_input = 1;

        if (global_index > 0 && global_index < N) {
         partial[buffer_output * BLOCK_SIZE + thread] = data_out[global_index-1];
        }else{
         partial[buffer_output * BLOCK_SIZE + thread] = 0;
        }
       
       __syncthreads();          

        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2)   {
                buffer_output = 1 - buffer_output;
                buffer_input = 1 - buffer_output;

            if (thread >= offset){
                partial[buffer_output * BLOCK_SIZE + thread] = partial[buffer_input * BLOCK_SIZE + thread - offset] + partial[buffer_input * BLOCK_SIZE + thread];
            }else{
                partial[buffer_output * BLOCK_SIZE + thread] = partial[buffer_input*BLOCK_SIZE+thread];
            }
            __syncthreads();                    

        }
        if (global_index < N) {
            data_out[global_index] = partial[buffer_output * BLOCK_SIZE + thread];
        }

    }


__global__ void scan3(int *input_data, int *maxes, int N ) {
    
    int global_index =  blockIdx.x * blockDim.x + threadIdx.x;
    // global_index = 13, BLOCK_SIZE = 4, blockIdx.x = 3
    if(global_index < N) {
        input_data[global_index] += maxes[blockIdx.x]; 
    }


}



int * serial_implementation(int * data, int vals) {
    int * output = (int *)malloc(sizeof(int) * vals);
    
    output[0] = 0;
    for (int i = 1; i < vals; i++) {
        output[i] = output[i-1] + data[i-1];
    }
    
    return output;
}

int main(int argc, char ** argv) {
    
    assert(argc == 2);
    int values = atoi(argv[1]); // Values is guaranteed to be no more than 10000000
    int * data = (int *)malloc(sizeof(int) * values);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_int_distribution<> dist(0, 50);
    for (int i = 0; i < values; i++) {
        data[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * h_output = (int *)malloc(sizeof(int) * values); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    

    // PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    int * buffer_output_1 = nullptr;
    int * buffer_output_2 = nullptr;
    int * buffer_output_3 = nullptr;

    int * buffer_input_1 = nullptr;
    int chunks = ((values + BLOCK_SIZE - 1) / BLOCK_SIZE);


     //allocate memory
     // not sure the size ?
    cudaMalloc(&buffer_output_1, sizeof(int) * values );
    cudaMalloc(&buffer_output_2, sizeof(int) * chunks );
    cudaMalloc(&buffer_output_3, sizeof(int) * chunks);
    cudaMalloc(&buffer_input_1, sizeof(int) * values );



    //PERFORM NECESSARY DATA TRANSFER HERE
    // copy from host to device
    cudaMemcpyAsync(buffer_input_1, data , sizeof(int) * values, cudaMemcpyHostToDevice, stream);



    cudaEventRecord(begin, stream);
    // K1
    dim3 block(BLOCK_SIZE);
    dim3 grid(chunks);
    scan1<<<grid, block, 0, stream>>>(buffer_output_1, buffer_input_1,buffer_output_2, values);

    // K2
    dim3 block2(BLOCK_SIZE);
    dim3 grid2(1);
    scan2<<<grid2, block2, 0, stream>>>(buffer_output_3, buffer_output_2, chunks);

    // K3
    scan3<<<grid,block,0,stream>>>(buffer_output_1, buffer_output_3, values);

    cudaEventRecord(end, stream);
 

    // PERFORM NECESSARY DATA TRANSFER HERE
    cudaMemcpyAsync(h_output, buffer_output_1, sizeof(int) * values, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    int * reference_output = serial_implementation(data, values);
    for (int i = 0; i < values; i++) {
        if (reference_output[i] != h_output[i]) {
            printf("ERROR: %d != %d at index %d\n", reference_output[i], h_output[i], i);
            abort();
        }
    }

    //FREE THE VARIABLES I USED
    cudaFree(buffer_output_1);
    cudaFree(buffer_output_2);
    cudaFree(buffer_output_3);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(reference_output);
    free(h_output);

    return 0;
}
