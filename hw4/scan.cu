#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>


constexpr int BLOCK_SIZE = 1024;


//KERNEL 1
__global__ void scan1(int *data_out, int *data_in, int *max, int N) {
      
        __shared__ int partial[BLOCK_SIZE * 2]; 
        
        int my_index = blockIdx.x * blockDim.x + threadIdx.x;
        int myThread = threadIdx.x;

        int buf_output = 0;
        int buf_input = 1;

        if (my_index > 0 && my_index < N) {
         partial[buf_output * BLOCK_SIZE + myThread] = data_out[my_index-1];
        }else{
         partial[buf_output * BLOCK_SIZE + myThread] = 0;
        }
       
       __syncthreads();          

        for (int i = 1; i < BLOCK_SIZE; i *= 2)   {
                buf_output = 1 - buf_output;
                buf_input = 1 - buf_output;

            if (myThread >= i){
                partial[buf_input * BLOCK_SIZE + myThread] = partial[buf_input * BLOCK_SIZE + myThread - i] + partial[buf_input * BLOCK_SIZE + myThread];
            }else{
                partial[buf_output * BLOCK_SIZE + myThread] = partial[buf_input*BLOCK_SIZE+myThread];
            }
            __syncthreads();                    

        }
        if (my_index < N) {
            data_out[my_index] = partial[buf_output * BLOCK_SIZE + myThread];
        }
        // Write the max value to a separate array   
        if(myThread == 0){
            max[blockIdx.x] = partial[buf_output * BLOCK_SIZE + BLOCK_SIZE -1];
        }
    }

//KERNEL 2
__global__ void scan2(int *data_out, int *data_in, int N) {
        __shared__ int partial[BLOCK_SIZE * 2]; 
        
        int my_index = blockIdx.x * blockDim.x + threadIdx.x;
        int myThread = threadIdx.x;
        // takes in two shared memory buffers

        int buf_output = 0;
        int buf_input = 1;

        if (my_index > 0 && my_index < N) {
         partial[buf_output * BLOCK_SIZE + myThread] = data_out[my_index-1];
        }else{
         partial[buf_output * BLOCK_SIZE + myThread] = 0;
        }
       
       __syncthreads();          

        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2)   {
                buf_output = 1 - buf_output;
                buf_input = 1 - buf_output;

            if (myThread >= offset){
                partial[buf_input * BLOCK_SIZE + myThread] = partial[buf_input * BLOCK_SIZE + myThread - offset] + partial[buf_input * BLOCK_SIZE + myThread];
            }else{
                partial[buf_output * BLOCK_SIZE + myThread] = partial[buf_input*BLOCK_SIZE+myThread];
            }
            __syncthreads();                    

        }
        if (my_index < N) {
            data_out[my_index] = partial[buf_output * BLOCK_SIZE + myThread];
        }

    }

//KERNEL 3
__global__ void scan3(int *input_data, int *maxes, int N ) {
    
    int my_index =  blockIdx.x * blockDim.x + threadIdx.x;
    if(my_index < N) {
        input_data[my_index] += maxes[blockIdx.x]; 
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

    

    //VARIABLE DECLARATIONS
    int * buf_output_1 = nullptr;
    int * buf_input_1 = nullptr;
        
    int * buf_output_2 = nullptr;
    int * buf_output_3 = nullptr;
    
    int chunks = ((values + BLOCK_SIZE - 1) / BLOCK_SIZE);


     //allocate memory
    cudaMalloc(&buf_output_1, sizeof(int) * values );
    cudaMalloc(&buf_output_2, sizeof(int) * chunks );
    cudaMalloc(&buf_output_3, sizeof(int) * chunks);
    cudaMalloc(&buf_input_1, sizeof(int) * values );



    // copy from host to device
    cudaMemcpyAsync(buf_input_1, data , sizeof(int) * values, cudaMemcpyHostToDevice, stream);



    cudaEventRecord(begin, stream);
        
        
    // KERNEL1
    dim3 block(BLOCK_SIZE);
    dim3 grid(chunks);
    scan1<<<grid, block, 0, stream>>>(buf_output_1, buf_input_1,buf_output_2, values);

    // KERNEL2
    dim3 block2(BLOCK_SIZE);
    dim3 grid2(1);
    scan2<<<grid2, block2, 0, stream>>>(buf_output_3, buf_output_2, chunks);

    // KERNEL3
    scan3<<<grid,block,0,stream>>>(buf_output_1, buf_output_3, values);

    cudaEventRecord(end, stream);
 
    cudaMemcpyAsync(h_output, buf_output_1, sizeof(int) * values, cudaMemcpyDeviceToHost, stream);

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

    //FREE CUDAFREE
    cudaFree(buf_output_1);
    cudaFree(buf_input_1);
    cudaFree(buf_output_2);
    cudaFree(buf_output_3);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(reference_output);
    free(h_output);

    return 0;
}
