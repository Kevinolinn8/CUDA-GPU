#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "util.h"



//BELOW IS MY KERNEL IMPLEMENTATION COUNT

__global__ void count(int rows,int cols,int * count,int * data) {

    __shared__ int shareCount;
	
   if(threadIdx.x == 0){
	shareCount = 0;
   }

    int my_index = blockDim.x * blockIdx.x + threadIdx.x;
   
   __syncthreads(); // place barrier for threads
   
   //IF STATEMENT TO ACCESS INDECIES OF ARRAY 
   if(my_index < cols*rows){
     if(data[my_index] == 1){
	atomicAdd(&shareCount, 1);
     }
   }
   __syncthreads(); //Another barrier

   
   if (threadIdx.x == 0){
    
    atomicAdd(count, shareCount);
   
   }
   
}



int serial_implementation(int * data, int rows, int cols) {
    int count = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (data[i] == 1) count++;
    }

    return count;
}

int main(int argc, char ** argv) {
    
    int rows = 0, cols = 0;

    assert(argc == 2);
    int * data = read_file(argv[1], &rows, &cols);

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int count_h = 0; // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    

 
    
    //ALL OF MY VARIBALE DECLARATIONS
    int *MyCount;
    int *MyData;
    int rowsXcols = (rows*cols) * sizeof(int);
    cudaMalloc(&MyCount, sizeof(int));
    cudaMalloc(&MyData, rowsXcols); //allocate memory
    
    //COPYING VARIABLES FROM HOST TO GPU
    cudaMemcpyAsync(MyData, data, rowsXcols, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(MyCount, &count_h, sizeof(int), cudaMemcpyHostToDevice, stream);


    //BLOCK SIZES
    int BLOCK_SIZE = 128;
    int BLK = (rowsXcols + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(BLK);
    

   

    //LAUNCH KERNEL
    cudaEventRecord(begin, stream);
    count<<<grid,block,0,stream>>>(rows,cols,MyCount,MyData);
    cudaEventRecord(end, stream);
    
    

    //NECESSARY DATA TRANSFER! GPU BACK TO HOST
    cudaMemcpyAsync(&count_h, MyCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    
    /*

    DEALLOCATE RESOURCES HERE

    */

    int count_serial = serial_implementation(data, rows, cols);
    if (count_serial != count_h) {
        printf("ERROR: %d != %d\n", count_serial, count_h);
    }
    
    
    //FREE THE VARIABLES I USED
    cudaFree(MyData);
    cudaFree(MyCount);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);

    return 0;
}
