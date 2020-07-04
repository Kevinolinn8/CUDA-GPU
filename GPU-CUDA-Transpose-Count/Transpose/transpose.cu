#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "math.h"
#include "util.h"

//BELOW IS MY KERNEL FOR TRANSPOSE

__global__ void Transpose(int *matrix_in, int* matrix_out, int rows,int cols) {

   int x-axis = blockDim.x * blockIdx.x + threadIdx.x;
   int y-axis = blockDim.y * blockIdx.y + threadIdx.y;
   
   //IF STATEMENT FOR TRANSPOSE OPERATION

   if(x-axis < rows && y-axis < cols){
   
     matrix_out[x-axis * rows + y-axis] = matrix_in[y-axis* cols + x-axis];
     
     }
   
}


int * serial_implementation(int * data_in, int rows, int cols) {
    int * out = (int *)malloc(sizeof(int) * rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = data_in[i * cols + j];
        }
    }
    return out;
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

    int * transpose_h = (int *)malloc(sizeof(int) * rows * cols); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

   //DECLARE VARIABLES AND MEMORY
    int *matrixIN;
    int *matrixOUT;
    int rowsXcols = (rows*cols) * sizeof(int);
    
    //ALLOCATING MEMORY FOR THE VARIABLES
    cudaMalloc(&matrixIN, rowsXcols);
    cudaMalloc(&matrixOUT, rowsXcols);
    
    
    //COPY VARIABLES FROM DEVICE TO THE GPU
    cudaMemcpyAsync(matrixIN, data, rowsXcols, cudaMemcpyHostToDevice, stream);
    
 
 
 
    //block sizes
    
    dim3 block(32,32);
    dim3 grid((rows + 31) / 32, (cols + 31) / 32);
    

    
    //LAUNCH THE KERNEL
    cudaEventRecord(begin, stream);
    Transpose<<<grid,block,0,stream>>>(matrixIN,matrixOUT,rows,cols);
    cudaEventRecord(end, stream);



    //COPY MEMORY FROM GPU BACK TO HOST
    cudaMemcpyAsync(transpose_h, matrixOUT, rowsXcols, cudaMemcpyDeviceToHost, stream);
    
  
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);


    //DEALLOCATE RESOURCES HERE
    int * transpose_serial = serial_implementation(data, rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        if (transpose_h[i] != transpose_serial[i]) {
            printf("ERROR: %d != %d\n", transpose_serial[i], transpose_h[i]);
            exit(-1);
        }
    }
    
    //FREE MATRIX VARIABLES
    cudaFree(matrixIN);
    cudaFree(matrixOUT);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(transpose_serial);
    free(transpose_h);

    return 0;
}
