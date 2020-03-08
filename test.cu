#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "common.h"
#include <cstdlib>
#define MAX_THREADS_PER_BLOCK 128

typedef struct {
    unsigned long state[5];
} sha1_gpu_context;


extern __global__ void sha1_kernel_global (unsigned char *data, sha1_gpu_context *ctx, int total_threads, unsigned long *extended);

__global__ void gpuSquare(float *d_in, float *d_out)
{
    int tid = blockIdx.x;

    float temp = d_in[tid];
    d_out[tid] = temp * temp;
}

__global__ void findSolution(unsigned char* data, int size, unsigned char* output)
{
    const int proc = 1;
    unsigned long *d_extended;
    sha1_gpu_context *ctx;
    cudaMalloc ((void**)&ctx, sizeof (sha1_gpu_context));
    cudaMalloc ((void**)&d_extended, proc * 80 * sizeof(unsigned long));
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    int threads_per_block = proc;
    sha1_kernel_global<<<1, 1>>>(data, ctx, threads_per_block, d_extended);
    PUT_UINT32_BE( ctx->state[0], output,  0 );
    PUT_UINT32_BE( ctx->state[1], output,  4 );
    PUT_UINT32_BE( ctx->state[2], output,  8 );
    PUT_UINT32_BE( ctx->state[3], output, 12 );
    PUT_UINT32_BE( ctx->state[4], output, 16 );
}


__global__ void findSolution2(unsigned char *input, int size, int proc)
{
    int total_threads;		/* Total number of threads in the grid */
    int blocks_per_grid;		/* Number of blocks in the grid */
    int threads_per_block;		/* Number of threads in a block */
    int pad, size_be;		/* Number of zeros to pad, message size in big-enadian. */
    int total_datablocks;		/* Total number of blocks message is split into */
    int i, k;			/* Temporary variables */
    unsigned char *d_message;	/* Input message on the device */
    unsigned long *d_extended;	/* Extended blocks on the device */
    unsigned char output[20];
    memset(output, 0, 20);
    sha1_gpu_context *d_ctx;	/* Intermediate hash states */
    cudaMalloc ((void**)&d_ctx, sizeof (sha1_gpu_context));

    /* Initialization vector for SHA-1 */
    d_ctx->state[0] = 0x67452301;
    d_ctx->state[1] = 0xEFCDAB89;
    d_ctx->state[2] = 0x98BADCFE;
    d_ctx->state[3] = 0x10325476;
    d_ctx->state[4] = 0xC3D2E1F0;

    pad = padding_256 (size);
    threads_per_block = proc;
    blocks_per_grid = 1;
    /* How many blocks in the message */
    total_datablocks = (size + pad + 8) / 64;

    if (total_datablocks > threads_per_block)
        total_threads = threads_per_block;
    else
        total_threads = total_datablocks;
    
    size_be = LETOBE32 (size * 8);

    /* Allocate enough memory on the device */
    cudaMalloc ((void**)&d_extended, proc * 80 * sizeof(unsigned long));
    cudaMalloc ((void**)&d_message, size + pad + 8);

    /*
    * Copy the data from host to device and perform padding
    */
    memcpy (d_message, input, size);
    memset (d_message + size, 0x80, 1);
    memset (d_message + size + 1, 0, pad + 7);
    memcpy (d_message + size + pad + 4, &size_be, 4);

    /*
    * Run the algorithm
    */
    i = 0;
    k = total_datablocks / total_threads;
    if (k - 1 > 0) {
        /*
        * Kernel is executed multiple times and only one block in the grid is used.
        * Since thread synchronization is allowed only within a block.
        */
        for (i = 0; i < k; i++) {
            sha1_kernel_global <<<blocks_per_grid, proc>>>(d_message + threads_per_block * i * 64, d_ctx, threads_per_block, d_extended);
            /*
            * Here I do not perform thread synchronization
            * since threads are shynchronized in the kernel
            */
        }
    }
    threads_per_block = total_datablocks - (i * total_threads);
    sha1_kernel_global <<<blocks_per_grid, proc>>>(d_message + total_threads * i * 64, d_ctx, threads_per_block, d_extended);
    printf("%d %d %d %d %d\n", d_ctx->state[0], 
    d_ctx->state[1],
    d_ctx->state[2],
    d_ctx->state[3],
    d_ctx->state[4]
    );
    /* Put the hash value in the users' buffer */
    PUT_UINT32_BE( d_ctx->state[0], output,  0 );
    PUT_UINT32_BE( d_ctx->state[1], output,  4 );
    PUT_UINT32_BE( d_ctx->state[2], output,  8 );
    PUT_UINT32_BE( d_ctx->state[3], output, 12 );
    PUT_UINT32_BE( d_ctx->state[4], output, 16 );
    cudaFree (d_message);
    cudaFree (d_ctx);
    cudaFree (d_extended);

    //print this
    printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", 
    (unsigned int)output[0], 
    (unsigned int)output[1], 
    (unsigned int)output[2], 
    (unsigned int)output[3],
    (unsigned int)output[4], 
    (unsigned int)output[5], 
    (unsigned int)output[6], 
    (unsigned int)output[7], 
    (unsigned int)output[8], 
    (unsigned int)output[9], 
    (unsigned int)output[10], 
    (unsigned int)output[11], 
    (unsigned int)output[12], 
    (unsigned int)output[13], 
    (unsigned int)output[14],
    (unsigned int)output[15], 
    (unsigned int)output[16], 
    (unsigned int)output[17], 
    (unsigned int)output[18], 
    (unsigned int)output[19]
    );
}



int main(void)
{
    // float h_in[N], h_out[N];

    // float *d_in, *d_out;

    // cudaMalloc((void**)&d_in, N * sizeof(float));
    // cudaMalloc((void**)&d_out, N * sizeof(float));
    // for(int i=0;i<N;i++)
    // {
    //     h_in[i] = i;
    // }

    // cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    // gpuSquare <<<N, 1>>>(d_in, d_out);
    // cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("Square of number of gpu \n");
    // for(int i=0;i<N;i++)
    // {
    //     printf("The square of %f is %f\n", h_in[i], h_out[i]);
    // }
    // cudaFree(d_in);
    // cudaFree(d_out);
    char data[100] = "hello2";
    unsigned char *d_data;
    cudaMalloc((void**)&d_data, sizeof(data));
    cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);
    findSolution2<<<1, 1>>>(d_data, strlen(data), 1);
}