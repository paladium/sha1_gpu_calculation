/*
 * SHA-1 benchmark program. Calculates execution time of SHA-1 on CPU and GPU.
 * Also includes function sha1_gpu_global() which prepares SHA-1 to be executed
 * on GPU.
 *
 * 2008, Tadas Vilkeliskis <vilkeliskis.t@gmail.com>
 */
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
 
 /*
  * Run sha1 kernel on GPU
  * input - message
  * size - message size
  * output - buffer to store hash value
  * proc - maximum threads per block
  */
 void sha1_gpu_global (unsigned char *input, unsigned long size, unsigned char *output, int proc)
 {
     int total_threads;		/* Total number of threads in the grid */
     int blocks_per_grid;		/* Number of blocks in the grid */
     int threads_per_block;		/* Number of threads in a block */
     int pad, size_be;		/* Number of zeros to pad, message size in big-enadian. */
     int total_datablocks;		/* Total number of blocks message is split into */
     int i, k;			/* Temporary variables */
     unsigned char *d_message;	/* Input message on the device */
     unsigned long *d_extended;	/* Extended blocks on the device */
     sha1_gpu_context ctx, *d_ctx;	/* Intermediate hash states */
 
     /* Initialization vector for SHA-1 */
     ctx.state[0] = 0x67452301;
     ctx.state[1] = 0xEFCDAB89;
     ctx.state[2] = 0x98BADCFE;
     ctx.state[3] = 0x10325476;
     ctx.state[4] = 0xC3D2E1F0;
 
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
     cudaMalloc ((void**)&d_ctx, sizeof (sha1_gpu_context));
 
     /*
      * Copy the data from host to device and perform padding
      */
     cudaMemcpy (d_ctx, &ctx, sizeof (sha1_gpu_context), cudaMemcpyHostToDevice);
     cudaMemcpy (d_message, input, size, cudaMemcpyHostToDevice);
     cudaMemset (d_message + size, 0x80, 1);
     cudaMemset (d_message + size + 1, 0, pad + 7);
     cudaMemcpy (d_message + size + pad + 4, &size_be, 4, cudaMemcpyHostToDevice);
 
     /*
      * Run the algorithm
      */
     i = 0;
     k = total_datablocks / total_threads;
     printf("%d %d\n", total_datablocks, total_threads);

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
 
     cudaMemcpy (&ctx, d_ctx, sizeof(sha1_gpu_context), cudaMemcpyDeviceToHost);
 
     printf("%d %d %d %d %d\n", ctx.state[0], 
    ctx.state[1],
    ctx.state[2],
    ctx.state[3],
    ctx.state[4]
    );

     /* Put the hash value in the users' buffer */
     PUT_UINT32_BE( ctx.state[0], output,  0 );
     PUT_UINT32_BE( ctx.state[1], output,  4 );
     PUT_UINT32_BE( ctx.state[2], output,  8 );
     PUT_UINT32_BE( ctx.state[3], output, 12 );
     PUT_UINT32_BE( ctx.state[4], output, 16 );
     cudaFree (d_message);
     cudaFree (d_ctx);
     cudaFree (d_extended);
 }
 
 
//  int main(int argc, char *argv[])
//  {
//      unsigned char hash[20];
//      unsigned char *data = NULL;
//      int i;
//      int max_threads_per_block = MAX_THREADS_PER_BLOCK;
//      unsigned int nonce = 0;
//      bool done = false;
//      //data = (unsigned char *) malloc (100);
//      if(argc <= 2)
//      {
//          printf("Give 2 arguments");
//          exit(1);
//      }
//      int maxIterations = atoi(argv[1]);
//      int difficulty = atoi(argv[2]);
//      for(int j=0;j<maxIterations;j++)
//      {
//         memset (hash, 0, 20);
//         char t[20];
//         itoa(j, t, 10);
//         char data[100] = "cgCevIRCeDhqIJExqnjwidTkKHeGOPXYgviPiwZhOImJbvKZUGEkjkrkHQPoSlFl";
//         for(int k=0;k<strlen(t);k++)
//         {
//             data[k + 64] = t[k];
//         }
//         sha1_gpu_global ((unsigned char *)data, strlen(data), hash, max_threads_per_block);
//         char hashHex[20];
//         sprintf(hashHex, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", 
//         (unsigned int)hash[0], 
//         (unsigned int)hash[1], 
//         (unsigned int)hash[2], 
//         (unsigned int)hash[3],
//         (unsigned int)hash[4], 
//         (unsigned int)hash[5], 
//         (unsigned int)hash[6], 
//         (unsigned int)hash[7], 
//         (unsigned int)hash[8], 
//         (unsigned int)hash[9], 
//         (unsigned int)hash[10], 
//         (unsigned int)hash[11], 
//         (unsigned int)hash[12], 
//         (unsigned int)hash[13], 
//         (unsigned int)hash[14],
//         (unsigned int)hash[15], 
//         (unsigned int)hash[16], 
//         (unsigned int)hash[17], 
//         (unsigned int)hash[18], 
//         (unsigned int)hash[19]
//     );
//         done = true;
//         for(i=0;i<difficulty;i++)
//             done = done && hashHex[i] == '0';
//         if(done){
//             printf("Solution: %d ", j);
//             printf("%s", hashHex);
//             break;
//         }
//      }
 
//     //  for (i = 1000; i < 100000000; i = i * 10) {
//     //      data = (unsigned char *) malloc (i);
//     //      if (data == NULL) {
//     //          printf ("ERROR: Insufficient memory on host\n");
//     //          return -1;
//     //      }
 
//     //      sha1_cpu (data, i, hash); 
//     //      memset (hash, 0, 20);
 
//     //      sha1_gpu_global (data, i, hash, max_threads_per_block);
//     //      free (data);
//     //  }
 
//      return 0;
//  }
int main(void)
{
    char data[100] = "hello2";
    unsigned char hash[20];
    sha1_gpu_global ((unsigned char *)data, strlen(data), hash, 1);
    char hashHex[20];
    sprintf(hashHex, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", 
        (unsigned int)hash[0], 
        (unsigned int)hash[1], 
        (unsigned int)hash[2], 
        (unsigned int)hash[3],
        (unsigned int)hash[4], 
        (unsigned int)hash[5], 
        (unsigned int)hash[6], 
        (unsigned int)hash[7], 
        (unsigned int)hash[8], 
        (unsigned int)hash[9], 
        (unsigned int)hash[10], 
        (unsigned int)hash[11], 
        (unsigned int)hash[12], 
        (unsigned int)hash[13], 
        (unsigned int)hash[14],
        (unsigned int)hash[15], 
        (unsigned int)hash[16], 
        (unsigned int)hash[17], 
        (unsigned int)hash[18], 
        (unsigned int)hash[19]
    );
    printf("%s", hashHex);
}