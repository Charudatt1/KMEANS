//Creating 4 streams, each assigns a local thread index to the array
//
#include <stdio.h>
#include <stdlib.h>

#define N 16
#define NCHUNK 2

//__device__ int *data;
__global__ 
void thread_multi(int t1,int *data)
{
        int i=blockDim.x * blockIdx.x + threadIdx.x;
        int j=threadIdx.x;
        printf(" %d %d\n",data[t1+i],threadIdx.x);
}

int main()
{
        int i=0;
        int *data;      
        cudaStream_t stream[NCHUNK];
        size_t size = N*NCHUNK*sizeof(int);

       // cudaMalloc((void **)&d_t1, size);
       // cudaMallocHost(&h_t1, size);
        
        int *h_data=(int *)malloc(sizeof(int)*32);
        for(i=0;i<32;i++)
                h_data[i]=i;
        
        cudaMalloc(&data,sizeof(int)*32);
        cudaMemcpy(data,h_data,sizeof(int)*32,cudaMemcpyHostToDevice);


        //for (i=0;i<N*NCHUNK;i++) {
          //      h_t1[i]=0;
        //}

//Create 4 streams
        for (i = 0; i < NCHUNK;i++) {
                cudaStreamCreate(&stream[i]);
	}

//4 events on each stream - Memory copy to the device, execution, memory copy to the host, stream destroyed
    //    for(i=0;i<NCHUNK;i++) {
      //          cudaMemcpyAsync(d_t1+i*N, h_t1+i*N, N*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
//	}
 //       for(i=0;i<NCHUNK;i++) {
   //             cudaStreamSynchronize(stream[i]);
     //   }

        for(i=0;i<NCHUNK;i++) {
                thread_multi<<<1,16,0,stream[i]>>>(i*16,data);
	}
        
        for(i=0;i<NCHUNK;i++) {
                cudaStreamSynchronize(stream[i]);
        }

      /*  for(i=0;i<NCHUNK;i++) {
                cudaMemcpyAsync(h_t1+i*N, d_t1+i*N, N*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
	}

        for(i=0;i<NCHUNK;i++) {
                cudaStreamSynchronize(stream[i]);
        }*/
        for (i=0; i < NCHUNK; i++) {
                cudaStreamDestroy(stream[i]);
	}
        
//Print result
       /* for(i=0;i<N*NCHUNK;i++) {
                printf("%d: %d\n",i, h_t1[i]);
        }*/       
       // cudaFree(d_t1);
        cudaFree(data);
        printf("\nDone\n");
        return 0;
}
