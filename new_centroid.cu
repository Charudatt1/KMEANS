#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#define DEBUG
#define GRID_SIZE 1024

/*To find dimensions of new centroids
step 1: sum all documents belonging to same cluster
step 2: divide sum in step 1 by document count per cluster
*/
using namespace std;
extern double deviceCall_time;
//extern double omptime;
const unsigned long long offset=4294967295;

void cudasafe( cudaError_t error, char* message);

template <class T> const T& min (const T& a, const T& b) {
  return !(b<a)?a:b;     // or: return !comp(b,a)?a:b; for version (2)
}

//something


//To do sum of all documents belonging to same cluster
//OPTIMIZED KERNEL//

__global__ void centroid_cal(double *d_data,double *d_centroid,unsigned long long *d_min,int start_offset,int end_offset,int Bdim,int Dim,int size)
{
	
	extern __shared__ unsigned long long sharemin[];					//size of sharemin array is assumed to be equal to blockDim.x

	//unsigned long long offset2=4294967295;
	//int threadID=threadIdx.x;
	int gid=threadIdx.x + blockIdx.x*blockDim.x;						//get the global id

	int factor = (int)ceil((double)size/blockDim.x);	//To decide each thread will read how many elements from d_min 

	int centroid_num;
	int doc_num;
	
	for(int i=0;i<factor;i++)
	{
		int curroffset=start_offset + i*blockDim.x +threadIdx.x; 		//calculate the current offset for copying the values between start_offset to end_offset

		if(curroffset <= end_offset )									//TODO replace this condition with hack branching
			sharemin[ threadIdx.x ]=d_min[curroffset];
		else
			sharemin[ threadIdx.x ]=ULLONG_MAX;							//If no more values exist or we have crossed the end_offset

	//	__syncthreads();												//wait for the shared memory values to be filled


		for(int j=0; j<blockDim.x ;j++)												//Indefinite Loop
		{
			if( sharemin[j] == ULLONG_MAX )  							// No more values exist
				break;

			//count++;

			centroid_num = (int)(sharemin[j] & offset);
			doc_num = (int)((sharemin[j] >>32) & offset);

			if(gid<Dim)																			   //check global limit for dimension
					d_centroid[blockIdx.x*blockDim.x*Bdim + threadIdx.x*Bdim+centroid_num]+=d_data[doc_num*Dim + blockIdx.x*blockDim.x+threadIdx.x]; //Data is stored in d x k format
			
		}

	}

	//if(gid==0)
	//	printf("%d\n",count);

}



double *calculate_new_centroid(double *h_data,double *h_centroid,unsigned long long *h_min,unsigned long long *h_unique,int Adim,int Bdim,int dim)
{
		int i=0;
        double *d_data,*d_centroid;

        double *new_centroids=(double *)malloc(sizeof(double)*Bdim*dim);
        
        unsigned long long *d_min;//*d_unique;

        cudasafe(cudaMalloc(&d_data,sizeof(double)*Adim*dim),"Allocating mem in d_data in new centroid");
        cudasafe(cudaMalloc(&d_centroid,sizeof(double)*Bdim*dim),"Allocating mem in d_centroid in new centroid");
	    cudasafe(cudaMalloc(&d_min,sizeof(unsigned long long)*Adim),"Allocating mem in d_min in new centroid");
	    //cudaMalloc(&d_unique,sizeof(unsigned long long)*Bdim);

		int NUMBEROFSTREAM=min(16,(int)(ceil((double)480/dim)));

        cudaStream_t stream[NUMBEROFSTREAM];
        cudaMemcpy(d_data,h_data,sizeof(double)*Adim*dim, cudaMemcpyHostToDevice);
       // cudaMemcpy(d_unique,h_unique,sizeof(double)*Bdim, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_min,h_min,sizeof(unsigned long long)*Adim,cudaMemcpyHostToDevice);    
	
		//Create number of streams = number of centroids
	    for (i = 0; i < NUMBEROFSTREAM ;i++) 
	    {
	  	          cudaStreamCreate(&stream[i]);  	
		}

		int grid=((dim<=480)? 1 : ((int)ceil((double)dim/1024)));
	    //Number of threads equals to dimension or maximum value 1024
	    int block=min(dim,1024);
	    int size;

	    int len=(int)(ceil((double)Bdim/NUMBEROFSTREAM));
	    clock_t begin = clock();
	    //omptime=omp_get_wtime();
	    for(i=0;i<NUMBEROFSTREAM;i++) 
	    {
	    	if((i*len) >= Bdim )
	        	break;
	        	
	       	int start_offset=h_unique[i*len];
	       	//int last_pos=((((i+1)*len)-1) >=Bdim) ? (Bdim-1) : (h_unique((i+1)*len)-1);       		
	       	int end_offset=(((((i+1)*len)) >=Bdim) ? (Adim-1) : (h_unique[(i+1)*len]-1));       		
	        //int end_offset=h_unique[last_pos];
	        size=end_offset - start_offset + 1;		//size of each array for the shared memory	
           centroid_cal<<<grid,block,block*sizeof(unsigned long long),stream[i]>>>(d_data,d_centroid,d_min,start_offset,end_offset,Bdim,dim,size); //inclusive of start offset and inclusive of end offset
		}
		//omptime=omp_get_wtime()-omptime;
		//printf("New Centroid time %lf\n",omptime);
		clock_t end = clock();
		deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);
	        
        for(i=0;i<NUMBEROFSTREAM;i++) 
        {
        	//Wait till all streams complete their operations
            cudaStreamSynchronize(stream[i]);
        }
  	
	    for (i=0; i < NUMBEROFSTREAM; i++) 
        {
	        cudaStreamDestroy(stream[i]);
		}

		cudaDeviceSynchronize();

		cudaMemcpy(new_centroids,d_centroid,sizeof(double)*Bdim*dim, cudaMemcpyDeviceToHost);

	    //Free unwanted memory allocations in GPU    
		
		cudaFree(d_data);	
	    cudaFree(d_min);       
        cudaFree(d_centroid);
       // cudaFree(d_unique);
      /*  for(int i=0;i<(Bdim);i++) 
	    {
	    	
	    	for (int j = 0; j < dim; j++)
	    	{
	    		cout<<new_centroids[j*Bdim+i]<<" ";

	    	}
	    	cout<<endl;
	    	      
	    }*/   
        return new_centroids;
}



