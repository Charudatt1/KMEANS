#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#define TILESIZE 16

using namespace std;
extern double deviceCall_time;
//extern double omptime;

//Dsub is data array
//Csub is centroid array
//Rsub is result array
//commonDin is the dimension of the data points
//Adim is the number  of the data points
//Bdimension number of cluster centers.
//dim_title is the tilesize of dimension array
void cudasafe( cudaError_t error, char* message);
__global__ void mul(double *Dsub,double *Csub,double *Rsub,int commondim,int Adim,int Bdim,int dim_tile)
{
	 __shared__ double A[TILESIZE][TILESIZE];	//Declare shared memory array
	 __shared__ double B[TILESIZE][TILESIZE];

	double Cval=0;

	int gidx=blockDim.x*blockIdx.x + threadIdx.x;   //global index in x dimension
	int gidy=blockDim.y*blockIdx.y + threadIdx.y;   //global index in y dimension

	int rindex= gridDim.x*blockDim.x*gidy + gidx;   //global index for Rsub

	int dindex,cindex;             
	

	for(int m=0;m<((int)ceil( ((double)commondim)/TILESIZE ) );++m)
	{

		 dindex=blockIdx.y*commondim*blockDim.y + blockDim.x*m;	//IF d<16 then m always 0 ,hence blockDim.x*m is always 0. If d>=16 then it means d is multiple of 16 and thus blockDim.x=16
		 cindex=m*Bdim*blockDim.x + blockDim.y*blockIdx.x;

		 if(threadIdx.x < dim_tile)                             //limited by D
			A[threadIdx.y][threadIdx.x]=Dsub[dindex + threadIdx.y*commondim + threadIdx.x];	
	
		if(threadIdx.y < dim_tile)								//limited by D
			B[threadIdx.y][threadIdx.x]=Csub[cindex + threadIdx.y*Bdim + threadIdx.x];

		__syncthreads();
		
		for(int i=0;i<dim_tile;++i)
		{
			Cval+=(A[threadIdx.y][i]-B[i][threadIdx.x])*(A[threadIdx.y][i]-B[i][threadIdx.x]);
		}

		__syncthreads();

		//if(blockIdx.x==0 && blockIdx.y==0 )
			

	}
//	if(blockIdx.x==0 && blockIdx.y==127)
//	printf(" %f %d %d %d\n",Cval,row*Bdim+col,blockIdx.x,blockIdx.y);
	//printf("%d\n",rindex);
	Rsub[rindex]=sqrt(Cval);


}

int call_mul(double *h_data,double *h_centroid,double *h_result,int commondim,int Adim,int Bdim)
{
	int dim_tile;
	
	int gridx=Bdim/16,gridy=Adim/16;
	double *d_data,*d_centroid,*d_result;
	//printf("\n hi");
	cudasafe(cudaMalloc(&d_data,sizeof(double)*Adim*commondim),"Allocate mem for d_data in matrix mul");
	cudasafe(cudaMalloc(&d_centroid,sizeof(double)*commondim*Bdim),"Allocate mem for d_centroid in matr mul");
	cudasafe(cudaMalloc(&d_result,sizeof(double)*Adim*Bdim),"Allocate mem for d_result in mat mul");

	cudaMemcpy(d_data,h_data,sizeof(double)*Adim*commondim,cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroid,h_centroid,sizeof(double)*commondim*Bdim,cudaMemcpyHostToDevice);
	cudaMemcpy(d_result,h_result,sizeof(double)*Adim*Bdim,cudaMemcpyHostToDevice);


	dim3 grid(gridx,gridy);
	dim3 block(TILESIZE,TILESIZE);
	if(commondim < 16)
	{
		//size=sizeof(double)*TILESIZE*commondim;
		dim_tile=commondim;
	}
	else
	{
		//size=sizeof(double)*TILESIZE*TILESIZE;
		dim_tile=TILESIZE;
	}
	//timer.Start();
	clock_t begin = clock();
	//omptime=omp_get_wtime();
	mul<<<grid,block>>>(d_data,d_centroid,d_result,commondim,Adim,Bdim,dim_tile);	
	//timer.Stop();
	cudaDeviceSynchronize();
	//omptime=omp_get_wtime()-omptime;
	//printf("Matrix Multiplication time %lf\n",omptime);
	clock_t end = clock();
	deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);
	
	cudaMemcpy(h_result,d_result,sizeof(double)*Adim*Bdim,cudaMemcpyDeviceToHost);

	
	//printf("\n Time=%g",timer.Elapsed());
	cudaFree(d_data);
	cudaFree(d_centroid);
	cudaFree(d_result);
	return 0;
}
