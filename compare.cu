#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
//#include "gputimer.h"
#define DEBUG
#define THRESHOLD 1

using namespace std;

extern double deviceCall_time;
void cudasafe( cudaError_t error, char* message);
//extern double omptime;
__global__ void comp_cent(double *d_centold,double *d_centnew,double basevalue,int *flag)
{
	__shared__ int *s_flag;
	s_flag=flag;

	double base=basevalue;

	double diff=abs(d_centnew[blockIdx.x*blockDim.x+threadIdx.x]-d_centold[blockIdx.x*blockDim.x+threadIdx.x]);

	if(base < diff)
		*s_flag=0;

}
int compare_centroids(double *h_centroid,double *new_centroid,int Bdim,int dim)
{
	double *d_centold,*d_centnew;
	int tmp=1;
	int *h_flag=&tmp; //(int *)malloc(sizeof(int));
	int *d_flag;
	double thre_shold=THRESHOLD;

	cudasafe(cudaMalloc(&d_centold,sizeof(double)*Bdim*dim),"Error compare d_centold");
	cudasafe(cudaMalloc(&d_centnew,sizeof(double)*Bdim*dim),"Error compare d_centnew");
	cudasafe(cudaMalloc(&d_flag,sizeof(int)),"Error compare d_flag");
	

	cudaMemcpy(d_centold,h_centroid,sizeof(double)*Bdim*dim,cudaMemcpyHostToDevice);
	cudaMemcpy(d_centnew,new_centroid,sizeof(double)*Bdim*dim,cudaMemcpyHostToDevice);
	cudaMemcpy(d_flag,h_flag,sizeof(int),cudaMemcpyHostToDevice);

	int size=Bdim*dim;
	int block=min(1024,size);					//minimum of 1024 or total numer of cells in h_centroid matrix
	int grid=(int)ceil((double)size/block);		

	clock_t begin = clock();
	//omptime=omp_get_wtime();
	comp_cent<<<grid,block>>>(d_centold,d_centnew,thre_shold,d_flag);
	//omptime=omp_get_wtime()-omptime;
	clock_t end = clock();
	//printf("Compare time %lf\n",omptime);
	deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);

	cudaDeviceSynchronize();
	cudaMemcpy(h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost);

	//Free Memory

	cudaFree(d_centold);
	cudaFree(d_centnew);
	cudaFree(d_flag);


	return (*h_flag);
}