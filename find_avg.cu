#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#define DEBUG
#define GRID_SIZE 1024

using namespace std;
extern double deviceCall_time;
void cudasafe( cudaError_t error, char* message);
//extern double omptime;
//To divide sum of dimensions of documents belonging to same cluster by number of documents per cluster
__global__ void findAvgDevice(double *d_centroid,unsigned long long *d_unique,int Bdim,int Adim,int dim)
{

	int i;
	unsigned long long val;
	int id=blockDim.x*blockIdx.x+threadIdx.x;
	//TODO try to avoid thread divergence
	if(id+1 < Bdim)
		val =(d_unique[id+1]-d_unique[id]);
	else
		val =((Adim)-d_unique[id]);
	
	for(i=0;i<dim;i++)
	{
		d_centroid[i*Bdim+id]/=val;
	}
}

double *findAvg(double *h_centroid,unsigned long long *h_unique,int Adim,int Bdim,int dim)
{

	double *d_centroid;
	unsigned long long *d_unique;
	double *new_centroids=(double *)malloc(sizeof(double)*Bdim*dim);
	
	cudasafe(cudaMalloc(&d_centroid,sizeof(double)*Bdim*dim),"Error findAvg d_centroid");
	cudasafe(cudaMalloc(&d_unique,sizeof(unsigned long long)*Bdim),"Error findAvg d_unique");

	cudaMemcpy(d_unique,h_unique,sizeof(unsigned long long)*Bdim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroid,h_centroid,sizeof(double)*Bdim*dim, cudaMemcpyHostToDevice);

		/*

	    for(int i=0;i<(Bdim);i++) 
	    {
			cout<<h_unique[i]<<" ";	    	
	    }
	    cout<<endl;
	    */
	int grid=(int)ceil((double)Bdim/1024);
	int block=min(1024,Bdim);
   	//block-1 to avoid thread divergence in kernel
	clock_t begin = clock();
	//omptime=omp_get_wtime();
   	findAvgDevice<<<grid,block>>>(d_centroid,d_unique,Bdim,Adim,dim);  
   	cudaDeviceSynchronize();
   	//omptime=omp_get_wtime()-omptime;
   	clock_t end = clock();
	//printf("Find average time %lf\n",omptime);
	deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);
   	cudaMemcpy(new_centroids,d_centroid,sizeof(double)*Bdim*dim, cudaMemcpyDeviceToHost);

/*
   for(int i=0;i<(Bdim);i++) 
	    {
	    	
	    	for (int j = 0; j < dim; j++)
	    	{
	    		cout<<new_centroids[j*Bdim+i]<<" ";

	    	}
	    	cout<<endl;
	    	      
	    }
	    */

	cudaFree(d_centroid);
	cudaFree(d_unique);

   	return new_centroids;
}