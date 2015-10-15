#include <stdio.h>
#include <float.h>
#include <fstream>
#include "gputimer.h"
#include <bitset>
#include <omp.h>

//#define DEBUG1

using namespace std;
extern double deviceCall_time;
//extern double omptime;

void cudasafe( cudaError_t error, char* message);

//minimization method takes in the Input array whose dimensions are K*N(i.e centroids * data_points)
//minCentroid is an array of size data_points . Each index in the minCentroid points to one value between 0- (centroids-1)

__global__ void minimization(double *d_Input,int centroids,int data_points,unsigned long long *d_minCentroid)
{
	double minimum=FLT_MAX;               //Variable to store the Minimum value initialized with max value of double
	unsigned int minIndex=0;						 //Variable to contain the index of the minimum value
	double val;							 //variable to store the value from the array

	unsigned long long gid=blockIdx.x*blockDim.x + threadIdx.x;

	for(int i=0;i<centroids;i++)
	{
		
		if( gid < data_points)
			val = d_Input[i*data_points + gid];
		else
			val = FLT_MAX;
	
	//	int z= ( data_points - 1 - (int)gid)>>31;       ///performance improvement achieved by removing if statement while reading
   // 	val = (1-z)*d_Input[i*data_points + (int)gid] + z*FLT_MAX;

		if(val < minimum)
		{
			minimum = val;
			minIndex = i;
		}
	}

	if((int)gid < data_points )
		d_minCentroid[gid] = gid<<32 | minIndex;


}


 unsigned long long* compute_minimization(double* d_Input,int data_points,int centroids)
 {
 	unsigned long long *d_minCentroid,*h_minCentroid;     //declare device and global memories
 
 	h_minCentroid=(unsigned long long*)malloc(sizeof(unsigned long long)*data_points);   //declare global centroid

 	cudasafe(cudaMalloc(&d_minCentroid,sizeof(unsigned long long)*data_points),"allocating memory in d_minCentroid");    //allocate memory for the device
  
 	dim3 block(1024);                                      //global block threads size is 1024
 	dim3 grid((int)(ceil((double)data_points/1024)));	   //grid size is datapoints/1024

 	clock_t begin = clock();
 	//omptime=omp_get_wtime();
 	minimization<<<block,grid>>>(d_Input,centroids,data_points,d_minCentroid); //call to function
 	clock_t end = clock();
 	//omptime=omp_get_wtime()-omptime;
 	//printf("Minimization time %lf\n",omptime);
 	
	deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);

 	cudasafe(cudaMemcpy(h_minCentroid,d_minCentroid,sizeof(unsigned long long)*data_points,cudaMemcpyDeviceToHost),"Copying from d_minCentroid to h_minCentroid");

 	cudasafe(cudaFree(d_minCentroid),"freeing memory");
 	cudasafe(cudaFree(d_Input),"Free d_input function name 'compute_minimization' line 70");

 	#ifdef DEBUG1

		fstream fout;

		fout.open("minimumcentroid.txt",ios::out);
		//printf("\n");

		//4294967295
		unsigned long long offset=4294967295;

		for(int i=0;i<data_points;i++)
		{
																						//std::bitset<64> x(h_minCentroid[i]);
			
			fout << ((h_minCentroid[i]) & offset) << endl; //print into a file
																						//fout << x << endl; 
		
			//printf("%llu \n",((h_minCentroid[i]) & (unsigned long long)4294967295));
		}	

		fout<<endl;

		fout.close();

	#endif


 	return h_minCentroid;
 }
