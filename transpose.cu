#include <stdio.h>
#include <fstream>
#include <iostream>
#include <omp.h>
//#define DEBUG

using namespace std;
extern double deviceCall_time;
//extern double omptime;

void cudasafe( cudaError_t error, char* message)
{
     if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

// assume matrix 1 should be transposed to matrix 2.
__global__ void transpose(double *mat1,double *mat2,const int row,const int col)
{
	int idx=threadIdx.x,idy=threadIdx.y;

	int globalBlkIndex=blockIdx.y*blockDim.y*col + blockIdx.x*blockDim.x;       //multiply by column 
	int globalIndexWrite=blockIdx.x*blockDim.x*row + blockIdx.y*blockDim.y;     //multiply by row

	__shared__ double shmat1[16][16];         //Shared mem equal to number of thread
//	__shared__ int shmat2[16][16]; 

	shmat1[idy][idx]=mat1[globalBlkIndex + idy*col+idx];    //Read Tile from mat1 row wise and write to shared memory row wise

	__syncthreads();

	mat2[globalIndexWrite + idy*row+idx]=shmat1[idx][idy];   //Read from Shared memory column wise and write to mat2 row wise

}

// This function recieves the original N*K matrix from the CPU and performs the Clustering on it.Also the values of N(Centroids) and K(data points) are passed into it. The Cpu also takes as input d_output where it will store the matrix in the transposed form. The dimensions of d_output is K*N
double* call_transpose(double *h_input,int centroids,int data_points)
{
	double *d_input,*d_output;
	
	cudasafe(cudaMalloc(&d_input,sizeof(double)*centroids*data_points),"Error in allocating memory to d_input");
	cudasafe(cudaMalloc(&d_output,sizeof(double)*centroids*data_points),"Error in allocating memory to d_output");

	cudasafe(cudaMemcpy(d_input,h_input,sizeof(double)*centroids*data_points,cudaMemcpyHostToDevice),"Error in copying data from the h_input to d_input");

	
	dim3 block(16,16);                      //multiple of 16 total 256 threads
	dim3 grid(centroids/16,data_points/16);   //number of blocks //
	
	//trans1<<<grid,block>>>(d_mat1,d_mat2,row,col);
	clock_t begin = clock();
	//omptime=omp_get_wtime();
	transpose<<<grid,block>>>(d_input,d_output,data_points,centroids);

	//cudasafe(cudaPeekAtLastError(),"errors occured at kernel");

	cudasafe(cudaDeviceSynchronize(),"Error in Transpose Kernel");
	//omptime=omp_get_wtime()-omptime;
	clock_t end = clock();
	//printf("Transpose time %lf\n",omptime);
	deviceCall_time += ((double)(end - begin)/CLOCKS_PER_SEC);

	//Below portion of the code is to print and check the value of the transposed matrix
       
	#ifdef DEBUG1

		fstream fout;

		double* h_output=(double *)malloc(sizeof(double)*data_points*centroids);  //declare an array in host to hold the transposed matrix values

		cudasafe(cudaMemcpy(h_output,d_output,sizeof(double)*data_points*centroids,cudaMemcpyDeviceToHost),"Copying from d_output to h_output");  //copy from d_output to h_output

		fout.open("outputtranspose.txt",ios::out);
		//printf("\n");
		for(int i=0;i<centroids;i++)
		{
			//printf("\n");
			fout << endl;

			for(int j=0;j<data_points;j++)
			{
				fout << h_output[i*data_points + j ]<< " ";
				//if(h_output[i*data_points + j] <=0 )
				//	printf("%f \n",h_output[i*data_points + j]);
			}
		}	

		fout<<endl;

		fout.close();

		free(h_output);
		

	#endif

	cudaFree(d_input);

	return d_output;
	
}
