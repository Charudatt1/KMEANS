#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ctime>
#include <math.h>
#include <omp.h>
#include <iomanip>
#include <omp.h>
#define getValue(N,D,K) ((2*100000000 - (D*K))/(16*(D+K)))
double deviceCall_time=0.0;
//double omptime=0.0;

int SIZE_LIMIT;

//#include "gputimer.h"
#define DEBUG
#define ITERATIONS 100
using namespace std;

unsigned long long offset=4294967295;
unsigned long long bigoffset=18446744069414584320ULL;

int call_mul(double *h_data,double *h_centroid,double *h_result,int commondim,int Adim,int Bdim);
double* call_transpose(double *h_input,int centroids,int data_points);
unsigned long long* compute_minimization(double* d_Input,int data_points,int centroids);
void sortByKey(unsigned long long *keys,unsigned long long *uniqueKeys,int N,int C);
double *calculate_new_centroid(double *h_data,double *h_centroid,unsigned long long *h_min,unsigned long long *h_unique,int Adim,int Bdim,int dim);
int compare_centroids(double *h_centroid,double *new_centroid,int Bdim,int dim);
void copy_fun(unsigned long long *bh_minCentroid,unsigned long long *lh_minCentroid,int mat_size,int t);
double *findAvg(double *h_centroid,unsigned long long *h_unique,int Adim,int Bdim,int dim);
void addCentroids(double *temp_new_centroid,double *b_new_centroids,int Bdim,int commondim);
void addValuesTobhUnique(unsigned long long *temp_bh_minCentroidUniqueValues,unsigned long long *bh_minCentroidUniqueValues1,int Bdim);
void intialize(double *b_new_centroids,unsigned long long *bh_minCentroidUniqueValues1,int Bdim,int commondim);

int main()
{

	fstream file1,fout2,fout3,file4;
	file1.open("input.txt");
	file4.open("centroid.txt");

	int flagContinue = 0; //This flag checks whether we need to continue iteration
	//Take number of input points and centroids multiple of 2^x

	//Commondim represents the number of dimension of the input vector.
	//Adim represents the Number of datapoints
	//Bdin represents the number of centroid clusters
	int commondim,Adim,Bdim;

	file1 >> Adim;
	file1 >> Bdim;
	file1 >> commondim;

	int interval=(int)ceil((double)Adim/Bdim); 
	
	double *h_data=new double[Adim*commondim]; //represents the N*D matrix
	double *h_centroid=new double[commondim*Bdim];  //represents the D*K matrix
	double *h_result=new double[Adim*Bdim];   //represents the N*K matrix

	//Using this when data points are less than SIZE_LIMIT
	unsigned long long *h_minCentroid; 
	//Using this when data points are greater than SIZE_LIMIT to store partial result
	unsigned long long *bh_minCentroid;

	unsigned long long *lh_minCentroid=new unsigned long long[Adim];

	//(unsigned long long*)(malloc(sizeof(unsigned long long)*Bdim)); 
	int i,j;
	double val;
	int k=0,l=0,t=1; 

	double time=0.0;
	//GpuTimer timer;
	//cout << Adim <<" "<<Bdim<<" "<<commondim<<endl;
	
	//Fill in the input data points into the h_data matrix from the file.
		for(i=0;i<Adim;i++)
		{
			//Centroids and data set will have same values centtroids are subset on data points
			for(j=0;j<commondim;j++)
			{	
				file1 >> val;
				h_data[i*commondim+j]=val;		
			}
		}

	//Fill in the input centroids into the h_data matrix from the file.
		for(i=0;i<Bdim;i++)
		{
		
			for(j=0;j<commondim;j++)
			{	
				file4 >> val;
				h_centroid[j*Bdim+i]=val;		
			}
		}									//32x4 4x16

	int count=0;

	SIZE_LIMIT=getValue(Adim,commondim,Bdim)*16;  //SIZE_LIMIT represents the Number of datapoints which is a multiple of 16 that we will load into the GPU at a given point of time.

	cout << SIZE_LIMIT << endl;    //print SIZE_LIMIT

	clock_t begin = clock();

	if(Adim<SIZE_LIMIT)			   // When N is less than the SIZE_LIMIT //
	{
			
		for(int i=0;(i<ITERATIONS) && (flagContinue==0);i++,count++)
		{	
			call_mul(h_data,h_centroid,h_result,commondim,Adim,Bdim);
			
			//call transpose to obtain the transposed matrix in the variable d_transposedMatrix  ///This variable is used to store the transposed matrix.	
			double *d_transposedMatrix=call_transpose(h_result,Bdim,Adim); 
		
			h_minCentroid=compute_minimization(d_transposedMatrix,Adim,Bdim);

			//this is used to hold the starting index of the current values for the centroid
			unsigned long long *h_minCentroidUniqueValues=(unsigned long long*)(malloc(sizeof(unsigned long long)*Bdim)); 

			sortByKey(h_minCentroid,h_minCentroidUniqueValues,Adim,Bdim);

			

			double *new_centroid=calculate_new_centroid(h_data,h_centroid,h_minCentroid,h_minCentroidUniqueValues,Adim,Bdim,commondim);

			new_centroid=findAvg(new_centroid,h_minCentroidUniqueValues,Adim,Bdim,commondim);
			
			//To compare centroids
			flagContinue=compare_centroids(h_centroid,new_centroid,Bdim,commondim);
			
			//Assign new centroid to old centroid matrix
			h_centroid=new_centroid;	
			//printf("here iterations %d flag %d\n",i,flagContinue);
		}
	}
	else
	{
		unsigned int mat_size;
		//To hold sum of partial centroids
		double *b_new_centroids=new double[Bdim*commondim];
		//To hold sum of partial unique values
		unsigned long long *bh_minCentroidUniqueValues1=new unsigned long long[Bdim];
		//To find out number of iterations
		int limit=(int)ceil((float)Adim/SIZE_LIMIT);
		//cout << "limit= "<<limit<<endl;
		//unsigned long long *bh_minCentroidUniqueValues=(unsigned long long*)(malloc(sizeof(unsigned long long)*Bdim)); 
		for(int iter=0;(iter<ITERATIONS) && (flagContinue==0);iter++,count++)
		{	

			intialize(b_new_centroids,bh_minCentroidUniqueValues1,Bdim,commondim);
			
			for(t=0;t<limit;t++)
			{
				mat_size=min(SIZE_LIMIT,(Adim-(t*SIZE_LIMIT)));
		//		cout<<"mat size="<<mat_size<<endl;
				double *bh_data=new double[mat_size*commondim];	
				double *bh_result=new double[mat_size*Bdim];
				//Get partial data points into bh_data array
				int add_offset=(t*SIZE_LIMIT*commondim);

				for(i=0;i<mat_size;i++)
				{	
					//Centroids and data set will have same values centtroids are subset on data points
					for(j=0;j<commondim;j++)
					{	
						
						bh_data[i*commondim+j]=h_data[i*commondim+j+add_offset];
					}
				}			
				//Compute distance between centroid and each data point, store result in h_result
				call_mul(bh_data,h_centroid,bh_result,commondim,mat_size,Bdim);
				
				//Call transpose to obtain the transposed matrix in the variable d_transposedMatrix  ///This variable is used to store the transposed matrix.	
				double *d_transposedMatrix=call_transpose(bh_result,Bdim,mat_size); 

				bh_minCentroid=compute_minimization(d_transposedMatrix,mat_size,Bdim);

				//this is used to hold the starting index of the current values for the centroid
				unsigned long long *temp_bh_minCentroidUniqueValues=new unsigned long long[Bdim];//(unsigned long long*)(malloc(sizeof(unsigned long long)*Bdim)); 
				

				sortByKey(bh_minCentroid,temp_bh_minCentroidUniqueValues,mat_size,Bdim);
				//To sum partial unique values from temp_bh_minCentroidUniqueValues to bh_minCentroidUniqueValues1
				addValuesTobhUnique(temp_bh_minCentroidUniqueValues,bh_minCentroidUniqueValues1,Bdim);


				double *temp_new_centroid=calculate_new_centroid(bh_data,h_centroid,bh_minCentroid,temp_bh_minCentroidUniqueValues,mat_size,Bdim,commondim);
				//To sum partial centroids into b_new_centroids
				addCentroids(temp_new_centroid,b_new_centroids,Bdim,commondim);
				
				//cout << " t " << t << endl;

				free (bh_data);
				free (bh_result);
				free (temp_bh_minCentroidUniqueValues);
				free (temp_new_centroid);

			}
			double *new_centroid=findAvg(b_new_centroids,bh_minCentroidUniqueValues1,Adim,Bdim,commondim);
				//To compare centroids
			flagContinue=compare_centroids(h_centroid,new_centroid,Bdim,commondim);
				//Assign new centroid to old centroid matrix
			h_centroid=new_centroid;

			//cout << "iter " << iter << endl;
		}

		free(b_new_centroids);
		free(bh_minCentroidUniqueValues1);
		//free(temp_bh_minCentroidUniqueValues);			
	}	
	
	clock_t end = clock();
	time += ((double)(end - begin)/CLOCKS_PER_SEC);

	printf("Number of Iteration required was %d \n",count);
	printf("Time required for function was %f\n",deviceCall_time);
	

	#ifdef DEBUG

		//cout<<"HI"<<endl;
		fout2.open("assigned_centroids.txt",ios::out);
		fout3.open("new_centroids.txt",ios::out);
		
		fout3<<"Time of Execution :"<< time << endl;
		fout3<<"Time of Device Execution :"<< deviceCall_time << endl;

		for(i=0;i<Adim;i++)
		{
			fout2 <<  ((h_minCentroid[i] & bigoffset) >> 32) << " " << (h_minCentroid[i] & offset) << endl;
		}
		
		fout2.close();

		//Print new centroids into file in format K x D
		for(i=0;i<(Bdim);i++) 
	    {
	    	
	    	for (int j = 0; j < commondim; j++)
	    	{
	    		fout3 << setprecision(10) << h_centroid[j*Bdim+i] << " ";

	    	}
	    	fout3<<endl;
	    	      
	    }   

		fout3.close();

	#endif
	
	file1.close();

//	free(h_minCentroid);
	free(h_data);
	free(h_centroid);
	free(h_result);
	

	return 0;
}

void intialize(double *b_new_centroids,unsigned long long *bh_minCentroidUniqueValues1,int Bdim,int commondim)
{
		int i;
		#pragma omp parallel
		{
			#pragma omp for
			for(i=0;i<Bdim;i++)
			{
				bh_minCentroidUniqueValues1[i]=0;
			}
			#pragma omp for
			for(i=0;i<(Bdim*commondim);i++)
			{
				b_new_centroids[i]=0;
			}
		}
		#pragma omp end parallel
}

void addValuesTobhUnique(unsigned long long *temp_bh_minCentroidUniqueValues,unsigned long long *bh_minCentroidUniqueValues1,int Bdim)
{
		int i;
		#pragma omp parallel for
		for(i=0;i<Bdim;i++)
		{
			bh_minCentroidUniqueValues1[i]+=temp_bh_minCentroidUniqueValues[i];
			//cout<<temp_bh_minCentroidUniqueValues[i]<<" ";	    	
		}
		#pragma omp end parallel
	/*	cout<<"dd"<<endl;
		for(int i=0;i<Bdim;i++)
		{
			cout<<bh_minCentroidUniqueValues1[i]<<" ";	    	
		}	
		cout<<"dd"<<endl;
	*/
}
void addCentroids(double *temp_new_centroid,double *b_new_centroids,int Bdim,int commondim)
{
	int i;
	#pragma omp parallel for
	for(i=0;i<(Bdim*commondim);i++)
	{
		b_new_centroids[i]+=temp_new_centroid[i];
	}
	#pragma omp end parallel
}


