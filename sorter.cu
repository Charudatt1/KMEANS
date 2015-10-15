#include <thrust/distance.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <omp.h>

//#include <cstdio>
//__host__ __device__
using namespace std;
//extern double omptime;






struct Cmp {     //This comparator is used to compare the least significant 32 bits to obtain the value

 unsigned long long offset;

  __host__ __device__
  bool operator()(const unsigned long long &o1, const unsigned long long &o2) {
      return (o1 & offset) < (o2 & offset);
  }

  Cmp():offset(4294967295)  //initialize the offset value by the given Number
  {
  	
  }

};


	//This constructor is used to compare two values
 struct is_number
  {

  	unsigned long long value;
  	unsigned long long offset;

    __host__ __device__
    bool operator()(const unsigned long long &x)
    {
      return ((x & offset) == value);
    }

    is_number(unsigned long long val)
    {
    	value=val;
    	offset=4294967295;
    }

  };


//This Method takes Keys as input Keys array consist of unsigned long long integers whose least 32 bits represents the centroid indexes and the most significant 32 bits represent the value of N(i.e Payload)
//This function calls a comparator cmp() for the comparision operation.
//N is the number of data points

//This function is passed an array called uniqueKeys which contains the starting offset in the keys array for each unique key.

void sortByKey(unsigned long long *keys,unsigned long long *uniqueKeys,int N,int C) 
{
	//unsigned long long offset=4294967295;
	thrust::device_vector<unsigned long long> d_K(keys,keys+N);  //allocate memory for storing keys
	thrust::device_vector<unsigned long long> C_K(C);   //containing the unique values
	thrust::device_vector<unsigned long long> P_Sum(C);   //containing the prefix Sum for the unique values

  //omptime=omp_get_wtime();
	thrust::sort(d_K.begin(),d_K.end(),Cmp());          //sort the arrays

	for(int i=0;i<C;i++)
	{
		C_K[i]=thrust::count_if(d_K.begin(),d_K.end(),is_number(i));  //count the number of occurence of a value i
	}
		    //

	thrust::exclusive_scan( C_K.begin(), C_K.end(), P_Sum.begin());  //scan to get the offset values

	thrust::copy(d_K.begin(),d_K.end(),keys); //copy back the values to V array
	thrust::copy(P_Sum.begin(),P_Sum.end(),uniqueKeys); //copy back the values to V array

  //omptime=omp_get_wtime()-omptime;
  //printf("Sorter time %lf\n",omptime);
	d_K.clear();
	C_K.clear();
	P_Sum.clear();

	C_K.shrink_to_fit();
	P_Sum.shrink_to_fit();
	d_K.shrink_to_fit();

 /* for(int i=0;i<C;i++)
  {
    if(i+1 < C)
      cout<<"K"<<i<<" :"<<uniqueKeys[i+1]-uniqueKeys[i]<<endl;
    else
      cout<<"K"<<i<<" :"<<N-uniqueKeys[i]<<endl;
  }*/
}