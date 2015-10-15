#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>
//#define FLOAT
#define DOUBLE


using namespace std;

int main()
{
	ifstream file1;
	ofstream file2;
	file1.open("input.txt");
	file2.open("centroid.txt",ios::out);

	unsigned int n,k,interval,i,d,j;

	/*
#ifdef FLOAT
	float dimension;
#else	
	double dimension;
#endif
	
	*/

	double dimension;

	file1 >> n;
	file1 >> k;
	file1 >> d;
	         
	interval=(int)(ceil((float)n/k));

	//cout << interval <<" "<<n<<" "<<k<<" "<<d;

	for(i=0;i<n;i++)
	{
		
		for(j=0;j<d;j++)
		{	
			file1 >> dimension;
			if(i%(interval)==0)
			{		
			
				file2 << std::setprecision(10) << dimension << " ";
				cout << dimension << endl;
			}
			
		}
		if(i%(interval)==0)
		{
			file2<<endl;
		
		}
		
	}		

	return 0;
}