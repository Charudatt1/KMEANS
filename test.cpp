#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>

using namespace std;

int main()
{
	ifstream fin("test.txt");

	double value;
	
	fin >> value;
	
	cout << setprecision(10) << value;

	fin.close();	

	return 0;
}
