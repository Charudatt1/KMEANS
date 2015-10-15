#Compiler variables
CC=g++
NVCC=nvcc

#Compiler flags
CFLAGS=-c
NFLAGS=-c -arch=sm_21
OFLAGS=-fopenmp

#Linker flags
LKER=-L/usr/local/cuda/lib64 -lcudart
OMPLKER=-lgomp


all: Kmeans clean

Kmeans: main.o transpose.o minimization_kernel.o matrix_mul_kernel.o sorter.o new_centroid.o compare.o find_avg.o
	$(CC) $(LKER) $(OMPLKER) main.o transpose.o minimization_kernel.o matrix_mul_kernel.o sorter.o new_centroid.o compare.o find_avg.o -o Kmeans

transpose.o: transpose.cu
		$(NVCC) $(NFLAGS) transpose.cu

minimization_kernel.o: minimization_kernel.cu
		$(NVCC) $(NFLAGS) minimization_kernel.cu

matrix_mul_kernel.o: matrix_mul_kernel.cu
		$(NVCC) $(NFLAGS) matrix_mul_kernel.cu

sorter.o: sorter.cu
	$(NVCC) $(NFLAGS) sorter.cu

main.o: main.cpp
	$(CC) $(CFLAGS) $(OFLAGS) main.cpp

new_centroid.o:new_centroid.cu
	$(NVCC) $(NFLAGS) new_centroid.cu

compare.o:compare.cu
	$(NVCC) $(NFLAGS) compare.cu
find_avg.o:find_avg.cu
	$(NVCC) $(NFLAGS) find_avg.cu

clean:
	rm -rf *.o
