nvcc.exe -c -arch=sm_35 -rdc=true test.cu -o test.o
nvcc.exe -arch=sm_35 -rdc=true -lcuda test.o sha1_kernel.o -o test
./test.exe