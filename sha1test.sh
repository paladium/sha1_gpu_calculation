nvcc.exe -c -arch=sm_35 -rdc=true sha1test.cu -o sha1test.o
nvcc.exe -arch=sm_35 -rdc=true -lcuda sha1test.o sha1_kernel.o -o sha1test
./sha1test.exe