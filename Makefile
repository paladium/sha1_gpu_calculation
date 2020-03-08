NVCC := nvcc --ptxas-options=-v
CFLAGS := -c# -D_DEBUG
LDFLAGS := -lcuda -lcutil
SHA1OBJS := sha1test.o sha1_kernel.o

all: sha1test

# SHA-1 benchmark test
sha1test: $(SHA1OBJS)
	$(NVCC) $(LDFLAGS) $(SHA1OBJS) -o sha1test
sha1test.o: sha1test.cu common.h
	$(NVCC) $(CFLAGS) sha1test.cu -o sha1test.o
sha1_kernel.o: sha1_kernel.cu common.h
	$(NVCC) $(CFLAGS) sha1_kernel.cu -o sha1_kernel.o