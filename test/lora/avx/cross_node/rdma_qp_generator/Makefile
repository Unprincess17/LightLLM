# test/lora/avx/cross_node/rdma_qp_generator/Makefile
CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -fPIC
LDFLAGS  = -shared
LDLIBS   = -libverbs -lpthread

CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart

TARGET_QP  = librdmaqpgen.so
TARGET_GDR = libgdrqpgen.so

SRC_QP  = rdmaqpgen.c
OBJ_QP  = rdmaqpgen.o
SRC_GDR = gdrqpgen.c
OBJ_GDR = gdrqpgen.o

.PHONY: all clean

all: $(TARGET_QP) $(TARGET_GDR)

$(TARGET_QP): $(OBJ_QP)
	$(CC) $(LDFLAGS) -o $@ $< $(LDLIBS)

$(OBJ_QP): $(SRC_QP) rdmaqpgen.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET_GDR): $(OBJ_GDR)
	$(CC) $(LDFLAGS) -o $@ $< $(LDLIBS) $(CUDA_LIB)

$(OBJ_GDR): $(SRC_GDR) gdrqpgen.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c $< -o $@

clean:
	rm -f $(OBJ_QP) $(OBJ_GDR) $(TARGET_QP) $(TARGET_GDR)

test_header: test_gdrqpgen_header.c gdrqpgen.h
	$(CC) $(CFLAGS) -c test_gdrqpgen_header.c -o /tmp/test_gdrqpgen_header.o

test_gdr_mr: test_gdr_mr.c libgdrqpgen.so
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $< -L. -lgdrqpgen $(CUDA_LIB) -libverbs -Wl,-rpath,.

test_gdr_loopback: test_gdr_loopback.c libgdrqpgen.so
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $< -L. -lgdrqpgen $(CUDA_LIB) -libverbs -Wl,-rpath,.

test_gdr_write: test_gdr_write.c libgdrqpgen.so
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $< -L. -lgdrqpgen $(CUDA_LIB) -libverbs -Wl,-rpath,.

test_gdr_read: test_gdr_read.c libgdrqpgen.so
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $< -L. -lgdrqpgen $(CUDA_LIB) -libverbs -Wl,-rpath,.
