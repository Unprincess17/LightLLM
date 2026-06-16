# test/lora/avx/cross_node/rdma_qp_generator/Makefile
CC       = gcc
CFLAGS   = -std=c11 -Wall -Wextra -O2 -fPIC
LDFLAGS  = -shared
LDLIBS   = -libverbs -lpthread

TARGET   = librdmaqpgen.so
SRC      = rdmaqpgen.c
OBJ      = rdmaqpgen.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $< $(LDLIBS)

$(OBJ): $(SRC) rdmaqpgen.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
