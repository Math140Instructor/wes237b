CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

debug: CFLAGS += -DOCL_DEVICE_TYPE=CL_DEVICE_TYPE_CPU
debug: MAKECMD += debug
debug: solution
	
solution: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a:
	cd ../../helper_lib; make

run: solution
	@for i in `seq 0 15`; do \
		echo "Running test $$i"; \
		./solution Dataset/without_strides/$$i/input0.raw Dataset/without_strides/$$i/kernel0.raw Dataset/without_strides/$$i/output.raw output.raw; \
	done

with_stride: solution
	@for i in `seq 0 37`; do \
		echo "Running test $$i"; \
		./solution Dataset/with_strides/$$i/input0.raw Dataset/with_strides/$$i/kernel0.raw Dataset/with_strides/$$i/output.raw output.raw; \
	done

time: solution
	python3 ../../utils/profile.py --args ./solution Dataset/without_strides/15/input0.raw Dataset/without_strides/15/kernel0.raw Dataset/without_strides/15/output.raw output.raw

time_with_stride: solution
	python3 ../../utils/profile.py --args ./solution Dataset/with_strides/37/input0.raw Dataset/with_strides/37/kernel0.raw Dataset/with_strides/37/output.raw output.raw

clean: 
	rm -f solution
	cd ../../helper_lib; make clean