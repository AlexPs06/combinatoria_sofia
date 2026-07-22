TARGET = sofia_cuda
SOURCE = sofia_cuda.cu

NVCC = nvcc

NVCCFLAGS = -O3 \
            -std=c++17 \
            -arch=sm_61 \
            -lineinfo \
            -Xptxas=-v \
            -Xcompiler=-fopenmp \
            -Xcompiler=-Wall \
            -Xcompiler=-Wextra
#           para 3060ti  -arch=sm_86 \


all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer ./$(TARGET)

debug:
	$(NVCC) -O0 -g -G \
	        -std=c++17 \
	        -arch=sm_86 \
	        -lineinfo \
	        -Xcompiler=-fopenmp \
	        $(SOURCE) \
	        -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run sanitize debug clean

