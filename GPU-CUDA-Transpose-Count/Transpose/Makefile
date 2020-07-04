CXX := nvcc
OPT_FLAGS := -O3
DEBUG_FLAGS := -G
GENCODE := -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75

.phony: clean all debug release

all: debug release

release: count_opt transpose_opt

debug: count_debug transpose_debug

clean:
	rm -f transpose_opt transpose_release count_opt count_release

count_opt: count.cu util.h
	$(CXX) $(OPT_FLAGS) -o $@ $< $(GENCODE)

transpose_opt: transpose.cu util.h
	$(CXX) $(OPT_FLAGS) -o $@ $< $(GENCODE)

count_debug: count.cu util.h
	$(CXX) $(DEBUG_FLAGS) -o $@ $< $(GENCODE)

transpose_debug: transpose.cu util.h
	$(CXX) $(DEBUG_FLAGS) -o $@ $< $(GENCODE)

handin.tar: transpose.cu count.cu
	tar -cvf handin.tar transpose.cu count.cu

.phony: clean

