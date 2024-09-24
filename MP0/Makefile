WB = ${WB_DIR}

template.o: template.cu
	nvcc -std=c++11 -rdc=true -I $(WB) -c template.cu -o template.o

solution: template.o
	nvcc -std=c++11 -o solution template.o $(WB)/lib/libwb.so

clean:
	-rm -f template.o
	-rm -f solution