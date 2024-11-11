WB = ${WB_DIR}

template.o: template.cu
	nvcc -std=c++11 -rdc=true -I $(WB) -c template.cu -o template.o
	

solution: template.o
	nvcc -std=c++11 -o template template.o $(WB)/lib/libwb.so

clean:
	-rm -f template.o
	-rm -f solution

test :
	nvcc -std=c++11 -I ~/libwb -rdc=true -c template.cu
	nvcc -std=c++11 -o template template.o ~/libwb/lib/libwb.so
