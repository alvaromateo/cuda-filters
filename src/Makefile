all: cudafilters.exe

cudafilters.exe: mainFilters.o test.o tools.o
	g++ -o cudafilters.exe mainFilters.o test.o tools.o -std=c++11

mainFilters.o: tools.cpp test.cpp mainFilters.cpp
	g++ -c tools.cpp -std=c++11
	g++ -c test.cpp -std=c++11
	g++ -c mainFilters.cpp -std=c++11
	
clean:
	rm -rf *.o *.exe