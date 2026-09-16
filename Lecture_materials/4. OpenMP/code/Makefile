all: hello parallelfor parallelfor2 parallelfor3 private reduction sections critical performance matrixmultiplication
hello: hello.c
	gcc -fopenmp hello.c -o hello
parallelfor: parallelfor.c
	gcc -std=gnu11 -fopenmp parallelfor.c -o parallelfor
parallelfor2: parallelfor2.c
	gcc -std=gnu11 -fopenmp parallelfor2.c -o parallelfor2
parallelfor3: parallelfor3.c
	gcc -std=gnu11 -fopenmp parallelfor3.c -o parallelfor3
private: private.c
	gcc -std=gnu11 -fopenmp private.c -o private		
reduction: reduction.c
	gcc -std=gnu11 -fopenmp reduction.c -o reduction	
sections: sections.c
	gcc -std=gnu11 -fopenmp sections.c -o sections	
critical: critical.c
	gcc -std=gnu11 -fopenmp critical.c -o critical
performance: performance.c
	gcc -std=gnu11 -fopenmp performance.c -o performance	
matrixmultiplication: matrixmultiplication.c
	gcc -std=gnu11 -fopenmp matrixmultiplication.c -o matrixmultiplication			
clean:
	rm hello parallelfor parallelfor2 parallelfor3 private reduction sections critical performance matrixmultiplication