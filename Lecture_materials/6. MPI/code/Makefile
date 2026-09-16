all: hello hello2 hello3 hello4 nonblocking bcast scatter gather gather2 reduce
hello: hello.c
	mpicc hello.c -o hello
hello2: hello2.c
	mpicc hello2.c -o hello2
hello3: hello3.c
	mpicc hello3.c -o hello3
hello4: hello4.c
	mpicc hello4.c -o hello4
nonblocking: nonblocking.c
	mpicc nonblocking.c -o nonblocking	
bcast: bcast.c
	mpicc bcast.c -o bcast
scatter: scatter.c
	mpicc scatter.c -o scatter		
gather: gather.c
	mpicc gather.c -o gather			
gather2: gather2.c
	mpicc gather2.c -o gather2
reduce: reduce.c
	mpicc reduce.c -o reduce		
clean:
	rm hello hello2 hello3 hello4 nonblocking bcast scatter gather gather2 reduce
