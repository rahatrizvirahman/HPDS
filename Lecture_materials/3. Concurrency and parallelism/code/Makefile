all: create_thread create_thread_arguments create_thread_arguments_wrong race race2 vector_race vector_race_mutex producer_consumer sequential_parallel sequential_parallel_flood sequential_parallel_persistent_threads atomic atomic_performance restroom_with_capacity
create_thread: create_thread.c
	gcc -pthread -g create_thread.c -o create_thread
create_thread_arguments: create_thread_arguments.c
	gcc -pthread -g create_thread_arguments.c -o create_thread_arguments
create_thread_arguments_wrong: create_thread_arguments_wrong.c
	gcc -pthread -g create_thread_arguments_wrong.c -o create_thread_arguments_wrong	
race: race.c
	gcc -pthread -g race.c -o race
race2: race2.c
	gcc -pthread -g race2.c -o race2
vector_race: vector_race.cpp
	g++ -std=c++11 -pthread -g vector_race.cpp -o vector_race
vector_race_mutex: vector_race_mutex.cpp
	g++ -std=c++11 -pthread -g vector_race_mutex.cpp -o vector_race_mutex
producer_consumer: producer_consumer.c
	gcc -pthread -g producer_consumer.c -o producer_consumer
sequential_parallel: sequential_parallel.c
	gcc -pthread -g sequential_parallel.c -o sequential_parallel
sequential_parallel_flood: sequential_parallel_flood.c
	gcc -pthread -g sequential_parallel_flood.c -o sequential_parallel_flood	
sequential_parallel_persistent_threads: sequential_parallel_persistent_threads.c
	gcc -pthread -g sequential_parallel_persistent_threads.c -o sequential_parallel_persistent_threads	
atomic: atomic.c
	gcc -std=gnu11 -pthread -g atomic.c -o atomic	
atomic_performance: atomic_performance.c
	gcc -std=gnu11 -pthread -g atomic_performance.c -o atomic_performance		
restroom_with_capacity: restroom_with_capacity.c
	gcc -pthread -g restroom_with_capacity.c -lm -o restroom_with_capacity
clean:
	rm create_thread create_thread_arguments create_thread_arguments_wrong race race2 vector_race vector_race_mutex producer_consumer sequential_parallel sequential_parallel_flood sequential_parallel_persistent_threads atomic atomic_performance restroom_with_capacity