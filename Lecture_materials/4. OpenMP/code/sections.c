#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main()
{
    #pragma omp parallel sections
    {
        #pragma omp section
        { 
            // Do something by one thread
            printf ("id = %d, \n", omp_get_thread_num());
        }

        #pragma omp section
        { 
            // Do something by another thread
            printf ("id = %d, \n", omp_get_thread_num());
        }

        #pragma omp section
        { 
            // Do something by yet another thread
            printf ("id = %d, \n", omp_get_thread_num());
        }
    }
}
