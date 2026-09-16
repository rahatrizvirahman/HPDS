#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main()
{
    int size = 100;
    int *array;

    array = (int*)malloc(size * sizeof(int));
    
    for(int i = 0; i < size; i++)
    {
        array[i] = i+1;
    }
    
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    
    printf("Parallel reduction result %d expected value %d\n", sum, ((1+size)*size)/2);
    
    free(array);
}
