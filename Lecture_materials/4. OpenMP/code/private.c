#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main()
{
    int size = 100;
    int *a, *b, *c;

    a = (int*)malloc(size * sizeof(int));
    b = (int*)malloc(size * sizeof(int));
    c = (int*)malloc(size * sizeof(int));
    
    for(int i = 0; i < size; i++)
    {
        a[i] = b[i] = i;
        c[i] = 0;
    }
    
    int result;
    #pragma omp parallel for private(result)
    for(int i = 0; i < size; i++)
    {
        result = a[i] + b[i];
        c[i] = result;
    }
    
    for(int i = 0; i < size; i++)
        printf("%d\n", c[i]);
    
    free(a);
    free(b);
    free(c);        
}
