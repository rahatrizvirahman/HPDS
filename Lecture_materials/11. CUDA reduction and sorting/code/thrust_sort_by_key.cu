#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>

int main(int argc, char* argv[])
{
    const int N = 6;
    int keys[N] = { 1, 4, 2, 8, 5, 7};
    char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

    thrust::sort_by_key(keys, keys + N, values);

    // keys is now { 1, 2, 4, 5, 7, 8}
    // values is now {'a', 'c', 'b', 'e', 'f', 'd'}

    for(int i = 0; i < N; i++)
        printf("%d ", keys[i]);
    printf("\n");

    for(int i = 0; i < N; i++)
        printf("%c ", values[i]);
    printf("\n");

    return 0;
}