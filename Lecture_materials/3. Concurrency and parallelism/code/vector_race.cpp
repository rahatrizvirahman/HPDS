#include <iostream>
#include <vector>
#include <thread>

using namespace std;

#define NTHREADS 100

vector<int> v;

void addElements() 
{
    v.push_back(1);
}
  
int main ()
{
    thread threads[NTHREADS];
    
    for(int i = 0; i < NTHREADS; i++)
        threads[i] = thread(addElements);
        
    for(int i = 0; i < NTHREADS; i++)
        threads[i].join();
    
    cout << v.size() << endl;
}
