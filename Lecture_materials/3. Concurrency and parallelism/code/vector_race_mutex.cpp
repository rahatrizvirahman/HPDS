#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

#define NTHREADS 100

vector<int> v;
mutex m;

void addElements() 
{
    m.lock();
    v.push_back(1);
    m.unlock();
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