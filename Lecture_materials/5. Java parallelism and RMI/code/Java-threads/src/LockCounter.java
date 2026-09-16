import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class LockCounter {

	private int c = 0;
	
	private Lock lock = new ReentrantLock();

	public void increment() {
		lock.lock();
		c++;
		lock.unlock();
	}

	public int value() {
		return c;
	}

	private static class MyThread extends Thread {

		private LockCounter counter;

		public MyThread (LockCounter counter) {
			this.counter = counter;
		}

		public void run() {
			for(int i = 0; i < 1000; i++)
				counter.increment();
		}
	}

	public static void main(String args[]) throws Exception { 

		LockCounter counter = new LockCounter();

		MyThread[] threads = new MyThread[10];

		for(int i = 0; i < 10; i++)
			threads[i] = new MyThread(counter);

		for(int i = 0; i < 10; i++)
			threads[i].start();

		for(int i = 0; i < 10; i++)
			threads[i].join();

		System.out.println(counter.value());	
	}
}

