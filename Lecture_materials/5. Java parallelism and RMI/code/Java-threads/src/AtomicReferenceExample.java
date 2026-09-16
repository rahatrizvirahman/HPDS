import java.util.concurrent.atomic.AtomicReference;

public class AtomicReferenceExample {
	
	public static class SynchronizedCounter {
		private int c = 0;
		private AtomicReference<Integer> atomicCounter = new AtomicReference<Integer>(c);
		
		public void increment() {
			atomicCounter.accumulateAndGet(1, Integer::sum);
		}

		public int value() {
			return atomicCounter.get();
		}
	}

	private static class MyThread extends Thread {

		private SynchronizedCounter counter;

		public MyThread (SynchronizedCounter counter) {
			this.counter = counter;
		}

		public void run() {
			for(int i = 0; i < 1000; i++)
				counter.increment();
		}
	}

	public static void main(String args[]) throws Exception { 

		SynchronizedCounter counter = new SynchronizedCounter();
		
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