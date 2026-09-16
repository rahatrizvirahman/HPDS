import java.util.concurrent.Semaphore;

public class SemaphoreCounter {

	private int c = 0;
	
	private Semaphore semaphore = new Semaphore(1);

	public void increment() {
		try {
			semaphore.acquire();
			c++;
			semaphore.release();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public int value() {
		return c;
	}

	private static class MyThread extends Thread {

		private SemaphoreCounter counter;

		public MyThread (SemaphoreCounter counter) {
			this.counter = counter;
		}

		public void run() {
			for(int i = 0; i < 1000; i++)
				counter.increment();
		}
	}

	public static void main(String args[]) throws Exception { 

		SemaphoreCounter counter = new SemaphoreCounter();

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

