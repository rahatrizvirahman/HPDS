import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerExample {

	private static AtomicInteger atomicInteger = new AtomicInteger(0);

	static class MyRunnable implements Runnable {

		private int myCounter;
		private int myPrevCounter;
		private int myCounterPlusFive;
		private boolean isNine;

		public void run() {
			myCounter = atomicInteger.incrementAndGet();
			System.out.println("Thread " + Thread.currentThread().getId() + "  / Counter : " + myCounter);
			
			myPrevCounter = atomicInteger.getAndIncrement();
			System.out.println("Thread " + Thread.currentThread().getId() + " / Previous : " + myPrevCounter);
			
			myCounterPlusFive = atomicInteger.addAndGet(5);		
			System.out.println("Thread " + Thread.currentThread().getId() + " / plus five : " + myCounterPlusFive);
			
			isNine = atomicInteger.compareAndSet(9, 3);
			
			if (isNine) {
				System.out.println("Thread " + Thread.currentThread().getId() + " / Value was equal to 9, so it was updated to " + atomicInteger.intValue());
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Thread t1 = new Thread(new MyRunnable());
		Thread t2 = new Thread(new MyRunnable());
		t1.start();
		t2.start();
		t1.join();
		t2.join();
	}
}