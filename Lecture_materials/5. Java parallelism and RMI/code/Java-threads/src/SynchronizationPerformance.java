public class SynchronizationPerformance {
	
	public static void main(String args[]) throws Exception { 

		long startTime, endTime, estimatedTime;
		
		startTime = System.nanoTime();    
		AtomicReferenceExample.main(null);
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Atomic reference \t" + estimatedTime + " ns");
		
		startTime = System.nanoTime();    
		LockCounter.main(null);
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Lock \t\t\t" + estimatedTime + " ns");
		
		startTime = System.nanoTime();    
		SemaphoreCounter.main(null);
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Semaphore \t\t" + estimatedTime + " ns");
		
		startTime = System.nanoTime();    
		SynchronizedBlock.main(null);
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Synchronized block \t" + estimatedTime + " ns");
		
		startTime = System.nanoTime();    
		SynchronizedMethod.main(null);
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Synchronized method \t" + estimatedTime + " ns");
	}
}