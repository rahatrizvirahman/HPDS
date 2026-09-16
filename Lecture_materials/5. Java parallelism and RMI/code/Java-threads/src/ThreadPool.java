import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ThreadPool {
	
	private static int numberThreads;
	
	public static class MyThread extends Thread {
		public void run() {
			// do something to keep busy the CPU
			double[] array = new double[1000];
			for(int i = 0; i < 1000; i++)
				array[i] = i * i;
		}
	}

	public static class ExecuteAllThreadsAtOnce {

		public void execute() throws Exception {
			
			MyThread[] threads = new MyThread[numberThreads];

			for(int i = 0; i < numberThreads; i++)
				threads[i] = new MyThread();

			for(int i = 0; i < numberThreads; i++)
				threads[i].start();

			for(int i = 0; i < numberThreads; i++)
				threads[i].join();
		}
	}
	
	public static class ThreadPoolExample {

		public void execute() {
			ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

			for(int i = 0; i < numberThreads; i++)
				executorService.execute(new MyThread());

			executorService.shutdown();
			try {
				executorService.awaitTermination(5, TimeUnit.SECONDS);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static void main(String args[]) throws Exception { 
		
		numberThreads = 100;
		ThreadPoolExample threadpool = new ThreadPoolExample();
		ExecuteAllThreadsAtOnce alltogether = new ExecuteAllThreadsAtOnce();
		long startTime, endTime, estimatedTime;

		startTime = System.nanoTime();    
		threadpool.execute();
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("Thread pool \t\t" + estimatedTime + " ns");

		startTime = System.nanoTime();    
		alltogether.execute();
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println("All threads at once \t" + estimatedTime + " ns");
	}
}