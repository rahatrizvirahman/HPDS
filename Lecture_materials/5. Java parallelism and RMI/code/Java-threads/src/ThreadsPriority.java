public class ThreadsPriority extends Thread {

	private int tid;

	public ThreadsPriority(int tid) {
		this.tid = tid;
	}

	public void run(){  
		System.out.println("Thread " + tid + " priority " + Thread.currentThread().getPriority());  
	}

	public static void main(String args[]) throws Exception {  
		ThreadsPriority t1 = new ThreadsPriority(1);  
		ThreadsPriority t2 = new ThreadsPriority(10);  
		t1.setPriority(Thread.MIN_PRIORITY);  
		t2.setPriority(Thread.MAX_PRIORITY);  
		t1.start();  
		t2.start();
		t1.join();
		t2.join();
	}
}