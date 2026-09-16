public class HelloRunnable implements Runnable {

    public void run() {
        System.out.println("Hello from a thread!");
    }

    public static void main(String args[]) throws Exception {
    	Thread t = new Thread(new HelloRunnable());
    	t.start();
    	t.join();
    }
}
