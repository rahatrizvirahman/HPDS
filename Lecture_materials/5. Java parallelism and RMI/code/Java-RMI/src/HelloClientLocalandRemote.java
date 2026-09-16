import java.rmi.Naming;

public class HelloClientLocalandRemote {
	public static void main (String[] args) throws Exception {
		HelloWorldInterface helloLocal  = (HelloWorldInterface) Naming.lookup("rmi://huff40/HelloWorldServer");
		HelloWorldInterface helloRemote = (HelloWorldInterface) Naming.lookup("rmi://huff41/HelloWorldServer");
		
		String localReturn, remoteReturn;
		long startTime, endTime, estimatedTime;

		// Warm up fist-time use
		localReturn = helloLocal.sayHello("Alberto");
		remoteReturn = helloRemote.sayHello("Alberto");
		
		startTime = System.nanoTime();    
		localReturn = helloLocal.sayHello("Alberto");
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println(localReturn + " in " + estimatedTime + " ns");
		
		startTime = System.nanoTime();    
		remoteReturn = helloRemote.sayHello("Alberto");
		endTime = System.nanoTime();
		estimatedTime = endTime - startTime;
		System.out.println(remoteReturn + " in " + estimatedTime + " ns");
	}
}