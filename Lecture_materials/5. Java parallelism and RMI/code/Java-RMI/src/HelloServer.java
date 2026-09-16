import java.net.InetAddress;
import java.rmi.Naming;

public class HelloServer {
	public static void main (String[] args) throws Exception {
		String hostname = InetAddress.getLocalHost().getHostName();
		System.out.println("Binding using hostname " + hostname);

		System.setProperty("java.rmi.server.hostname", hostname);
		
		Naming.rebind("rmi://" + hostname + "/HelloWorldServer", new HelloWorldServer());
	}
}
