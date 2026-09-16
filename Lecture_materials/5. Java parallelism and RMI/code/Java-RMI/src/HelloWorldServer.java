import java.net.InetAddress;
import java.net.UnknownHostException;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class HelloWorldServer extends UnicastRemoteObject implements HelloWorldInterface {

	public HelloWorldServer() throws RemoteException {
		super();
	}

	public String sayHello(String name) throws RemoteException {
		String hostname;

		try {
			hostname = InetAddress.getLocalHost().getHostName();
		} catch (UnknownHostException e) {
			hostname = "unknown";
		}

		return "Hello " + name + " I'm responding from " + hostname;
	}
}
