import java.rmi.Naming;

public class HelloClient {
	public static void main (String[] args) throws Exception {
		HelloWorldInterface hello = (HelloWorldInterface) Naming.lookup("rmi://huff40/HelloWorldServer");
		System.out.println(hello.sayHello("Alberto"));
	}
}