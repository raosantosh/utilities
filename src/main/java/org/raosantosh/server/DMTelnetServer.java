package com.yahoo.cdm.server;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

import com.yahoo.cdm.main.CDMMain;

public class DMTelnetServer {

	protected void start(String[] args) {
		ServerSocket s;
		int port = 8001;
		CDMMain dmMain = new CDMMain();
		System.out.println("Webserver starting up on port 8001");
		System.out.println("(press ctrl-c to exit)");

		System.out.println("Port is: " + port);

		try {
			// create the main server socket

			s = new ServerSocket(port);
		} catch (Exception e) {
			System.out.println("Error: " + e);
			return;
		}

		System.out.println("Waiting for connection");
		for (;;) {
			try {
				// wait for a connection
				Socket remote = null;
				if (true) {
					Socket remote1 = s.accept();
					System.out.println("New Connection Reqeust");
					TravelTelnetThread trelTelnetThread = new TravelTelnetThread(remote1, dmMain);
					(new Thread(trelTelnetThread)).start();
					continue;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	static class TravelTelnetThread implements Runnable {

		private Socket socket;
		private CDMMain dmMain;

		public TravelTelnetThread(Socket socket, CDMMain dmMain) {
			this.socket = socket;
			this.dmMain = dmMain;
		}

		@Override
		public void run() {
			try {
				// remote is now the connected socket
				System.out.println("Connection, sending data.");
				BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
				PrintWriter out = new PrintWriter(socket.getOutputStream());

				// read the data sent. We basically ignore it,
				// stop reading once a blank line is hit. This
				// blank line signals the end of the client HTTP
				// headers.
				String user = "User";
				String str = ".";
				out.print(user + ": ");
				out.flush();
				while (!str.equals("")) {
					str = in.readLine();
					String params[] = str.split(",");
					System.out.println(str);
					try {
						if (params[0].equals("query")) {
							String utterance = params[1];
							user = params[2];
							out.println("CDM Bot: Debug State ... " + dmMain.processUtterance(utterance, user));
						} else if (params[0].equals("debug")) {
							user = params[1];
							out.println("CDM Bot: Debug State ... " + dmMain.getDMBelief(user));
						} else if (params[0].equals("exit")) {
							out.println("CDM Bot: Exitting ... ");
							break;
						} else {
							out.println("CDM Bot: Sorry :( ... Unknown command" + params[0]);
						}

					} catch (Throwable e) {
						out.println("CDM Bot: Error executing command -" + str);
						e.printStackTrace();
					}
					out.print(user + ": ");
					out.flush();
				}

				out.flush();
				socket.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Start the application.
	 * 
	 * @param args
	 *            Command line parameters are not used.
	 */
	public static void main(String args[]) {
		DMTelnetServer ws = new DMTelnetServer();
		ws.start(args);
	}
}
