package com.yahoo.cdm.server;

import java.io.IOException;
import java.net.URI;
import java.util.Properties;

import com.yahoo.cdm.main.CDMMain;

public class DMlHttpServer extends NanoHTTPD {

	private static final Response ERROR = new Response(HTTP_BADREQUEST, MIME_PLAINTEXT, "unsupported request");
	private static final Response OK = new Response(HTTP_OK, MIME_PLAINTEXT, "DONE");
	private static final Response NO_RESULT = new Response(HTTP_OK, MIME_PLAINTEXT, "No results");

	private CDMMain cdmMain = new CDMMain();

	public DMlHttpServer(int port) throws IOException {
		super(port);
	}

	public static void main(String[] args) throws Exception {
		int port = 12346;
		try {
			port = Integer.parseInt(args[0]);
		} catch (Exception e) {

		}
		System.out.println("Service started at Port: " + port);
		DMlHttpServer webService = new DMlHttpServer(port);
		webService.start();
	}

	@Override
	public Response serve(String iUri, String method, Properties header, Properties params, Properties files)
			throws IOException, InterruptedException {
		URI uri;
		try {
			uri = new URI(iUri);
		} catch (Exception e) {
			e.printStackTrace();
			return ERROR;
		}

		String responseText = getResponseData(method, iUri, params);

		Response response = new Response(HTTP_OK, "text/json", responseText);
		response.addHeader("Content-Length", responseText.getBytes().length + "");

		return new Response(HTTP_OK, "text/json", responseText);
	}

	private String getResponseData(String method, String iUri, Properties params) {
		String data = "";
		try {

			String[] uriParts = iUri.split("/");
			String user = "newUser";
			String utterance = "dummy";

			System.out.println("Received call: " + iUri);

			if (method.equals("GET")) {
				if (iUri.startsWith("/query")) {
					utterance = params.getProperty("utt");
					user = params.getProperty("user");
					data = cdmMain.processUtterance(utterance, user);

				} else if (iUri.startsWith("/belief")) {
					user = params.getProperty("user");
					data = cdmMain.getDMBelief(user);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.println("Output: " + data);

		return data;
	}
}
