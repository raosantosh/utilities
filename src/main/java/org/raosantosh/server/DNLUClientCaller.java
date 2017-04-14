package com.yahoo.cdm.server;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URLDecoder;
import java.net.URLEncoder;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.json.JSONObject;

public class DNLUClientCaller {

	private String apiPrefix = "http://perf-serving03.broadway.gq1.yahoo.com:4080/nlu?debug=true&sentence=";

	private HttpClient client = new DefaultHttpClient();

	public JSONObject getResponseFromServer(String sentence) {

		StringBuffer apiResponse = new StringBuffer();

		try {
			HttpGet request = new HttpGet(apiPrefix + URLEncoder.encode(sentence));
			HttpResponse response = client.execute(request);

			// Get the response
			BufferedReader rd = new BufferedReader(new InputStreamReader(response.getEntity().getContent()));

			String line = "";
			while ((line = rd.readLine()) != null) {
				apiResponse.append(line);
			}

		} catch (Exception exception) {
			exception.printStackTrace();
			System.out.println("API call failed for: " + sentence);
		}

		return new JSONObject(apiResponse.toString());
	}

	public static void main(String args[]) {
		DNLUClientCaller caller = new DNLUClientCaller();

		JSONObject response = caller.getResponseFromServer(
				"i want to have dinner with amit near his house");

//		System.out.println(response.toString(2)); 
		
		response = caller.getResponseFromServer(
				"i want to have dinner with amit on his birthday");
		
//		System.out.println(response.toString(2));
		
		response = caller.getResponseFromServer("I'm having brunch with my CEO, Jerry Jones, so I need a reservation for 2 at 7 am at Dak Prescott's in Irving.");
		System.out.println(response.toString(2));
		
		response = caller.getResponseFromServer("add santosh for dinner");
		System.out.println(response.toString(2));
		
		response = caller.getResponseFromServer("Dinner with Amit. Alex also coming");
//		System.out.println(response.toString(2));
	
	}

}
