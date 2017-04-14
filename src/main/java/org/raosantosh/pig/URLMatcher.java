package org.santrao.pig.udf;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;

public class URLMatcher extends EvalFunc<String> {

	private static Set<String> patterns = new HashSet<String>();

	private String checkIt(String s) {

		System.out.println("pattern size " + patterns.size());

		for (String pattern : patterns) {
			if (s != null && s.contains(pattern)) {
				System.out.println("==" + pattern + "==");
				return "MATCH";
			}
		}

		return "NOT_MATCH";

	}

	public URLMatcher() {

		try {
			if (patterns.isEmpty()) {
				InputStream is = null;
//				InputStream is1 = null;
				try {
					is = URLMatcher.class.getResourceAsStream("/url2.txt");

					Scanner scanner = new Scanner(is).useDelimiter("\n");

					while (scanner.hasNext()) {
						String text = scanner.next();
						patterns.add(text);
					}

//					is1 = URLMatcher.class.getResourceAsStream("/url1.txt");
//
//					Scanner scanner1 = new Scanner(is1).useDelimiter("\n");
//
//					int ctr = 0;
//
//					while (scanner1.hasNext()) {
//						String text = scanner1.next();
//						patterns.add(text);
//					}

				} finally {
					if (is != null)
						is.close();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public String exec(Tuple input) throws IOException {

		if (input == null || input.size() == 0)
			return "NO_MATCH";

		String answer = (String) input.get(0);

		for (String pattern : patterns) {
			if (answer != null && answer.contains(pattern))
				return "MATCH";
		}

		return "NO_MATCH";
	}

	public static void main(String args[]) {

		String s = "he is 8 years older than you!! 13 and 21 sounds ridiculous together... why are you even talking to him";
		System.out.println(s.contains("http://adf.ly/bfMtO"));

		InputStream is = URLMatcher.class.getResourceAsStream("/url2.txt");

		URLMatcher matcher = new URLMatcher();

		String output = matcher
				.checkIt("TVGuide.com has that info for usa tv shows&#xA;&#xA;http://www.tvguide.com/special/fall-preview/fall-preview-a-z.aspx&#xA;&#xA;&#xA;you can also go to the shows website");

		System.out.println("output is " + output);

		// Scanner scanner = new Scanner(is).useDelimiter("\n");

		// while (scanner.hasNext())
		// patterns.add(".*" + scanner.next() + ".*");

		// System.out.println(patterns);
	}

}
