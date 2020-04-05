package probak;

import optimizing.BayesNetObject;
import optimizing.BayesNetParamOpt;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;

public class Aa {

	public static void main(String[] args) throws Exception {
		
		BayesNetObject b = new BayesNetObject(new SimpleEstimator(), new K2(), 0.1, 4, 0.74);
		
		BayesNetParamOpt.modeloaGorde(args[0], b, args[1], "BOW", "NonSparse");
	}
}
