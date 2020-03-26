package optimizing;

import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.K2;

public class BayesNetObject {
	
	// Parametroak
	private BayesNetEstimator estimator;
	private K2 searchAlgorithm;
	private double alpha;
	private int maxNrOfParents;
	
	private double fMeasure;
	
	
	public BayesNetObject(BayesNetEstimator estimator, K2 searchAlgorithm, double alpha,
			int maxNrOfParents, double fMeasure) {
		super();
		this.estimator = estimator;
		this.searchAlgorithm = searchAlgorithm;
		this.alpha = alpha;
		this.maxNrOfParents = maxNrOfParents;
		this.fMeasure = fMeasure;
	}

	

	public BayesNetEstimator getEstimator() {
		return estimator;
	}



	public K2 getSearchAlgorithm() {
		return searchAlgorithm;
	}



	public double getAlpha() {
		return alpha;
	}



	public int getMaxNrOfParents() {
		return maxNrOfParents;
	}



	public double getfMeasure() {
		return fMeasure;
	}



	@Override
	public String toString() {
		return "BayesNetObject [estimator=" + estimator.getClass().getSimpleName() + ", searchAlgorithm=" + searchAlgorithm.getClass().getSimpleName() + ", alpha=" + alpha
				+ ", maxNrOfParents=" + maxNrOfParents + ", fMeasure=" + fMeasure + "]";
	}



	

	

}
