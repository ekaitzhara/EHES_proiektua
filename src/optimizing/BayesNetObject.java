package optimizing;

import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;

public class BayesNetObject {
	
	// Parametroak
	private BayesNetEstimator estimator;
	private SearchAlgorithm searchAlgorithm;
	private String batchSize;
	private int numDecimalPlaces;
	
	private double fMeasure;
	
	public BayesNetObject(BayesNetEstimator pEst, SearchAlgorithm searchAlg) {
		
		this.estimator = pEst;
		this.searchAlgorithm = searchAlg;
	}

	public BayesNetObject(BayesNetEstimator estimator, SearchAlgorithm searchAlgorithm, String batchSize,
			int numDecimalPlaces, double fMeasure) {
		
		this.estimator = estimator;
		this.searchAlgorithm = searchAlgorithm;
		this.batchSize = batchSize;
		this.numDecimalPlaces = numDecimalPlaces;
		this.fMeasure = fMeasure;
	}

	
	
	public BayesNetEstimator getEstimator() {
		return estimator;
	}

	public SearchAlgorithm getSearchAlgorithm() {
		return searchAlgorithm;
	}

	public String getBatchSize() {
		return batchSize;
	}

	public int getNumDecimalPlaces() {
		return numDecimalPlaces;
	}

	public double getfMeasure() {
		return fMeasure;
	}


	@Override
	public String toString() {
		return "BayesNetObject [estimator=" + estimator.getClass().getSimpleName() + ", searchAlgorithm=" + searchAlgorithm.getClass().getSimpleName() + ", batchSize="
				+ batchSize + ", numDecimalPlaces=" + numDecimalPlaces + ", fMeasure=" + fMeasure + "]";
	}

	

}
