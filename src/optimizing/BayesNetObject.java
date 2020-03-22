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
	private String summary;
	private String classDetails;
	private String matrix;
	
	public BayesNetObject(BayesNetEstimator pEst, SearchAlgorithm searchAlg) {
		
		this.estimator = pEst;
		this.searchAlgorithm = searchAlg;
	}

	public BayesNetObject(BayesNetEstimator estimator, SearchAlgorithm searchAlgorithm, String batchSize,
			int numDecimalPlaces, double fMeasure, String summary, String classDetails, String matrix) {
		
		this.estimator = estimator;
		this.searchAlgorithm = searchAlgorithm;
		this.batchSize = batchSize;
		this.numDecimalPlaces = numDecimalPlaces;
		this.fMeasure = fMeasure;
		this.summary = summary;
		this.classDetails = classDetails;
		this.matrix = matrix;
	}

	@Override
	public String toString() {
		return "BayesNetObject [estimator=" + estimator.getClass().getSimpleName() + ", searchAlgorithm=" + searchAlgorithm.getClass().getSimpleName() + ", batchSize="
				+ batchSize + ", numDecimalPlaces=" + numDecimalPlaces + ", fMeasure=" + fMeasure + ", \nsummary="
				+ summary + ", \nclassDetails=" + classDetails + ", \nmatrix=" + matrix + "]";
	}

	

}
