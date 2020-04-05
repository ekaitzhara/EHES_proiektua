package optimizing;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;

/**
 * {@link BayesNet} algoritmorako parametro optimoak lortzeko {@link Class}.
 * Parametro optimoak gordetzeko objektua da klase hau.
 * 
 * @author ekaitzhara
 *
 */
public class BayesNetObject {
	
	// Parametroak
	private SimpleEstimator estimator;
	private K2 searchAlgorithm;
	private double alpha;
	private int maxNrOfParents;
	
	private double fMeasure;
	
	
	public BayesNetObject(SimpleEstimator estimator, K2 searchAlgorithm, double alpha,
			int maxNrOfParents, double fMeasure) {
		super();
		this.estimator = estimator;
		this.searchAlgorithm = searchAlgorithm;
		this.alpha = alpha;
		this.maxNrOfParents = maxNrOfParents;
		this.fMeasure = fMeasure;
	}

	

	public SimpleEstimator getEstimator() {
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
