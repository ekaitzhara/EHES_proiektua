package probak;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.bayes.net.search.local.LAGDHillClimber;
import weka.classifiers.bayes.net.search.local.RepeatedHillClimber;
import weka.classifiers.bayes.net.search.local.SimulatedAnnealing;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.bayes.net.search.local.TabuSearch;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BayesNetProba {
	
	// Bayes -> BayesNet
	
	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		// Arff-ren izena lortzeko
		String pathCsv = args[0];
		String arff = pathCsv.split("\\.")[0];
		arff = arff + ".arff";
		System.out.println(arff);
		
		BayesNet classifier = new BayesNet();
		Evaluation evaluator = null;
		
		ArrayList<BayesNetEstimator> allEstimators = new ArrayList<BayesNetEstimator>();
		allEstimators.add(new SimpleEstimator());
//		allEstimators.add(new BayesNetEstimator());
		allEstimators.add(new BMAEstimator());
		allEstimators.add(new MultiNomialBMAEstimator());
		
		ArrayList<SearchAlgorithm> allSearchAlgorithms = new ArrayList<SearchAlgorithm>();
		allSearchAlgorithms.add(new K2());
//		allSearchAlgorithms.add(new GeneticSearch());		// Tarda mucho
		allSearchAlgorithms.add(new HillClimber());
		allSearchAlgorithms.add(new LAGDHillClimber());
		allSearchAlgorithms.add(new RepeatedHillClimber());
		allSearchAlgorithms.add(new SimulatedAnnealing());
		allSearchAlgorithms.add(new TabuSearch());
		allSearchAlgorithms.add(new TAN());
		
		
		// batchSize eta numDecimalPlaces eskuz aldatu egin ditut wekan eta ez dit aldaketarik eman emaitzan
		
		for (BayesNetEstimator estimator : allEstimators) {
			for (SearchAlgorithm searchAlgorithm : allSearchAlgorithms) {
				try {
					classifier.setEstimator(estimator);
					classifier.setSearchAlgorithm(searchAlgorithm);
					
					classifier.buildClassifier(train);
					evaluator = new Evaluation(train);
					evaluator.crossValidateModel(classifier, train, 10, new Random(1));
					
					System.out.println("	" + estimator.getClass().getSimpleName() + "	" +
					searchAlgorithm.getClass().getSimpleName() + " => " + evaluator.pctCorrect());
				}catch (Exception e) {	// Parametroren bat ez bada egokia, salta egin dezan
					break;
				}
			}
		}
		
		
	}

}
