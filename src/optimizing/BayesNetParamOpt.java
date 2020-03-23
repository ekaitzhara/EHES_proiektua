package optimizing;

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
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class BayesNetParamOpt {
	
	public static BayesNetObject optimizatuParametroak(String arffPath) throws Exception {
		BayesNetObject paramsOpt = null;
		
		DataSource source = new DataSource(arffPath);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(0);
		
		int klaseMinoritarioa = klaseMinoritarioaLortu(train);
		
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
		
		BayesNet classifier = new BayesNet();
		Evaluation evaluator = null;
		
		String summary = null;
		String classDetails = null;
		String matrix = null;
		
		double fMeasureOpt = -1.0;
		String batchSizeOpt = null;
		int numDecPlacesOpt = -1;
		BayesNetEstimator estimatorOpt = null;
		SearchAlgorithm searchAlgOpt = null;
		
		
		for (BayesNetEstimator estimator : allEstimators) {
			for (SearchAlgorithm searchAlgorithm : allSearchAlgorithms) {
				try {
					classifier.setEstimator(estimator);
					classifier.setSearchAlgorithm(searchAlgorithm);
					
					classifier.buildClassifier(train);
					evaluator = new Evaluation(train);
					
					evaluator.crossValidateModel(classifier, train, 2, new Random(1));
					System.out.println("fMeasure: " + evaluator.fMeasure(klaseMinoritarioa));
					
					if (evaluator.fMeasure(klaseMinoritarioa) > fMeasureOpt) {
						fMeasureOpt = evaluator.fMeasure(klaseMinoritarioa);
						estimatorOpt = estimator;
						searchAlgOpt = searchAlgorithm;
						
						summary = evaluator.toSummaryString("\n === Summary ===\n", false);
						classDetails = evaluator.toClassDetailsString();
						matrix = evaluator.toMatrixString();
					}
				}catch (Exception e) {	// Parametroren bat ez bada egokia, salta egin dezan
					break;
				}
			}
		}
		
		System.out.println();
		System.out.println("Hoberenaren emaitzak: " + fMeasureOpt+ " lortu duena" );
		System.out.println(summary);
		System.out.println(classDetails);
		System.out.println(matrix);
		
		paramsOpt = new BayesNetObject(estimatorOpt, searchAlgOpt, null, -1000, fMeasureOpt, summary, classDetails, matrix);
		
		return paramsOpt;
	}

	
	public static int klaseMinoritarioaLortu(Instances dataSet) {
		int klaseMinoritarioa = Utils.minIndex(dataSet.attributeStats(dataSet.classIndex()).nominalCounts);
		if (klaseMinoritarioa == 0) {
			int[] classCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts;
			int min = Integer.MAX_VALUE;
			int min_pos = -1;
			for (int i = 0; i < classCounts.length; i++) {
				if ((classCounts[i] < min) && (classCounts[i] != 0)) {
					min = classCounts[i];
					min_pos = i;
				}
			}
			klaseMinoritarioa = min_pos;
		}
		System.out.println("Klase minoritarioa => " + klaseMinoritarioa);
		return klaseMinoritarioa;
	}
}
