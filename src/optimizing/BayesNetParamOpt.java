package optimizing;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.MakeCompatible;
import entregatzeko.TransformRaw;
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
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.Estimator;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class BayesNetParamOpt {
	
	public static BayesNetObject optimizatuParametroak(String arffPath) throws Exception {
		BayesNetObject paramsOpt = null;
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		ArrayList<BayesNetEstimator> allEstimators = new ArrayList<BayesNetEstimator>();
		allEstimators.add(new SimpleEstimator());
//		allEstimators.add(new BayesNetEstimator());
		allEstimators.add(new BMAEstimator());
//		allEstimators.add(new MultiNomialBMAEstimator());
		
		/*
		ArrayList<SearchAlgorithm> allSearchAlgorithms = new ArrayList<SearchAlgorithm>();
		allSearchAlgorithms.add(new K2());
		allSearchAlgorithms.add(new GeneticSearch());		// Tarda mucho
		allSearchAlgorithms.add(new HillClimber());
		allSearchAlgorithms.add(new LAGDHillClimber());
		allSearchAlgorithms.add(new RepeatedHillClimber());
		allSearchAlgorithms.add(new SimulatedAnnealing());
		allSearchAlgorithms.add(new TabuSearch());
		allSearchAlgorithms.add(new TAN());
		*/
		
		K2 searchAlgorithm = new K2();
		
		BayesNet classifier = new BayesNet();
		
		double fMeasureOpt = -1.0;
		BayesNetEstimator estimatorOpt = null;
		double alphaOpt = -1.0;
		int maxNrOfParentsOpt = -1;
		
		
		for (BayesNetEstimator estimator : allEstimators) {
//			for (SearchAlgorithm searchAlgorithm : allSearchAlgorithms) {
			for (int i = 0; i < 6; i++) {	// MaxParents
				for (double j = 0.0; j < 6.0; j=j+1.0) {	// Alpha
						
					try {
						estimator.setAlpha(j);
						classifier.setEstimator(estimator);
						
						searchAlgorithm.setMaxNrOfParents(i);
						classifier.setSearchAlgorithm(searchAlgorithm);
	
						// HOLD-OUT		10 aldiz egin --> for
						// Aukerak
						String errepresentazioa = "BOW";
						String bektoreMota = "NonSparse";
						
						
						double fMeasureAvg = holdOutAplikatu(dataSet, arffPath, errepresentazioa, bektoreMota, classifier);
						
						System.out.println("Estimator: " + estimator.getClass().getSimpleName() + " - searchAlgorithm: " + searchAlgorithm.getClass().getSimpleName() 
								+ " - maxParents: " + i + " - alpha: " + j + " | fMeasureAvg => " + fMeasureAvg);
						
						if (fMeasureAvg > fMeasureOpt) {	// Klase minimoarekin -> Nan edo 0.0
							fMeasureOpt = fMeasureAvg;
							estimatorOpt = estimator;
							alphaOpt = j;
							maxNrOfParentsOpt = i;
						}
					}catch (Exception e) {	// Parametroren bat ez bada egokia, salta egin dezan
						break;
					}
				}
			}
		}
		
		System.out.println();
		System.out.println("Hoberenaren emaitzak: " + fMeasureOpt+ " lortu duena" );
		
		paramsOpt = new BayesNetObject(estimatorOpt, searchAlgorithm, alphaOpt, maxNrOfParentsOpt, fMeasureOpt);
		
		return paramsOpt;
	}

	
	public static int klaseMinoritarioaLortu(Instances dataSet) {
		int klaseMinoritarioa = Utils.minIndex(dataSet.attributeStats(dataSet.classIndex()).nominalCounts);
		if (dataSet.attributeStats(dataSet.classIndex()).nominalCounts[klaseMinoritarioa] == 0) {
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


	public static void modeloaGorde(String arffPath, BayesNetObject paramsOpt, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		BayesNet classifier = new BayesNet();
		Evaluation evaluator = new Evaluation(dataSet);
		
		BayesNetEstimator estim = paramsOpt.getEstimator();
		estim.setAlpha(paramsOpt.getAlpha());
		classifier.setEstimator(paramsOpt.getEstimator());
		
		K2 searchAlg = paramsOpt.getSearchAlgorithm();
		searchAlg.setMaxNrOfParents(paramsOpt.getMaxNrOfParents());
		classifier.setSearchAlgorithm(searchAlg);
		
		classifier.buildClassifier(dataSet);
		evaluator.evaluateModel(classifier, dataSet);
		
		// Model karpeta ez badago sortuta
		String[] auxModel = modelPath.split("/");
		String modelName = auxModel[auxModel.length-1];
        File modelDirectory = new File(modelPath.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		SerializationHelper.write(modelPath, classifier);
		
		FileWriter f = new FileWriter(modelPath.split("\\.")[0] + "Opt_estimatutakoKalitatea.txt");
		f.write(evaluator.toSummaryString("=== SUMMARY ===", false));
		f.write("\n" + evaluator.toClassDetailsString());
		f.write("\n" + evaluator.toMatrixString());
		f.close();
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}
	
	
	private static double holdOutAplikatu(Instances dataSet, String arffPath, String errepresentazioa, String bektoreMota, BayesNet classifier) throws Exception {
		double emaitza = -1.0;
		double totala = 0.0;
		int iterazioKop = 10;
		
		System.out.println("--------------");
		
		for (int i = 0; i < iterazioKop; i++) {
			
			int seed = 1;
			dataSet.randomize(new Random(seed));
			RemovePercentage removePercentage = new RemovePercentage();
			
			// Train zatia lortzeko
			removePercentage.setInputFormat(dataSet);
			removePercentage.setPercentage(70);
			removePercentage.setInvertSelection(true);	// %70-a lortzeko
			Instances train = Filter.useFilter(dataSet, removePercentage);
			
			// Dev zatia lortzeko
			removePercentage.setInputFormat(dataSet);
			removePercentage.setInvertSelection(false);
			Instances dev = Filter.useFilter(dataSet, removePercentage);
			
			String[] aux = arffPath.split("/");
			String direktorioa = arffPath.replace(aux[aux.length-1],"");
			String dictionaryPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt";
			
			Instances train_BOW = TransformRaw.transformRawInstances(train, errepresentazioa, bektoreMota, dictionaryPath);
			
			Instances dev_BOW = MakeCompatible.makeCompatibleInstances(dev, dictionaryPath);
			
			Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
			
			Instances dev_BOW_FSS = FSS_MakeCompatible.make2InstancesCompatibles(train_BOW_FSS, dev_BOW);
			
	//					int klaseMinoritarioa = klaseMinoritarioaLortu(dataSet);	// HAU ERABILI BEHAR DA
			int klaseMax = Utils.maxIndex(train_BOW_FSS.attributeStats(train_BOW_FSS.classIndex()).nominalCounts);
			
			
			Evaluation evaluator = new Evaluation(train_BOW_FSS);
			classifier.buildClassifier(train_BOW_FSS);
			evaluator.evaluateModel(classifier, dev_BOW_FSS);
			
			System.out.println("	" + i + " bueltaren fMeasure: " + evaluator.fMeasure(klaseMax));
			totala = totala + evaluator.fMeasure(klaseMax);
		}
		System.out.println("--------------");
		emaitza = totala / iterazioKop;
		return emaitza;
	}
	
	private static double fCVAplikatu(Instances dataSet, String arffPath, String errepresentazioa, String bektoreMota, BayesNet classifier) throws Exception {
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt";
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota, dictionaryPath);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		int klaseMax = Utils.maxIndex(train_BOW_FSS.attributeStats(train_BOW_FSS.classIndex()).nominalCounts);
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		
		evaluator.crossValidateModel(classifier, dataSet, 10, new Random(1));
		
		return evaluator.fMeasure(klaseMax);
		
	}
}
