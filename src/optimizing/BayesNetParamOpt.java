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
//			for (SearchAlgorithm searchAlgorithm : allSearchAlgorithms) {
			for (int i = 2; i < 11; i++) {
				try {
					classifier.setEstimator(estimator);
					
//					estimator.setAlpha(10.0);
					searchAlgorithm.setMaxNrOfParents(i);
					classifier.setSearchAlgorithm(searchAlgorithm);

					// HOLD-OUT		10 aldiz egin --> for
					// Aukerak
					String errepresentazioa = "BOW";
					String bektoreMota = "NonSparse";
					
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
					
					System.out.println("Estimator: " + estimator.getClass().getSimpleName() + " - searchAlgorithm: " + searchAlgorithm.getClass().getSimpleName() 
							+ " - maxParents: " + i);
					
					evaluator = new Evaluation(train_BOW_FSS);
					classifier.buildClassifier(train_BOW_FSS);
					evaluator.evaluateModel(classifier, dev_BOW_FSS);
					
					System.out.println("fMeasure: " + evaluator.fMeasure(klaseMax));
					
					
					if (evaluator.fMeasure(klaseMax) > fMeasureOpt) {	// Klase minimoarekin -> Nan edo 0.0
						fMeasureOpt = evaluator.fMeasure(klaseMax);
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
		
		paramsOpt = new BayesNetObject(estimatorOpt, searchAlgOpt, null, -1000, fMeasureOpt);
		
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
		
		classifier.setEstimator(paramsOpt.getEstimator());
		classifier.setSearchAlgorithm(paramsOpt.getSearchAlgorithm());
		
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
}
