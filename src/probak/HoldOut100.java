package probak;

import java.io.FileWriter;
import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.MakeCompatible;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOut100 {
	
	public static void main(String[] args) throws Exception {
		String arffPath = args[0];
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		double pctCorrect = 0.0;
		double fMeasure = 0.0;
		
		long startTime = System.currentTimeMillis();
		Runtime rt = Runtime.getRuntime();
		long startMemory = (rt.totalMemory()-rt.freeMemory()) / (1024 * 1024);
		
		for (int i = 0; i < 10; i++) {
			
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
			
			Instances train_BOW = TransformRaw.transformRawInstances(train, errepresentazioa, bektoreMota);
			
			Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
			
			String[] aux = arffPath.split("/");
			String direktorioa = arffPath.replace(aux[aux.length-1],"");
			String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
			
			FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
			
			Instances dev_BOW_FSS = FSS_MakeCompatible.makeFSSCompatibleInstances(dev, dictionaryFSSPath);
			
			int klaseMinoritarioa = NaiveBayesHoldOut.klaseMinoritarioaLortu(dataSet);	// HAU ERABILI BEHAR DA
			
			BayesNet classifier = new BayesNet();
			
//			SimpleEstimator estimator = new SimpleEstimator();
//			estimator.setAlpha(0.1);
//			classifier.setEstimator(estimator);
			
//			K2 searchAlgorithm = new K2();
//			searchAlgorithm.setMaxNrOfParents(5);
//			classifier.setSearchAlgorithm(searchAlgorithm);
			
			Evaluation evaluator = new Evaluation(train_BOW_FSS);
			classifier.buildClassifier(train_BOW_FSS);
			evaluator.evaluateModel(classifier, dev_BOW_FSS);
			
			pctCorrect = pctCorrect + evaluator.pctCorrect();
			fMeasure = fMeasure + evaluator.fMeasure(klaseMinoritarioa);
			System.out.println(i + " - " + evaluator.pctCorrect());
			System.out.println();
			
			System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
			System.out.println(evaluator.toClassDetailsString());
			System.out.println(evaluator.toMatrixString());
		}
		
		pctCorrect = pctCorrect / 10;
		fMeasure = fMeasure / 10;
		
		long holdOut_time = (System.currentTimeMillis()-startTime)/1000;
		rt = Runtime.getRuntime();
		long finalMemory = (rt.totalMemory()-rt.freeMemory()) / (1024 * 1024);
		
		System.out.println("pctCorrect: " + pctCorrect);
		System.out.println("fMeasure: " + fMeasure);
		System.out.println("\nHold-out Denbora: " + holdOut_time + " seg");
		System.out.println("Used memory: " + (finalMemory - startMemory) + " MB");
	}

}
