package probak;

import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ezZintzoa_proba {

	public static void main(String[] args) throws Exception {

		long startTime = System.nanoTime();
		
		String arffPath = args[0];
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		BayesNet classifier = new BayesNet();
			
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_FSS_dictionary.txt";
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		System.out.println(train_BOW_FSS.numAttributes());
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
//		int klaseMax = Utils.maxIndex(train_BOW.attributeStats(train_BOW.classIndex()).nominalCounts);
		
		SimpleEstimator estimator = new SimpleEstimator();
		estimator.setAlpha(0.0001);
		classifier.setEstimator(estimator);
		
		K2 searchAlgorithm = new K2();
		searchAlgorithm.setMaxNrOfParents(5);
		classifier.setSearchAlgorithm(searchAlgorithm);
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		
		evaluator.evaluateModel(classifier, train_BOW_FSS);
		
		int klaseMinoritarioa = NaiveBayesHoldOut.klaseMinoritarioaLortu(dataSet);
		
		System.out.println("fMeasure: " + evaluator.fMeasure(klaseMinoritarioa));
		System.out.println(train_BOW_FSS.numAttributes());
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
		long fCV_time = (System.nanoTime()-startTime)/1000000000;
		System.out.println("\nfCV Denbora: " + fCV_time + " seg");
		
	}
}
