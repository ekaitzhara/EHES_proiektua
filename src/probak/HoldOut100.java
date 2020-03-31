package probak;

import java.io.FileWriter;
import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.MakeCompatible;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOut100 {
	
	public static void main(String[] args) throws Exception {
		String arffPath = args[0];
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
			
		String errepresentazioa = "TF";
		String bektoreMota = "NonSparse";
		
		double pctCorrect = 0.0;
		double fMeasure = 0.0;
		
		for (int i = 0; i < 100; i++) {
			
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
			
			
			int klaseMinoritarioa = NaiveBayesHoldOut.klaseMinoritarioaLortu(dataSet);	// HAU ERABILI BEHAR DA
//			int klaseMax = Utils.maxIndex(train_BOW.attributeStats(train_BOW.classIndex()).nominalCounts);
			
			BayesNet classifier = new BayesNet();
			
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
		
		pctCorrect = pctCorrect / 100;
		fMeasure = fMeasure / 100;
		
		System.out.println("pctCorrect: " + pctCorrect);
		System.out.println("fMeasure: " + fMeasure);
	}

}
