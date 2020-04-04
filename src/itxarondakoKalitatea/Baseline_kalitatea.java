package itxarondakoKalitatea;

import java.io.FileWriter;
import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Baseline_kalitatea {
	
	public static void main(String[] args) throws Exception {
		if(args.length == 0) {
			
			System.out.println("SARTU ONDO ARGUMENTUAK\n");
			
			
			System.exit(0);
		}
		
//		holdOutAplikatu(args[0], args[1]);
//		fCVAplikatu(args[0], args[1]);
		reSubstitution(args[0], args[1]);
	}
	
	
	public static void holdOutAplikatu(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		NaiveBayes classifier = (NaiveBayes) SerializationHelper.read(modelPath);
		
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		double precisionNUM = 0.0;
		double precisionLOC = 0.0;
		double precisionHUM = 0.0;
		double precisionDESC = 0.0;
		double precisionENTY = 0.0;
		double precisionABBR = 0.0;
		
		double recallNUM = 0.0;
		double recallLOC = 0.0;
		double recallHUM = 0.0;
		double recallDESC = 0.0;
		double recallENTY = 0.0;
		double recallABBR = 0.0;
		
		double fMNUM = 0.0;
		double fMLOC = 0.0;
		double fMHUM = 0.0;
		double fMDESC = 0.0;
		double fMENTY = 0.0;
		double fMABBR = 0.0;
		
		
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
			
			Instances train_BOW = TransformRaw.transformRawInstances(train, errepresentazioa, bektoreMota);
			
			Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
			
			String[] aux = arffPath.split("/");
			String direktorioa = arffPath.replace(aux[aux.length-1],"");
			String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
			
			FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
			
			Instances dev_BOW_FSS = FSS_MakeCompatible.makeFSSCompatibleInstances(dev, dictionaryFSSPath);
			
			Evaluation evaluator = new Evaluation(train_BOW_FSS);
			classifier.buildClassifier(train_BOW_FSS);
			evaluator.evaluateModel(classifier, dev_BOW_FSS);
			
			precisionNUM = precisionNUM + evaluator.precision(0);
			precisionLOC = precisionLOC + evaluator.precision(1);
			precisionHUM = precisionHUM + evaluator.precision(2);
			precisionDESC = precisionDESC + evaluator.precision(3);
			precisionENTY = precisionENTY + evaluator.precision(4);
			precisionABBR = precisionABBR + evaluator.precision(5);
			
			recallNUM = recallNUM + evaluator.recall(0);
			recallLOC = recallLOC + evaluator.recall(1);
			recallHUM = recallHUM + evaluator.recall(2);
			recallDESC = recallDESC + evaluator.recall(3);
			recallENTY = recallENTY + evaluator.recall(4);
			recallABBR = recallABBR + evaluator.recall(5);
			
			fMNUM = fMNUM  + evaluator.fMeasure(0);
			fMLOC = fMLOC  + evaluator.fMeasure(1);
			fMHUM = fMHUM  + evaluator.fMeasure(2);
			fMDESC = fMDESC  + evaluator.fMeasure(3);
			fMENTY = fMENTY  + evaluator.fMeasure(4);
			fMABBR = fMABBR  + evaluator.fMeasure(5);
			
		}
		
		System.out.println("Precision");
		System.out.println("NUM: " + precisionNUM / 100);
		System.out.println("LOC: " + precisionLOC / 100);
		System.out.println("HUM: " + precisionHUM / 100);
		System.out.println("DESC: " + precisionDESC / 100);
		System.out.println("ENTY: " + precisionENTY / 100);
		System.out.println("ABBR: " + precisionABBR / 100);
		System.out.println("--------------");
		
		System.out.println("Recall");
		System.out.println("NUM: " + recallNUM / 100);
		System.out.println("LOC: " + recallLOC / 100);
		System.out.println("HUM: " + recallHUM / 100);
		System.out.println("DESC: " + recallDESC / 100);
		System.out.println("ENTY: " + recallENTY / 100);
		System.out.println("ABBR: " + recallABBR / 100);
		System.out.println("--------------");
		
		System.out.println("F-Measure");
		System.out.println("NUM: " + fMNUM / 100);
		System.out.println("LOC: " + fMLOC / 100);
		System.out.println("HUM: " + fMHUM / 100);
		System.out.println("DESC: " + fMDESC / 100);
		System.out.println("ENTY: " + fMENTY / 100);
		System.out.println("ABBR: " + fMABBR / 100);
		
		
		
	}

	
	public static void fCVAplikatu(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		NaiveBayes classifier = (NaiveBayes) SerializationHelper.read(modelPath);
		
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		
		evaluator.crossValidateModel(classifier, train_BOW_FSS, 10, new Random(1));
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}
	
	public static void reSubstitution(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		NaiveBayes classifier = (NaiveBayes) SerializationHelper.read(modelPath);
		
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		evaluator.evaluateModel(classifier, train_BOW_FSS);
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}
}
