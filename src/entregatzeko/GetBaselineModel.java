package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class GetBaselineModel {

	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		baselineSortu(args[0], args[1], args[2]);
	}
	
	public static void baselineSortu(String trainArff, String testArff, String modelPath) throws Exception {
		DataSource source = new DataSource(trainArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		
		DataSource sourceTest = new DataSource(testArff);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(0);
		
		NaiveBayes classifier = new NaiveBayes();
		Evaluation evaluator = new Evaluation(train);
		
		classifier.buildClassifier(train);
		evaluator.evaluateModel(classifier, test);
		
		// Model karpeta ez badago sortuta
		String[] aux = modelPath.split("/");
		String modelName = aux[aux.length-1];
        File modelDirectory = new File(modelPath.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		
		FileWriter f = new FileWriter(modelPath.split("\\.")[0] + "_estimatutakoKalitatea.txt");
		f.write(evaluator.toSummaryString("=== SUMMARY ===", false));
		f.write("\n" + evaluator.toClassDetailsString());
		f.write("\n" + evaluator.toMatrixString());
		f.close();
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
	}
	
}
