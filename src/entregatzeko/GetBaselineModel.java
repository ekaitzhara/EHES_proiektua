package entregatzeko;

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
	}
	
}
