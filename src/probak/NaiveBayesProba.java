package probak;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesProba {
	
	public static void main(String[] args) throws Exception {
		
		DataSource source = new DataSource(args[0]);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(0);
		
		DataSource sourceTest = new DataSource(args[1]);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(0);
		
		System.out.println(train.numInstances());
		System.out.println(test.numInstances());
		
		NaiveBayes classifier = new NaiveBayes();
		Evaluation evaluator = new Evaluation(train);
		classifier.buildClassifier(train);
		evaluator.evaluateModel(classifier, test);
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
	}

}
