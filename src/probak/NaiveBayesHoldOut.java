package probak;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class NaiveBayesHoldOut {
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		nb_holdOut_aplikatu(args[0]);
		
	}
	
	public static void nb_holdOut_aplikatu(String arffPath) throws Exception {
		
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(0);
		
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
		
		String summary = null;
		String classDetails = null;
		String matrix = null;
		double fMeasureOpt = -1.0;
		
		for (int i = 0; i < 50; i++) {	// 50 aldiz entrenatuko dugu, randomize-k aleatorioa egiten duelako
			
			int seed = 1;
			dataSet.randomize(new Random(seed));
			RemovePercentage removePercentage = new RemovePercentage();
			
			// Train zatia lortzeko
			removePercentage.setInputFormat(dataSet);
			removePercentage.setPercentage(70);
			removePercentage.setInvertSelection(true);	// %70-a lortzeko
			Instances train = Filter.useFilter(dataSet, removePercentage);
			
			// Test zatia lortzeko
			removePercentage.setInputFormat(dataSet);
			removePercentage.setInvertSelection(false);
			Instances test = Filter.useFilter(dataSet, removePercentage);
			
			NaiveBayes classifier = new NaiveBayes();
			classifier.buildClassifier(train);
			
			Evaluation evaluator = new Evaluation(train);
			evaluator.evaluateModel(classifier, test);
			System.out.println(i+" bueltaren zuzenen fMeasure: " + evaluator.fMeasure(klaseMinoritarioa));
			
			if (evaluator.fMeasure(klaseMinoritarioa) > fMeasureOpt) {
				fMeasureOpt = evaluator.fMeasure(klaseMinoritarioa);
				summary = evaluator.toSummaryString("\n === Summary ===\n", false);
				classDetails = evaluator.toClassDetailsString();
				matrix = evaluator.toMatrixString();
			}
		}	
		System.out.println();
		System.out.println("Hoberenaren emaitzak: " + fMeasureOpt+ " lortu duena" );
		System.out.println(summary);
		System.out.println(classDetails);
		System.out.println(matrix);
	
		
	}

}
