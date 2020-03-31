package probak;

import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class fCV_proba {
	
	public static void main(String[] args) throws Exception {

		String arffPath = args[0];
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		BayesNet classifier = new BayesNet();
			
		String errepresentazioa = "TF";
		String bektoreMota = "NonSparse";
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt";
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota, dictionaryPath);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
//		int klaseMax = Utils.maxIndex(train_BOW.attributeStats(train_BOW.classIndex()).nominalCounts);
		
		Evaluation evaluator = new Evaluation(train_BOW);
		classifier.buildClassifier(train_BOW_FSS);
		
		evaluator.crossValidateModel(classifier, train_BOW_FSS, 10, new Random(1));
		
		int klaseMinoritarioa = NaiveBayesHoldOut.klaseMinoritarioaLortu(dataSet);
		
		System.out.println("fMeasure: " + evaluator.fMeasure(klaseMinoritarioa));
		System.out.println(train_BOW_FSS.numAttributes());
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}

}
