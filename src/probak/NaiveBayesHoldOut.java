package probak;

import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.GetRaw;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class NaiveBayesHoldOut {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
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
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
			
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
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		Instances train_BOW = TransformRaw.transformRawInstances(train, errepresentazioa, bektoreMota);
				
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		Instances dev_BOW_FSS = FSS_MakeCompatible.makeFSSCompatibleInstances(dev, dictionaryFSSPath);
		
		int klaseMinoritarioa = klaseMinoritarioaLortu(dataSet);	// HAU ERABILI BEHAR DA
		
		BayesNet classifier = new BayesNet();
		classifier.setEstimator(new BMAEstimator());
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		evaluator.evaluateModel(classifier, dev_BOW_FSS);
		
		System.out.println("fMeasure: " + evaluator.fMeasure(klaseMinoritarioa));
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}

	public static int klaseMinoritarioaLortu(Instances dataSet) {
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
		return klaseMinoritarioa;
	}
	
	

}
