package entregatzeko;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class GetBaselineModel {

	// PROZEDURA:	
	// raw arff -> train,dev -> train_BOW_9999 (StringToWordVector) -> dictionary_9999 ->
	// -> dev_BOW_9999 (compatible) -> train_BOW_1000 (FSS) -> dev_BOW_1000 (FSS compatible) ->
	// -> bateragarriak -> NaiveBayes
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		baselineSortu(args[0], args[1]);
	}
	
	public static void baselineSortu(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		NaiveBayes classifier = new NaiveBayes();
		Evaluation evaluator = null;
		
		// Aukerak
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
		String dictionaryPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt";
		
		Instances train_BOW = TransformRaw.transformRawInstances(train, errepresentazioa, bektoreMota, dictionaryPath);
		
		Instances dev_BOW = MakeCompatible.makeCompatibleInstances(dev, dictionaryPath);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		Instances dev_BOW_FSS = FSS_MakeCompatible.make2InstancesCompatibles(train_BOW_FSS, dev_BOW);
		
		evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		evaluator.evaluateModel(classifier, dev_BOW_FSS);
		
		// Model karpeta ez badago sortuta
		String[] auxModel = modelPath.split("/");
		String modelName = auxModel[auxModel.length-1];
        File modelDirectory = new File(modelPath.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		SerializationHelper.write(modelPath, classifier);
		System.out.println(modelPath + " modeloa gordeta");
        
		FileWriter f = new FileWriter(modelPath.split("\\.")[0] + "_estimatutakoKalitatea.txt");
		f.write(evaluator.toSummaryString("=== SUMMARY ===", false));
		f.write("\n" + evaluator.toClassDetailsString());
		f.write("\n" + evaluator.toMatrixString());
		f.close();
		
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
