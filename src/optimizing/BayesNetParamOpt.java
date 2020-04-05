package optimizing;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import entregatzeko.FSS_InfoGain;
import entregatzeko.FSS_MakeCompatible;
import entregatzeko.TransformRaw;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class BayesNetParamOpt {
	
	public static BayesNetObject optimizatuParametroak(String arffPath, String errepresentazioa, String bektoreMota) throws Exception {
		BayesNetObject paramsOpt = null;
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		SimpleEstimator estimator = new SimpleEstimator();
		K2 searchAlgorithm = new K2();
		
		BayesNet classifier = new BayesNet();
		
		double fMeasureOpt = -1.0;
		double alphaOpt = -1.0;
		int maxNrOfParentsOpt = -1;
		
		for (int i = 1; i < 6; i++) {	// MaxParents
			for (double j = 0.1; j > 0.0001; j=j/10) {	// Alpha
					
				try {
					estimator.setAlpha(j);
					classifier.setEstimator(estimator);
					
					searchAlgorithm.setMaxNrOfParents(i);
					classifier.setSearchAlgorithm(searchAlgorithm);

					// Aukerak
					double fMeasureAvg = holdOutAplikatu(dataSet, arffPath, errepresentazioa, bektoreMota, classifier);
//					double fMeasureAvg = fCVAplikatu(dataSet, arffPath, errepresentazioa, bektoreMota, classifier);
					
					System.out.println("Estimator: " + estimator.getClass().getSimpleName() + " - searchAlgorithm: " + searchAlgorithm.getClass().getSimpleName() 
							+ " - maxParents: " + i + " - alpha: " + j + " | fMeasure => " + fMeasureAvg);	// pctCorrect -> fMeasure
					
					if (fMeasureAvg > fMeasureOpt) {	// Klase minimoarekin -> Nan edo 0.0
						fMeasureOpt = fMeasureAvg;
						alphaOpt = j;
						maxNrOfParentsOpt = i;
					}
				}catch (Exception e) {	// Parametroren bat ez bada egokia, salta egin dezan
					System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
					System.out.println("ERROR:");
					System.out.println("Estimator: " + estimator.getClass().getSimpleName() + " - searchAlgorithm: " + searchAlgorithm.getClass().getSimpleName() 
							+ " - maxParents: " + i + " - alpha: " + j);
					break;
				}
			}
		}
		
		System.out.println();
		System.out.println("Hoberenaren emaitzak: " + fMeasureOpt+ " lortu duena" );
		
		paramsOpt = new BayesNetObject(estimator, searchAlgorithm, alphaOpt, maxNrOfParentsOpt, fMeasureOpt);
		
		return paramsOpt;
	}

	
	public static int klaseMinoritarioaLortu(Instances dataSet) {
		int klaseMinoritarioa = Utils.minIndex(dataSet.attributeStats(dataSet.classIndex()).nominalCounts);
		if (dataSet.attributeStats(dataSet.classIndex()).nominalCounts[klaseMinoritarioa] == 0) {
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
//		System.out.println("Klase minoritarioa => " + klaseMinoritarioa);
		return klaseMinoritarioa;
	}


	public static void modeloaGorde(String arffPath, BayesNetObject paramsOpt, String modelPath, String errepresentazioa, String bektoreMota) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		BayesNet classifier = new BayesNet();
		Evaluation evaluator = new Evaluation(dataSet);
		
		BayesNetEstimator estim = paramsOpt.getEstimator();
		estim.setAlpha(paramsOpt.getAlpha());
		classifier.setEstimator(paramsOpt.getEstimator());
		
		K2 searchAlg = paramsOpt.getSearchAlgorithm();
		searchAlg.setMaxNrOfParents(paramsOpt.getMaxNrOfParents());
		classifier.setSearchAlgorithm(searchAlg);
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
		classifier.buildClassifier(train_BOW_FSS);
		evaluator.evaluateModel(classifier, train_BOW_FSS);
		
		// Model karpeta ez badago sortuta
		String[] auxModel = modelPath.split("/");
		String modelName = auxModel[auxModel.length-1];
        File modelDirectory = new File(modelPath.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		SerializationHelper.write(modelPath, classifier);
		
		FileWriter f = new FileWriter(modelPath.split("\\.")[0] + "Opt_estimatutakoKalitatea.txt");
		f.write(evaluator.toSummaryString("=== SUMMARY ===", false));
		f.write("\n" + evaluator.toClassDetailsString());
		f.write("\n" + evaluator.toMatrixString());
		f.close();
		
		System.out.println(evaluator.toSummaryString("\n=== SUMMARY ===", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	}
	
	
	private static double holdOutAplikatu(Instances dataSet, String arffPath, String errepresentazioa, String bektoreMota, BayesNet classifier) throws Exception {
		double emaitza = -1.0;
		double totala = 0.0;
		int iterazioKop = 10;
		
		System.out.println("--------------");
		
		for (int i = 0; i < iterazioKop; i++) {
			
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
			
			int klaseMinoritarioa = klaseMinoritarioaLortu(dataSet);	// HAU ERABILI BEHAR DA			
			
			Evaluation evaluator = new Evaluation(train_BOW_FSS);
			classifier.buildClassifier(train_BOW_FSS);
			evaluator.evaluateModel(classifier, dev_BOW_FSS);
			
//			System.out.println("	" + i + " bueltaren fMeasure: " + evaluator.fMeasure(klaseMax));
			
			totala = totala + evaluator.fMeasure(klaseMinoritarioa);

		}
//		System.out.println("--------------");
		emaitza = totala / iterazioKop;
		return emaitza;
	}
	
	private static double fCVAplikatu(Instances dataSet, String arffPath, String errepresentazioa, String bektoreMota, BayesNet classifier) throws Exception {
		
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		

		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
		int klaseMax = Utils.maxIndex(train_BOW_FSS.attributeStats(train_BOW_FSS.classIndex()).nominalCounts);
		
		Evaluation evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		
		evaluator.crossValidateModel(classifier, train_BOW_FSS, 10, new Random(1));
		
		return evaluator.fMeasure(klaseMax);
		
	}
}
