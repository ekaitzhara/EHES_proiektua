package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class GetBaselineModel {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw entrenamendu datu-sortarako aplikatu izana, horrela arff zuzena sortuta edukiko duzulako");
			System.out.println("	Erabaki nola deituko den modeloa");
			System.out.println("	Sartu beharreko modeloaren path-a honelakoa izan behar da: /home/erabiltzaileIzena/workdir/Model/adibide.model");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Model direktorioa automatikoki sortuko du ez badago sortuta");
			System.out.println("	Aukeratu duzun izenarekin gordeko da modeloa");
			System.out.println("	Aurrebalditzetan aipatutako path-a ondo sartu baduzu, ''Model'' karpetan edukiko duzu modeloa");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Modeloa sortzeko erabiliko den arff-a");
			System.out.println("	2 -> Non gorde nahi duzun modelo fitxategiaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar getBaselineModel.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/workdir/Model/adibide.model\n");
			
			System.exit(0);
		}
		
		baselineSortu(args[0], args[1]);
	}
	
	public static void baselineSortu(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		// Baseline-a gure kasuan NaiveBayes da
		NaiveBayes classifier = new NaiveBayes();
		Evaluation evaluator = null;
		
		// Aukerak
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		// Errepresentaziodun instantziak lortu
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		// Atributu hautapena bete, horiek murrizteko
		Instances train_BOW_FSS = FSS_InfoGain.atributuenHautapenaInstances(train_BOW);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryFSSPath = direktorioa + "/train_" + errepresentazioa+ "_FSS_dictionary.txt";
		
		FSS_MakeCompatible.gordeHiztegia(train_BOW_FSS, dictionaryFSSPath);
		
		evaluator = new Evaluation(train_BOW_FSS);
		classifier.buildClassifier(train_BOW_FSS);
		evaluator.evaluateModel(classifier, train_BOW_FSS);
		
		// Model karpeta ez badago sortuta
		String[] auxModel = modelPath.split("/");
		String modelName = auxModel[auxModel.length-1];
        File modelDirectory = new File(modelPath.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
        // Modeloa gorde
		SerializationHelper.write(modelPath, classifier);
		System.out.println(modelPath + " modeloa gordeta");
        
		FileWriter f = new FileWriter(modelPath.split("\\.")[0] + "_estimatutakoKalitatea.txt");
		f.write(evaluator.toSummaryString("=== SUMMARY ===", false));
		f.write("\n" + evaluator.toClassDetailsString());
		f.write("\n" + evaluator.toMatrixString());
		f.close();
		
		System.out.println("\n######################");
		System.out.println("ESTIMATUTAKO KALITATEA");
		System.out.println("######################");
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
