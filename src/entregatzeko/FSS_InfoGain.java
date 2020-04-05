package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FSS_InfoGain {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw aplikatu izana, horrela arff zuzena sortuta edukiko duzulako");
			System.out.println("	Gainera, TranformRaw ere aplikatuta egon behar da, sortu den fitxategia lehenengo argumentua izango da");
			System.out.println("	Erabaki nola deituko den fss fitxategia. Honela izan behar du: ad. adibidea_BOW_FSS.arff");
			System.out.println("	Sartu beharreko fss-aren path-a honelakoa izan behar da: /home/erabiltzaileIzena/workdir/Transform/FSS/adibide.model");
			System.out.println("	Erabaki datuen zein portzentai gertatzea nahi duzun");
			System.out.println("Ondorengo balditza:");
			System.out.println("	FSS direktorioa automatikoki sortuko du ez badago sortuta");
			System.out.println("	Aukeratu duzun izenarekin gordeko da FSS arff berria");
			System.out.println("	Aurrebalditzetan aipatutako path-a ondo sartu baduzu, ''FSS'' karpetan edukiko duzu modeloa");
			System.out.println("	Fitxategi berri hori, erabaki duzun portzentaiagan atributu kopurua izango ditu. 75 sartuta atributuen artean informazio gutxien eskaintzen duten %25 kenduko ditu");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> TranformRaw-ean sortutako arff fitxategia");
			System.out.println("	2 -> Non gorde nahi duzun FSS arff berriaren path-a");
			System.out.println("	3 -> Datuen zein portzentai geratuko diren filtroa aplikatu ostean");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar FSS.jar /home/erabiltzaileIzena/workdir/Transform/adibideBOW.arff /home/erabiltzaileIzena/workdir/Transform/FSS/adibideFSS.arff 10\n");

			System.exit(0);
		}
		
		atributuenHautapenaInfoGain(args[0], args[1], Integer.valueOf(args[2]));
	}
	
	public static void atributuenHautapenaInfoGain(String trainArff, String arffToSave, int portzentaia) throws Exception {
		
		System.out.println(trainArff + " fitxategiko atributuen %" + portzentaia + "-rekin utziko diogu");
		DataSource source = new DataSource(trainArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(0);
		
		Instances newTrain = FSS_InfoGain.atributuenHautapenaInstancesPortzentai(train, portzentaia);
		
		String[] aux = arffToSave.split("/");
		newTrain.setRelationName(aux[aux.length-1].split("\\.")[0]);
		
		// FSS karpeta ez badago sortuta, sortu
        File modelDirectory = new File(arffToSave.replace("/" + aux[aux.length-1], ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		FileWriter f = new FileWriter(arffToSave);
		f.write(newTrain.toString());
		f.close();
		System.out.println("\nFitxategi berria atributuak murriztuta hemen gorde egin da: "
				+ "\n	" + arffToSave);
	}
	
	public static Instances atributuenHautapenaInstances(Instances dataSet) throws Exception {
		
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(0);
		String relationName = dataSet.relationName();
		
		AttributeSelection attSelection = new AttributeSelection();
		InfoGainAttributeEval attEvaluator = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		
		// Aukeratutako balioak parametroentzako
		ranker.setNumToSelect((dataSet.numAttributes() * 10 / 100));
		ranker.setThreshold(0.0);
		
		attSelection.setEvaluator(attEvaluator);
		attSelection.setSearch(ranker);
		
		attSelection.SelectAttributes(dataSet);
		
		dataSet = attSelection.reduceDimensionality(dataSet);
		dataSet.setRelationName(relationName);
		
		return dataSet;
	}
	
	public static Instances atributuenHautapenaInstancesPortzentai(Instances dataSet, Integer portzentaia) throws Exception {
		
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(0);
		String relationName = dataSet.relationName();
		
		AttributeSelection attSelection = new AttributeSelection();
		InfoGainAttributeEval attEvaluator = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		
		// Kasu honeta portzentaia dinamikoa da
		ranker.setNumToSelect((dataSet.numAttributes() * portzentaia / 100));
		ranker.setThreshold(0.0);
		
		attSelection.setEvaluator(attEvaluator);
		attSelection.setSearch(ranker);
		
		attSelection.SelectAttributes(dataSet);
		
		dataSet = attSelection.reduceDimensionality(dataSet);
		dataSet.setRelationName(relationName);
		
		return dataSet;
	}

}
