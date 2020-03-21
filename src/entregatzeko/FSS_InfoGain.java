package entregatzeko;

import java.io.FileWriter;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FSS_InfoGain {
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		atributuenHautapenaInfoGain(args[0], args[1]);
	}
	
	public static void atributuenHautapenaInfoGain(String trainArff, String arffToSave) throws Exception {
		
		DataSource source = new DataSource(trainArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(0);
		String relationName = train.relationName();
		
		AttributeSelection attSelection = new AttributeSelection();
		InfoGainAttributeEval attEvaluator = new InfoGainAttributeEval();
		Ranker ranker = new Ranker();
		
		ranker.setNumToSelect((int) (train.numAttributes() * 0.8));	// % 80 jarriko diogu
//		ranker.setThreshold(2.0);
		attSelection.setEvaluator(attEvaluator);
		attSelection.setSearch(ranker);
		
		attSelection.SelectAttributes(train);
		
		Instances newTrain = attSelection.reduceDimensionality(train);
		
		System.out.println("Instances -   Attributes -  Classes");
		System.out.println(train.numInstances() + " -		 " + train.numAttributes() + " -		" + train.numClasses());
		System.out.println(newTrain.numInstances() + " -		 " + newTrain.numAttributes() + " - 	" + newTrain.numClasses());
		
		String[] aux = arffToSave.split("/");
		newTrain.setRelationName(aux[aux.length-1].split("\\.")[0]);
		FileWriter f = new FileWriter(arffToSave);
		f.write(newTrain.toString());
		f.close();
		System.out.println("\nArff-a atributuak kenduta ondo gorde da hemen: " + arffToSave);
	}

}
