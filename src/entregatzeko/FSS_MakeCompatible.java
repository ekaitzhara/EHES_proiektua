package entregatzeko;

import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class FSS_MakeCompatible {
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		make2ArffCompatible(args[0], args[1]);
	}
	
	public static void make2ArffCompatible(String trainArff, String toChangeArff) throws Exception {
		

		DataSource source = new DataSource(trainArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		
		DataSource sourceChange = new DataSource(toChangeArff);
		Instances toChange = sourceChange.getDataSet();
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(0);
		
		int index = 0;
		int[] indexToDelete = new int[toChange.numAttributes()-train.numAttributes()];
		System.out.println(indexToDelete.length);
		System.out.println(train.numAttributes());
		System.out.println(toChange.numAttributes());
		
		for (int i = 0; i < toChange.numAttributes(); i++) {
			boolean badago = false;
			for (int j = 0; j < train.numAttributes(); j++) {
				if (train.attribute(j).name().equals(toChange.attribute(i).name()))
					badago = true;
			}
			if (badago == false) {
				indexToDelete[index] = i;
				index++;
			}
		}
		
		Remove remove = new Remove();
		remove.setAttributeIndicesArray(indexToDelete);
		remove.setInvertSelection(false);
		remove.setInputFormat(toChange);
		Instances changed = Filter.useFilter(toChange, remove);
		
		System.out.println("\n" + changed.numAttributes());
		System.out.println(train.attribute(0).name() + " - " + changed.attribute(0).name());
		System.out.println(train.classIndex() + " - " + changed.classIndex());
		System.out.println(train.attribute(train.classIndex()).name() + " - " + changed.attribute(changed.classIndex()).name());
		
		
		String[] aux = toChangeArff.split("/");
		String fileName = aux[aux.length-1];
		changed.setRelationName(fileName.split("\\.")[0]);
		
		
		FileWriter f = new FileWriter(toChangeArff.replace(fileName, "") + "FSS/" + fileName.split("\\.")[0] + "_FSS.arff");
		f.write(changed.toString());
		f.close();
		
		System.out.println("\nAtributuak konpatible kenduta ondo gorde da hemen: " + toChangeArff.replace(fileName, "") + "FSS/" + fileName.split("\\.")[0] + "_FSS.arff");
	}

}
