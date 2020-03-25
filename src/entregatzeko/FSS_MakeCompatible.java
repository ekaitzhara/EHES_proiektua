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
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik fssInfoGain aplikatu izana, horrela FSS arff zuzena sortuta edukiko duzulako");
			System.out.println("	Gainera, transformRaw ere aplikatuta egon behar da, sortu den fitxategia bigarren argumentua izango da");
			System.out.println("	Bigarren arff hori errepresentazioa edukiko beharko du (ad. BOW) eta hori makeCompatible edo transformRaw-rekin lor daiteke");
			System.out.println("Ondorengo balditza:");
			System.out.println("	FSS direktorioan sortuko da FSS arff berria, bigarren argumentuko datuetatik abiatuta");
			System.out.println("	FSS arff berri hori lehenengo argumentuko arff-aren atributu kopuru berdina izango ditu");
			System.out.println("	Horrela, bi datu-sortak konpatibleak egingo ditugu, atributu berdinak edukita");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> fssInfoGain-en sortutako arff-aren path-a (atributuen hautapena egikarituta duena)");
			System.out.println("	2 -> Aldatu nahi duzun arff fitxategiaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar fssMakeCompatible.jar /home/erabiltzaileIzena/workdir/Transform/FSS/adibideFSS.arff /home/erabiltzaileIzena/workdir/Transform/adibideBOW.arff\n");

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
	
	public static Instances make2InstancesCompatibles(Instances good, Instances toChange) throws Exception {
		if (good.classIndex() == -1)
			good.setClassIndex(good.numAttributes() - 1);
		
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(0);
		String relationName = toChange.relationName();
		
		int index = 0;
		int[] indexToDelete = new int[toChange.numAttributes()-good.numAttributes()];
		
		for (int i = 0; i < toChange.numAttributes(); i++) {
			boolean badago = false;
			for (int j = 0; j < good.numAttributes(); j++) {
				if (good.attribute(j).name().equals(toChange.attribute(i).name()))
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
		changed.setRelationName(relationName + "_makeCompatibleFSS");
		
		return changed;
	}

}
