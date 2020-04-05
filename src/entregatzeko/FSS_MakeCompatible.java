package entregatzeko;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

public class FSS_MakeCompatible {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik FSSInfoGain aplikatu izana, horrela FSS arff zuzena sortuta edukiko duzulako");
			System.out.println("	Gainera, TransformRaw ere aplikatuta egon behar da, sortu den fitxategia bigarren argumentua izango da");
			System.out.println("	Aurrekoa egikarituta, bere hiztegia sortu egin da");
			System.out.println("	Beste aukera bat gordeHiztegia metodoarekin nahi duzun fitxategiaren hiztegia gorde");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Lehen argumentua dagoen direktorio berdinean sortuko da fitxategi bateratua");
			System.out.println("	Fitxategi berri hori bigarren argumentuko hiztegiaren araberako atributu kopurua izango ditu");
			System.out.println("	Horrela, bi datu-sortak konpatibleak egingo ditugu, atributu berdinak edukita");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Bateragarri egin nahi duzun fitxategiaren path-a (GetRaw-ean sortu dena, filtro barik)");
			System.out.println("	2 -> Aldatzeko beharko dugun hiztegiaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar FSSMakeCompatible.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/nonDago/dictionary.txt\n");

			System.exit(0);
		}
		
		make2ArffCompatible(args[0], args[1]);
	}
	
	public static void make2ArffCompatible(String toChangeArff, String dictionaryPath) throws Exception {
		
		System.out.println(toChangeArff + "fitxategia bateragarri egingo dugu.");
		System.out.println(dictionaryPath + " hiztegia erabiliko dugu hori lortzeko.");
		
		DataSource sourceChange = new DataSource(toChangeArff);
		Instances toChange = sourceChange.getDataSet();
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(0);
		
		Instances changed = FSS_MakeCompatible.makeFSSCompatibleInstances(toChange, dictionaryPath);
		
		String[] aux = toChangeArff.split("/");
		String fileName = aux[aux.length-1];
		
		FileWriter f = new FileWriter(toChangeArff.replace(fileName, "") + fileName.split("\\.")[0] + "_compatible.arff");
		f.write(changed.toString());
		f.close();
		
		System.out.println("\nAtributuak bateragarri eginda ondo gorde da hemen: "
				+ "\n" + toChangeArff.replace(fileName, "") + fileName.split("\\.")[0] + "_compatible.arff");
		
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
	
	public static Instances makeFSSCompatibleInstances(Instances toChange, String dictionaryPath) throws Exception {
		
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(toChange.numAttributes()-1);
		
		String relationName = toChange.relationName();
		
		// Lehenik sortu dugun hiztegiko atributuekin bategaragarri egin
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(dictionaryPath));
		fixedDictionary.setInputFormat(toChange);
		toChange = Filter.useFilter(toChange, fixedDictionary);
		
		// Lehenik StringToWordVector aplikatu dugunez train-ari, klasea 0. posizioan geratu egin da
		// Beraz, arff honen klasea posizio berean jarri behar dugu, atributuak guztiz berdinak izateko
		Reorder reorderFilter = new Reorder();
		reorderFilter.setAttributeIndices("2-last,first");
		reorderFilter.setInputFormat(toChange);
		toChange = Filter.useFilter(toChange, reorderFilter);
		
		toChange.setRelationName(relationName + "_compatible");
		return toChange;
		
	}
	
	public static void gordeHiztegia(Instances dataSet, String dictionaryPath) throws IOException {
		
		FileWriter f = new FileWriter(dictionaryPath);
		// Atributu guztiak lortuko ditugu
		Enumeration<Attribute> allAttributes = dataSet.enumerateAttributes();
		
		while (allAttributes.hasMoreElements())
			f.write(allAttributes.nextElement().name() + ",1\n");
		
		f.close();
		
	}

}
