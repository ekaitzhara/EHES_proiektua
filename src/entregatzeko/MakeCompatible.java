package entregatzeko;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

public class MakeCompatible {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw aldatu nahi duzun datu-sortara aplikatu izana, horrela arff zuzena sortuta edukiko duelako");
			System.out.println("	Gainera, TranformRaw ere aplikatuta egon behar da, sortu den fitxategia lehenengo argumentua izango da");
			System.out.println("	Hiztegia sortuta egon behar du (TransformRaw-en sortzen da)");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Errepresentazioa duen (ad. BOW) fitxategiaren direktorio berean sortuko da arff berria");
			System.out.println("	Fitxategi berri hori, arff_BOW -ren atributu berdinak izango ditu, hau da, beren konpatiblea izango da");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> transformRaw-en sortutako arff-aren path-a (errepresentazioa duena)");
			System.out.println("	2 -> Aldatu nahi duzun arff fitxategiaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar makeCompatible.jar /home/erabiltzaileIzena/workdir/Transform/adibideBOW.arff /home/erabiltzaileIzena/workdir/ARFF/aldatuNahiDuguna.arff\n");
			System.exit(0);
		}
		
		try {
			makeCompatibleArffFile(args[0], args[1]);
		}catch (IOException e) {
			// TODO: handle exception
			System.out.println("Hiztegia ez dago sortuta, beraz, ez dago arff-a transformatuta");
		}
	}
	
	public static void makeCompatibleArffFile(String goodArff, String arffToChange) throws Exception {
		
		String[] aux1 = goodArff.split("/");
		String directory = goodArff.replace(aux1[aux1.length-1], "");
		
		String goodName = goodArff.split("_")[0];
		aux1 = arffToChange.split("/");
		String changeName = aux1[aux1.length-1].split("\\.")[0];
		
		String newArff = directory + changeName + goodArff.replace(goodName, "");
		System.out.println(newArff);
		
		String dictionaryPath = goodArff.split("\\.")[0] + "_dictionary.txt";
		
		DataSource sourceToChange = new DataSource(arffToChange);
		Instances toChange = sourceToChange.getDataSet();
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(toChange.numAttributes() - 1);
		
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(dictionaryPath));
		fixedDictionary.setInputFormat(toChange);
		toChange = Filter.useFilter(toChange, fixedDictionary);
		
		
		toChange.setRelationName(changeName + goodArff.replace(goodName, "").split("\\.")[0]);
		FileWriter f = new FileWriter(newArff);
		f.write(toChange.toString());
		f.close();
		
		System.out.println("Arff berria gordeta hemen: " + newArff);
		
	}
	
	public static Instances makeCompatibleInstances(Instances toChange, String dictionaryPath) throws Exception {
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(toChange.numAttributes() - 1);
		
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(dictionaryPath));
		fixedDictionary.setInputFormat(toChange);
		toChange = Filter.useFilter(toChange, fixedDictionary);
		
		toChange.setRelationName("dev_transformRaw_compatible");
		
		return toChange;
	}

}
