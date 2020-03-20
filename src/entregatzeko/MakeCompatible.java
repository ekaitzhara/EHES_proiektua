package entregatzeko;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

public class MakeCompatible {
	
	public static void main(String[] args) throws Exception {
		try {
			makeCompatibleArffFile(args[0], args[1]);
		}catch (IOException e) {
			// TODO: handle exception
			System.out.println("Hiztegia ez dago sortuta, beraz, ez dago arff-a transformatuta");
		}
	}
	
	public static void makeCompatibleArffFile(String goodArff, String arffToChange) throws Exception {
		
		
		String goodName = goodArff.split("_")[0];
		String newArff = arffToChange.split("\\.")[0] + goodArff.replace(goodName, "");
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
		
		
		FileWriter f = new FileWriter(newArff);
		f.write(toChange.toString());
		f.close();
		
		System.out.println("Arff berria gordeta hemen: " + newArff);
		
	}

}
