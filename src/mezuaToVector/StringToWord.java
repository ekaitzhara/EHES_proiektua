package mezuaToVector;

import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class StringToWord {
	
	public static void main(String[] args) throws Exception {
		stringToWordVector(args[0]);
	}

	public static void stringToWordVector(String pathToArff) throws Exception {
		DataSource source = new DataSource(pathToArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes()-1);
		
		Integer hiztegiZabalera = Integer.MAX_VALUE;
		System.out.println(hiztegiZabalera);
		
		StringToWordVector stwv = new StringToWordVector();
		stwv.setAttributeIndices("first-last");
		stwv.setInputFormat(train);
		stwv.setWordsToKeep(hiztegiZabalera);
		stwv.setMinTermFreq(3);
		train = Filter.useFilter(train, stwv);
		train.setClassIndex(0);
		
		// Gorde trainBOW.arff
		String pathToSaveSTWV = pathToArff.split("\\.")[0];
		pathToSaveSTWV = pathToSaveSTWV + "BOW.arff";
		FileWriter f = new FileWriter(pathToSaveSTWV);
		f.write(train.toString());
		f.close();
		System.out.println("Train BOW gordeta hemen: " + pathToSaveSTWV);
		
	}

}
