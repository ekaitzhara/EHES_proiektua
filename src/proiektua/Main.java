package proiektua;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import mezuaToVector.AllCsvToArff;
import mezuaToVector.NonSparsetikSparsera;
import mezuaToVector.OptimizeTermFreq;
import mezuaToVector.StringToWord;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;


public class Main {
	
	public static void main(String[] args) throws Exception {
		
		
		File file = new File(args[0]); 
		BufferedReader br = new BufferedReader(new FileReader(file)); 
		String st;
		String[] csvGuztiak = new String[3];
		String[] arffGuztiak = new String[3];
		int i = 0;
		while ((st = br.readLine()) != null && !"-".equals(st)) {
			csvGuztiak[i] = st;
			i++;
		}
		i = 0;
		while ((st = br.readLine()) != null) {
			arffGuztiak[i] = st;
			i++;
		}
		
		// ARFF -> 0: test  1: test_unk  2: train
		
		// CSV guztiak ARFF formatura pasa
		AllCsvToArff.convertCsvToArff(csvGuztiak);
		
		// BOW sortu
		StringToWord.stringToWordVector(arffGuztiak[2]);
		
		// NonSparseToSparse
		Instances train = NonSparsetikSparsera.nonSparseToSparse(arffGuztiak[2]);
		
		DataSource sourceTest = new DataSource(arffGuztiak[0]);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setInputFormat(test);
		fixedDictionary.setDictionaryFile(file);
		
	} 
		

}
