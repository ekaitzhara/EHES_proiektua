package proiektua;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import mezuaToVector.AllCsvToArff;
import mezuaToVector.NonSparsetikSparsera;
import mezuaToVector.OptimizeTermFreq;
import mezuaToVector.StringToWord;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
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
		while ((st = br.readLine()) != null && !"-".equals(st)) {
			arffGuztiak[i] = st;
			i++;
		}
		String pathDictionaryFile = br.readLine();
		
		// ARFF -> 0: test  1: test_unk  2: train
		
		// CSV guztiak ARFF formatura pasa
//		AllCsvToArff.convertCsvToArff(csvGuztiak);
		
		// BOW sortu
		StringToWord.stringToWordVector(arffGuztiak[2], pathDictionaryFile);
		
		// NonSparseToSparse
		Instances train = NonSparsetikSparsera.nonSparseToSparse(arffGuztiak[2]);
		
		DataSource sourceTest = new DataSource(arffGuztiak[0]);
		Instances test = sourceTest.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		System.out.println("\n" + test.firstInstance());
		
		// Fixed aplikatu test-ari train-aren berdina izateko
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(pathDictionaryFile));
		fixedDictionary.setInputFormat(test);
		test = Filter.useFilter(test, fixedDictionary);
		
		
		
		System.out.println(test.firstInstance());
		
		System.out.println("Train: " + train.numAttributes() + " eta test: " + test.numAttributes());
		
		NaiveBayes classifier = new NaiveBayes();
		Evaluation evaluator = new Evaluation(train);
		
		classifier.buildClassifier(train);
		evaluator.evaluateModel(classifier, test);
		System.out.println(evaluator.toSummaryString("\nSUMMARY", false));
		System.out.println(evaluator.toClassDetailsString());
		System.out.println(evaluator.toMatrixString());
		
	} 
		

}
