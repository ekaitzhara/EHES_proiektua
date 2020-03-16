package mezuaToVector;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class OptimizeTermFreq {
	
	public static void main(String[] args) throws Exception {
		getBestMinTermFreq(args[0]);
	}

	public static int getBestMinTermFreq(String pathToArff) throws Exception {
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
		
		Instances train2 = null;
		double pctCorrect = -1.0;
		int bestMinTermFreq = -1;
		
		IBk classifier = new IBk();
		Evaluation evaluator = null;
		
		for (int i = 2; i < 4; i++) {
			stwv.setInputFormat(train);
			stwv.setMinTermFreq(i);
			
			train2 = Filter.useFilter(train, stwv);
			train2.setClassIndex(0);
			
			classifier.buildClassifier(train2);
			evaluator = new Evaluation(train2);
			evaluator.crossValidateModel(classifier, train2, 10, new Random(1));
			
			System.out.print(i + " - " + train2.numAttributes() + " - ");
			System.out.print(evaluator.pctCorrect() + "\n");
			
			if (pctCorrect < evaluator.pctCorrect()) {
				pctCorrect = evaluator.pctCorrect();
				bestMinTermFreq = i;
			}
				
		}
		System.out.println("--------------");
		stwv.setInputFormat(train);
		stwv.setMinTermFreq(bestMinTermFreq);
		
		train2 = Filter.useFilter(train, stwv);
		train2.setClassIndex(0);
		for (int i = 0; i < 20; i++) {
			System.out.println(train2.instance(i));
		}
		
		// nonSparse -> sparse
		// https://weka.sourceforge.io/doc.dev/weka/filters/unsupervised/instance/NonSparseToSparse.html
		
		
		// Test -> fixedDictionaryStringToWordVector
		// https://weka.sourceforge.io/doc.dev/weka/filters/unsupervised/attribute/FixedDictionaryStringToWordVector.html
		
//		4.3 atalean dagoena
//		txostena
//		proiektuko kodea
//		GrAL txostena 

		// Bayes Network
		
		
		classifier.buildClassifier(train2);
		evaluator = new Evaluation(train2);
		evaluator.crossValidateModel(classifier, train2, 10, new Random(1));
		System.out.println("Correct => " + evaluator.pctCorrect());
		
		System.out.println("Best MinTermFreq => " + bestMinTermFreq);
		System.out.println("Atributu kopurua => " + train2.numAttributes());
		
		return bestMinTermFreq;
		
		// /home/ekaitzhara/Documentos/Libraries/weka-3-8-4-azul-zulu-linux/weka-3-8-4/data/ReutersGrain-train.arff
		// horrekin probatu
	}

}
