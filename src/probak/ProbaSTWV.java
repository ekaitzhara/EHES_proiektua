package probak;

import entregatzeko.TransformRaw;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class ProbaSTWV {
	
	public static void main(String[] args) throws Exception {
		String arffPath = args[0];
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		BayesNet classifier = new BayesNet();
			
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		String dictionaryPath = direktorioa + "/train_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt";
		
		StringToWordVector stwv = new StringToWordVector();
		
		stwv.setAttributeIndices("first-last");
		stwv.setWordsToKeep(10000000);
		stwv.setMinTermFreq(-1);
		stwv.setPeriodicPruning(100.0);
		stwv.setInputFormat(dataSet);
		Instances vectors = Filter.useFilter(dataSet, stwv);
		
		System.out.println(dataSet.firstInstance());
		System.out.println("How and did in leave then Russia");
		System.out.println(vectors.numAttributes());
		System.out.println(vectors.firstInstance());
		String[] a = vectors.firstInstance().toString().split(",");
		System.out.println(a.length);
		
	}

}
