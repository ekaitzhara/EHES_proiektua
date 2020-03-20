package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class TransformRaw {
	
	public static void main(String[] args) throws Exception {
		transFormRawMetodoa(args[0], args[1], args[2]);
	}

	// Errepresentazioa --> BOW ala TFIDF
	// String bektoreMota --> NonSparse ala Sparse
	public static void transFormRawMetodoa(String pathArff, String errepresentazioa, String bektoreMota) throws Exception {
		
		String dataName = pathArff.split("\\.")[0];
		String newArff = dataName + "_" + errepresentazioa + "_" + bektoreMota + ".arff";
		
		DataSource source = new DataSource(pathArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes()-1);
		
		System.out.println("1" + train.firstInstance());
		
		Integer hiztegiZabalera = Integer.MAX_VALUE;
		System.out.println(hiztegiZabalera);
		
		StringToWordVector stwv = new StringToWordVector();
		stwv.setAttributeIndices("first-last");
		stwv.setInputFormat(train);
		stwv.setWordsToKeep(hiztegiZabalera);
		stwv.setMinTermFreq(3);
		if ("TFIDF".equals(errepresentazioa)) {
			stwv.setTFTransform(true);
			stwv.setIDFTransform(true);
		}
		
		// Gorde dictionary
		stwv.setDictionaryFileToSaveTo(new File(dataName + "_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt"));
		stwv.setPeriodicPruning(100.0);
		
		train = Filter.useFilter(train, stwv);
		train.setClassIndex(0);
//		System.out.println(train.numAttributes());
		System.out.println("2" + train.firstInstance());
		
		if ("Sparse".equals(bektoreMota)) { 
			// NonSparsetik Sparsera 
			SparseToNonSparse nsts = new SparseToNonSparse();
			nsts.setInputFormat(train);
			train = Filter.useFilter(train, nsts);
		}
		
		FileWriter f = new FileWriter(newArff);
		f.write(train.toString());
		f.close();
		
		System.out.println("Train " + errepresentazioa + " eta " + bektoreMota + " gordeta hemen: " + newArff);
		
	}
}
