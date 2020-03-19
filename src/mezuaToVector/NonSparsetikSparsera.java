package mezuaToVector;

import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class NonSparsetikSparsera {
	
	public static void main(String[] args) throws Exception {
		nonSparseToSparse(args[0]);
	}
	
	public static Instances nonSparseToSparse(String pathArff) throws Exception {
		DataSource source = new DataSource(pathArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		int hiztegiZabalera = train.attributeStats(0).distinctCount;
		
		StringToWordVector stwv = new StringToWordVector();
		stwv.setAttributeIndices("first-last");
		stwv.setInputFormat(train);
		stwv.setWordsToKeep(hiztegiZabalera);
		stwv.setMinTermFreq(3);
		
		train = Filter.useFilter(train, stwv);
		train.setClassIndex(0);
		
		
		
		System.out.println("NonSparse instantzia = > " + train.firstInstance());
//		System.out.println(train.instance(2));
		SparseToNonSparse nsts = new SparseToNonSparse();
		nsts.setInputFormat(train);
		train = Filter.useFilter(train, nsts);
		
		System.out.println("Sparse instantzia = > " + train.firstInstance());
//		System.out.println(train.instance(2));
		
		// Gorde trainBOWSparse.arff
		String pathToSaveSTWV = pathArff.split("\\.")[0];
		pathToSaveSTWV = pathToSaveSTWV + "BOWSparse.arff";
		FileWriter f = new FileWriter(pathToSaveSTWV);
		f.write(train.toString());
		f.close();
		
		return train;
	}

}
