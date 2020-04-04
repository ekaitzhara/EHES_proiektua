package probak;

import entregatzeko.TransformRaw;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ProbaTransformRaw {
	
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
		
		Instances train_BOW = TransformRaw.transformRawInstances(dataSet, errepresentazioa, bektoreMota);
		
		System.out.println(train_BOW.numAttributes());
		System.out.println(train_BOW.firstInstance());
		String[] a = train_BOW.firstInstance().toString().split(",");
		System.out.println(a.length);
		
	}

}
