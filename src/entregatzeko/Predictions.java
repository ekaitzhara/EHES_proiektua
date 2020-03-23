package entregatzeko;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NumericTransform;

public class Predictions {
	
	public static void main(String[] args) throws Exception {
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		sailkatuDatuak(args[0], args[1]);
	}
	
	public static void sailkatuDatuak(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(0);	// ???????????????????????
		
		
		
		Classifier classifier = (Classifier) SerializationHelper.read(modelPath);
		
		
		for (int i = 0; i < dataSet.numInstances(); i++) {
			double predictedValue = classifier.classifyInstance(dataSet.instance(i));
			String predicted = dataSet.classAttribute().value((int) predictedValue);
			System.out.println(i + ".garreb instantziaren estimazioa: " + predicted);
		}
		
	}

}
