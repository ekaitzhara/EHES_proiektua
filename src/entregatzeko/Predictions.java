package entregatzeko;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.NumericTransform;

public class Predictions {
	
	public static void main(String[] args) throws Exception {
		
//		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw estimaziorako datu-sortarako aplikatu izana, horrela arff zuzena sortuta edukiko duzulako");
			System.out.println("	Gainera ParamOptimization aplikatu izana, modelo optimoa sortuta edukitzeko");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Atzetik parametro optimoenak lortu egin dira, eta horiekin modeloa sortu");
			System.out.println("	Sartutako arff-a klaserik gabe dago, hau da, klase atributuan ez dauzka daturik => '?'");
			System.out.println("	Modelo optimoa erabilita instantzi guztiak estimatu egingo dira");
			System.out.println("	Estimazio guztiak fitxategi batean gordeko dira");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Estimaziorako arff-a");
			System.out.println("	2 -> Modelo optimoaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar predictions.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/workdir/Model/adibide.model\n");
			
			System.exit(0);
		}
		
		sailkatuDatuak(args[0], args[1]);
	}
	
	public static void sailkatuDatuak(String arffPath, String modelPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		BayesNet classifier = (BayesNet) SerializationHelper.read(modelPath);
		
		
		for (int i = 0; i < dataSet.numInstances(); i++) {
			double predictedValue = classifier.classifyInstance(dataSet.instance(i));
			String predicted = dataSet.classAttribute().value((int) predictedValue);
			System.out.println(i + ".garren instantziaren estimazioa: " + predicted);
		}
		
	}

}
