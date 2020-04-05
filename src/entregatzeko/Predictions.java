package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

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
		
		sailkatuDatuak(args[0], args[1], args[2]);
//		compareResults(args[0], args[1], args[2], args[3]);
		
	}
	
	public static void sailkatuDatuak(String csvPath, String modelPath, String dictionaryPath) throws Exception {
		
		Instances dataSet = GetRaw.datuGordinetikInstances(csvPath);
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		BayesNet classifier = (BayesNet) SerializationHelper.read(modelPath);
		
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(dictionaryPath));
		fixedDictionary.setInputFormat(dataSet);
		dataSet = Filter.useFilter(dataSet, fixedDictionary);
		dataSet.setClassIndex(0);
		
		String[] aux = modelPath.split("/");
		String direktorioa = modelPath.replace(aux[aux.length-1],"");
		String predictionsPath = direktorioa + "/allPredictions.txt";
		FileWriter f = new FileWriter(predictionsPath);
		
		for (int i = 0; i < dataSet.numInstances(); i++) {
			double predictedValue = classifier.classifyInstance(dataSet.instance(i));
			String predicted = dataSet.classAttribute().value((int) predictedValue);
			System.out.println(i + ".garren instantziaren estimazioa: " + predicted);
			f.write(i + ".garren instantziaren estimazioa: " + predicted + "\n");
		}
		
		f.close();
		
	}
	
	public static void compareResults(String arffPath, String realAffPath, String modelPath, String dictionaryPath) throws Exception {
		
		DataSource source = new DataSource(arffPath);
		Instances dataSet = source.getDataSet();
		if (dataSet.classIndex() == -1)
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		DataSource sourceReal = new DataSource(realAffPath);
		Instances test = sourceReal.getDataSet();
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes()-1);
		
		BayesNet classifier = (BayesNet) SerializationHelper.read(modelPath);
		
		FixedDictionaryStringToWordVector fixedDictionary = new FixedDictionaryStringToWordVector();
		fixedDictionary.setDictionaryFile(new File(dictionaryPath));
		fixedDictionary.setInputFormat(dataSet);
		dataSet = Filter.useFilter(dataSet, fixedDictionary);
		dataSet.setClassIndex(0);
		
		int guztiak = dataSet.numInstances();
		int asmatutakoak = 0;
		
		System.out.println("inst#	actual	predicted	error");
		System.out.println("-----	------	---------	-----");
		
		for (int i = 0; i < dataSet.numInstances(); i++) {
			double predictedValue = classifier.classifyInstance(dataSet.instance(i));
			String predicted = dataSet.classAttribute().value((int) predictedValue);
			String actual = test.classAttribute().value((int) test.instance(i).classValue());
			
			System.out.print(i + "	" + actual + "	" + predicted);
			if (!actual.equals(predicted))
				System.out.print("		+\n");
			else {
				asmatutakoak++;
				System.out.print("\n");
			}
		}

		System.out.println("-----------------------------");
		System.out.println("Asmatutakoak: " + asmatutakoak);
		double precision = (double) asmatutakoak/guztiak;

		System.out.println("Precision: " + precision);
		
	}

}
