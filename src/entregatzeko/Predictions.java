package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

/**
 * Estimazioa egiteko {@link Class}.
 * 
 * @author ekaitzhara
 *
 */
public class Predictions {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw estimaziorako datu-sortarako aplikatu izana, horrela arff zuzena sortuta edukiko duzulako");
			System.out.println("	Gainera ParamOptimization aplikatu izana, modelo optimoa sortuta edukitzeko");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Atzetik parametro optimoenak lortu egin dira, eta horiekin modeloa sortu");
			System.out.println("	Sartutako arff-a klaserik gabe dago, hau da, klase atributuan ez dauzka daturik => '?'");
			System.out.println("	Modelo optimoa erabilita instantzia guztiak estimatu egingo dira");
			System.out.println("	Estimazio guztiak fitxategi batean gordeko dira");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Estimaziorako arff-a");
			System.out.println("	2 -> Modelo optimoaren path-a");
			System.out.println("	3 -> Modelo optimoan sortu egin den hiztegiaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar predictions.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/workdir/Model/adibide.model /home/erabiltzaileIzena/workdir/ARFF/OPT_dictionary.txt\n");
			
			System.exit(0);
		}
		
		sailkatuDatuak(args[0], args[1], args[2]);
//		compareResults(args[0], args[1], args[2], args[3]);
		
	}
	
	/**
	 * Datu gordineko fitxategi bat, arff fitxategi batera eraldatuko da, eta {@link BayesNet}-eko modelo optimorako bateragarria egingo du.
	 * Hau, adierazitako hiztegiarekin egingo du. Gero, modelo honekin, fitxategi horren gainean estimazioak egingo dira eta gordeko dira.
	 * 
	 * @param csvPath
	 * @param modelPath
	 * @param dictionaryPath
	 * @throws Exception
	 */
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
	
	/**
	 * Estimazioak egiteko arff fitxategi bat edukita, eta estimazio horien emaitza errealak dituen beste fitxategia edukita.
	 * Modelo optimoa zenbat estimazio asmatzen dituen erakutsiko eta gordeko ditu.
	 * 
	 * @param arffPath
	 * @param realAffPath
	 * @param modelPath
	 * @param dictionaryPath
	 * @throws Exception
	 */
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
		
		String[] aux = modelPath.split("/");
		String direktorioa = modelPath.replace(aux[aux.length-1],"");
		String predictionsPath = direktorioa + "/allPredictions.txt";
		FileWriter f = new FileWriter(predictionsPath);
		
		System.out.println("inst#	actual	predicted	error");
		System.out.println("-----	------	---------	-----");
		f.write("inst#	actual	predicted	error\n");
		f.write("-----	------	---------	-----\n");
		
		for (int i = 0; i < dataSet.numInstances(); i++) {
			double predictedValue = classifier.classifyInstance(dataSet.instance(i));
			String predicted = dataSet.classAttribute().value((int) predictedValue);
			String actual = test.classAttribute().value((int) test.instance(i).classValue());
			
			System.out.print(i + "	" + actual + "	" + predicted);
			f.write(i + "	" + actual + "	" + predicted);
			if (!actual.equals(predicted)) {
				System.out.print("		+\n");
				f.write("		+\n");
			} else {
				asmatutakoak++;
				System.out.print("\n");
				f.write("\n");
			}
		}

		System.out.println("-----------------------------");
		System.out.println("Asmatutakoak: " + asmatutakoak);
		double precision = (double) asmatutakoak/guztiak;

		System.out.println("Precision: " + precision);
		
		f.write("\n-----------------------------");
		f.write("\nAsmatutakoak: " + asmatutakoak);
		f.write("\nPrecision: " + precision);
		
		f.close();
	}

}
