package entregatzeko;

import java.io.FileWriter;

import optimizing.BayesNetObject;
import optimizing.BayesNetParamOpt;
import weka.classifiers.bayes.BayesNet;

/**
 * {@link BayesNet} algoritmorako parametro optimoak lortzeko klasea {@link Class}.
 * Gainera, modelo optimoa gorde dezake, eta baita itxarondako kalitatea.
 * 
 * @author ekaitzhara
 *
 */
public class ParamOptimization {
	
	
	// Parametroak  hautatzeko  irizpidea  klase  minoritarioarekiko f-measure-a da
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik getRaw aplikatu izana, horrela arff zuzena sortuta edukiko duzulako");
			System.out.println("	Erabaki non gordeko den modeloa eta zein izenarekin");
			System.out.println("	Sartu beharreko modeloaren path-a honelakoa izan behar da: /home/erabiltzaileIzena/workdir/Model/adibide.model");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Datu-sorta duen arff-a hainbat prozesu pasako ditu");
			System.out.println("	Prozesu horiek, arff batetik, train eta dev berain artean konpatible direnak sortuko ditu");
			System.out.println("	Train-arekin modeloa entrenatu egingo da, eta dv-arekin ebaluatu");
			System.out.println("	Hauek erabilita klasifikatzailearen parametro optimoenak lortuko ditu");
			System.out.println("	Parametro optimoenak edukita modeloa optimoa sortuko da parametro horiek erabiliz");
			System.out.println("	Erabaki duzun kokalekuan sortuko da modeloa");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Entrenamendu eta ebaluazioa burutzeko datuen arff-a");
			System.out.println("	2 -> Non gorde nahi duzun modelo optimoaren path-a");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar ParamOptimization.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/workdir/Model/adibideOpt.model\n");
			
			System.exit(0);
		}
		
		parametroakOptimizatu(args[0], args[1]);
	}

	/**
	 * Arff fitxategi bat, entrenamendu bezala erabilita, {@link BayesNet} algoritmorako parametro optimoak lortuko ditu.
	 * Parametro horiek erabilita, adierazita modelo optimoa gordeko du, eta bere itxarondako kalitatea.
	 * 
	 * @param arffPath
	 * @param modelPath
	 * @throws Exception
	 */
	public static void parametroakOptimizatu(String arffPath, String modelPath) throws Exception {
		
		// Erabakitako aukerak
		String errepresentazioa = "BOW";
		String bektoreMota = "NonSparse";
		
		BayesNetObject paramsOpt = BayesNetParamOpt.optimizatuParametroak(arffPath, errepresentazioa, bektoreMota);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		
		FileWriter f = new FileWriter(direktorioa + "BayesNetParamsOpt.txt");
		f.write(paramsOpt.toString());
		f.close();
		
		System.out.println(paramsOpt.toString());
		
		BayesNetParamOpt.modeloaGorde(arffPath, paramsOpt, modelPath, errepresentazioa, bektoreMota);
	}
	
}
