package entregatzeko;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;

/**
 * Datu gordinetik Wekak irakur dezakeen datu formatu batera bihurtzeko {@link Class}.
 * 
 * @author ekaitzhara
 *
 */
public class GetRaw {

	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if (args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Fitxategi batean (csv, txt...) datu zuzenak eduki");
			System.out.println("	Datu zuzenak -(hiru zutabe lerroko)-> id, mezua, klasea || Dokumentu bakoitza instantzi bat izango da\n");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Adierazi duzun direktorioan gordeko da .arff fitxategia, Wekak irakurri dezakeen formatu zuzena");
			System.out.println("	mezua String motako atributu bakar batekin karakterizatuko da, eta klasea, atributu nominal bat bezala\n");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> Datuak dauzkan fitxategiaren kokalekua (txt, csv...)");
			System.out.println("	2 -> Arff fitxategia non eta zein izenarekin gorde nahi duzun kokalekua\n");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar getRaw.jar /home/erabiltzaileIzena/nonDagoCSV/adibidea.csv /home/erabiltzaileIzena/nonGordeNahi/adibidea.arff\n");
			
			
			System.exit(0);
		}
		
		datuGordinetikArff(args[0], args[1]);
	}
	
	/**
	 * Datu gordinetik arff fitxategia sortu egiten du.
	 * 
	 * @param pathToData
	 * @param pathArff
	 * @throws Exception
	 */
	public static void datuGordinetikArff(String pathToData, String pathArff) throws Exception {
		
		Instances data = GetRaw.datuGordinetikInstances(pathToData);
        
		// Arff karpeta ez badago sortuta
		String[] aux = pathArff.split("/");
		String modelName = aux[aux.length-1];
        File modelDirectory = new File(pathArff.replace(modelName, ""));
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();        
        
        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(pathArff));
        saver.setDestination(new File(pathArff));
        saver.writeBatch();
        System.out.println(pathArff + " fitxategia-a ondo sortua");
			        
	}
	
	/**
	 * Datu gordinetik, Wekako {@link Instances} objektua sortu egiten du.
	 * 
	 * @param pathToData
	 * @return
	 * @throws Exception
	 */
	public static Instances datuGordinetikInstances(String pathToData) throws Exception {
		
		// Argumentuetaik datuak lortu
		String dataName = pathToData.split("\\.")[0];
		String fileType = pathToData.split("\\.")[1];
		String pathFileAux = dataName + "2." + fileType;
		
		System.out.println(pathToData + " fitxategia irakurtzen...");
		
		// Arazoak ematen dituzten karaktereak kendu
		FileWriter fw = new FileWriter(pathFileAux);
        BufferedReader br = new BufferedReader(new FileReader(pathToData)); 
        String line;
        while((line = br.readLine()) != null) {
            line = line.replace("'","`");
            fw.write(line + "\n");
        }
        fw.close();
        
        Instances data = null;
        // Gure datuak csv moduan daude
        if ("csv".equals(fileType)) {
		    // load CSV
		    CSVLoader loader = new CSVLoader();
		    loader.setSource(new File(pathFileAux));
		    data = loader.getDataSet();
        }
        
        // Erlazio izena oso luze ez geratzeko, gerorako gordeko dugu
        String relationName = data.relationName().substring(0, data.relationName().length()-1);
        
        // Leheneko zutabea identifikatzailea da, ez digu balio
        Remove r = new Remove();
	    int[] indice = {0};
	    r.setAttributeIndicesArray(indice);
	    r.setInvertSelection(false);
	    r.setInputFormat(data);
	    data = Filter.useFilter(data, r);
	    
	    // Lehen atributua mezua eramaten duena da, beraz, string motakoa izan behar da
	    NominalToString filterString = new NominalToString();
		filterString.setAttributeIndexes("first");
		filterString.setInputFormat(data);
		data = Filter.useFilter(data, filterString);
        if(data.classIndex() == -1)
        	data.setClassIndex(data.numAttributes() - 1);
        
        // Arazoak ez edukitzeko klaseko izenarekin aldatu egingo dugu
        RenameAttribute rename = new RenameAttribute();
        rename.setAttributeIndices("last");
        rename.setReplace("klasea");
        rename.setInputFormat(data);
        data = Filter.useFilter(data, rename);
        
        // Klasearen datua ez badauka, numeric atributu bezala sortuko du
        // Beraz, guk nominal bezala sortu beharko dugu
        if (data.numClasses() == 1) {
        	NumericToNominal numToNom = new NumericToNominal();
        	numToNom.setAttributeIndices("last");
        	numToNom.setInputFormat(data);
        	data = Filter.useFilter(data, numToNom);
        	
        	Remove remove = new Remove();
        	remove.setAttributeIndices("last");
        	remove.setInputFormat(data);
        	data = Filter.useFilter(data, remove);
        	
        	Add addNominal = new Add();
        	addNominal.setAttributeIndex("last");
        	addNominal.setAttributeName("klasea");
        	addNominal.setNominalLabels("NUM,LOC,HUM,DESC,ENTY,ABBR");
        	addNominal.setInputFormat(data);
        	data = Filter.useFilter(data, addNominal);
        }
        
        // Behar izan dugun fitxategi osagarria ezabatu
        File csv2 = new File(pathFileAux);
        csv2.delete();
		
        data.setRelationName(relationName);
        System.out.println(pathToData + " fitxategitik Instances objektua lortuta.");
        
        return data;
	}
	
	/**
	 * Warning-ak kentzeko.
	 */
	public static void disableWarning() {
		System.err.close();
		System.setErr(System.out);
	}
}
