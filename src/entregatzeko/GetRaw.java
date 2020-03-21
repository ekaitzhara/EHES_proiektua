package entregatzeko;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;

public class GetRaw {

	public static void main(String[] args) throws Exception {
		
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
	
	public static void datuGordinetikArff(String pathToData, String pathArff) throws Exception {
		
		
		String dataName = pathToData.split("\\.")[0];
		String fileType = pathToData.split("\\.")[1];
		String pathFileAux = dataName + "2." + fileType;
		
		System.out.println(pathToData + " fitxategia aldatzen");
		
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
		    
		    Remove r = new Remove();
		    int[] indice = {0};
		    r.setAttributeIndicesArray(indice);
		    r.setInvertSelection(false);
		    r.setInputFormat(data);
		    data = Filter.useFilter(data, r);
		    
		    NominalToString filterString = new NominalToString();
			filterString.setAttributeIndexes("first");
			filterString.setInputFormat(data);
			data = Filter.useFilter(data, filterString);
			System.out.println("Arff-ko instantzia => " + data.instance(2));
        }
        
        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(pathArff));
        saver.setDestination(new File(pathArff));
        saver.writeBatch();
        System.out.println(pathArff + " fitxategia-a ondo sortua");
		System.out.println("------------------");
		
        File csv2 = new File(pathFileAux);
        csv2.delete();
			        
	}
}
