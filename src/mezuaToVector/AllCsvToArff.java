package mezuaToVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;

public class AllCsvToArff {
	
	
	public static void main(String[] args) throws Exception {
		
		convertCsvToArff(args);
	}

	public static void convertCsvToArff(String[] args) throws IOException, FileNotFoundException, Exception {
		for (int i = 0; i < args.length; i++) {	// Args-ean aldatu nahi ditugun csv guztiak jartzen ditugu
			
			// Arff-ren izena lortzeko
			String pathCsv = args[i];
			String pathArff = pathCsv.split("\\.")[0];
			String pathCsvAux = pathArff + "2.csv";
			pathArff = pathArff + ".arff";
			
			System.out.println(pathCsv + " fitxategia aldatzen");
			
			FileWriter fw = new FileWriter(pathCsvAux);
	        BufferedReader br = new BufferedReader(new FileReader(pathCsv)); 
	        String line;
	        while((line = br.readLine()) != null) {
	            line = line.replace("'","`");
	            fw.write(line + "\n");
	        }
	        fw.close();
	
	        
	        // load CSV
	        CSVLoader loader = new CSVLoader();
	        loader.setSource(new File(pathCsvAux));
	        Instances data = loader.getDataSet();
	        
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
	        
	        
	        // save ARFF
	        ArffSaver saver = new ArffSaver();
	        saver.setInstances(data);
	        saver.setFile(new File(pathArff));
	        saver.setDestination(new File(pathArff));
	        saver.writeBatch();
	        System.out.println(pathArff + " fitxategia-a ondo sortua");
			System.out.println("------------------");
			
	        File csv2 = new File(pathCsvAux);
	        csv2.delete();
	        
		}
	}

}
