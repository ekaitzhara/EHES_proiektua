package mezuaToVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;

public class CsvToArff {

	/**
     * takes 2 arguments:
     * - CSV input file
     * - ARFF output file
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("\nUsage: CSV2Arff <input.csv> <output.arff>\n");
//            System.exit(1);
        }
        
        // `` guztiak eta '' batzuk kendu behar izan ditut
        FileWriter fw = new FileWriter(args[1]);
        BufferedReader br = new BufferedReader(new FileReader(args[0])); 
        String line;

        while((line = br.readLine()) != null) {
            line = line.replace("'","\\'");
            System.out.println(line);
            fw.write(line + "\n");
        }
        fw.close();

        
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[1]));
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
        
        
        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[2]));
        saver.setDestination(new File(args[2]));
        saver.writeBatch();
        
        System.out.println(args[2] + " fitxategia-a ondo sortua");


    }
    
}
