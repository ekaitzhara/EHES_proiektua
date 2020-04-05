package entregatzeko;

import java.io.File;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class TransformRaw {
	
	public static void main(String[] args) throws Exception {
		
		GetRaw.disableWarning();
		
		if(args.length == 0) {
			System.out.println("=== PROGRAMAREN FUNTZIONAMENDURAKO LAGUNTZA ===\n");
			System.out.println("Aurrebaldintza:");
			System.out.println("	Lehenik GetRaw aplikatu izana, horrela arff zuzena sortuta edukiko duelako");
			System.out.println("	Hemendik aurrera sortuko diren fitxategi aldatuak gordeko diren direktorioa erabaki (bakarrik erabaki, programak sortuko du)");
			System.out.println("	Errepresentazioa erabaki. Hiru aukera daude: BOW, TF edo TFIDF");
			System.out.println("	Bektore mota erabaki. Bi aukera daude: NonSparse edo Sparse\n");
			System.out.println("Ondorengo balditza:");
			System.out.println("	Zehaztu dugun direktorioan, aukeratutako erreprensentazio eta bektore motako arff-a sortuko da");
			System.out.println("	Adibidez, BOW eta Spase aukeratu baduzu, filtro horiek dituen datuz osatutako arff-a sortuko da\n");
			System.out.println("Argumentuen zerrenda eta deskribapena:");
			System.out.println("	1 -> getRaw-en sortutako arff-aren path-a");
			System.out.println("	2 -> Gordeko den direktorioa (ARFF karpetaren leku berdinean sortu)");
			System.out.println("	3 -> Errepresentazio mota: BOW, TF ala TFIDF");
			System.out.println("	4 -> Bektore mota: NonSparse ala Sparse");
			System.out.println("Adibide hau jarraitu:\n");
			System.out.println("		java -jar transformRaw.jar /home/erabiltzaileIzena/workdir/ARFF/adibide.arff /home/erabiltzaileIzena/workdir/Direktorioa BOW NonSparse\n");
			
			System.exit(0);
		}
		transFormRawMetodoa(args[0], args[1], args[2], args[3]);
	}

	// Errepresentazioa --> BOW, TF ala TFIDF
	// String bektoreMota --> NonSparse ala Sparse
	public static void transFormRawMetodoa(String pathArff, String direktorioa ,String errepresentazioa, String bektoreMota) throws Exception {
		
		System.out.println(pathArff + " fitxategia aukera hauekin eraldatuko da:");
		System.out.println("	- Errepresentazioa -> " + errepresentazioa);
		System.out.println("	- Bektore mota -> " + bektoreMota + "\n");
		// Direktorioa karpeta ez badago sortuta
        File modelDirectory = new File(direktorioa);
        if (!modelDirectory.exists())
        	modelDirectory.mkdir();
		
		String dataName = pathArff.split("\\.")[0];
		String[] aux = dataName.split("/");
		String fileName = aux[aux.length-1];
		String newArff = direktorioa + "/" + fileName + "_" + errepresentazioa + "_" + bektoreMota + ".arff";
		
		DataSource source = new DataSource(pathArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes()-1);
		
		train = TransformRaw.transformRawInstancesDictionary(train, errepresentazioa, bektoreMota, direktorioa, fileName);
		
		FileWriter f = new FileWriter(newArff);
		f.write(train.toString());
		f.close();
		
		System.out.println(fileName + " fitxategia erabakitako aukerekin gordeta hemen: "
				+ "\n	" + newArff);
		
	}
	
	public static Instances transformRawInstances(Instances dataSet, String errepresentazioa, String bektoreMota) throws Exception {
		
		// Hitz guztiak sar daitezen hiztegirako zabalera handia sartuko diogu 
		Integer hiztegiZabalera = Integer.MAX_VALUE;
		
		StringToWordVector stwv = new StringToWordVector();
		stwv.setWordsToKeep(hiztegiZabalera);
		stwv.setPeriodicPruning(100.0);
		stwv.setMinTermFreq(-1);
		stwv.setAttributeIndices("first-last");
		if ("TFIDF".equals(errepresentazioa)) {
			stwv.setTFTransform(true);
			stwv.setIDFTransform(true);
		} else if("TF".equals(errepresentazioa)) {
			stwv.setTFTransform(true);
		} else {
			stwv.setIDFTransform(false);
			stwv.setTFTransform(false);
		}
		stwv.setInputFormat(dataSet);
		
		dataSet = Filter.useFilter(dataSet, stwv);
		dataSet.setClassIndex(0);
		
		if ("Sparse".equals(bektoreMota)) { 
			// NonSparsetik Sparsera 
			SparseToNonSparse nsts = new SparseToNonSparse();
			nsts.setInputFormat(dataSet);
			dataSet = Filter.useFilter(dataSet, nsts);
		}
		
		dataSet.setRelationName("train_" + errepresentazioa + "_" + bektoreMota);
		
		return dataSet;
	}
	
	public static Instances transformRawInstancesDictionary(Instances dataSet, String errepresentazioa, String bektoreMota, String direktorioa, String fileName) throws Exception {
		
		// Hitz guztiak sar daitezen hiztegirako zabalera handia sartuko diogu 
		Integer hiztegiZabalera = Integer.MAX_VALUE;
		
		StringToWordVector stwv = new StringToWordVector();
		stwv.setWordsToKeep(hiztegiZabalera);
		stwv.setPeriodicPruning(100.0);
		stwv.setMinTermFreq(-1);
		stwv.setAttributeIndices("first-last");
		if ("TFIDF".equals(errepresentazioa)) {
			stwv.setTFTransform(true);
			stwv.setIDFTransform(true);
		} else if("TF".equals(errepresentazioa)) {
			stwv.setTFTransform(true);
		} else {
			stwv.setIDFTransform(false);
			stwv.setTFTransform(false);
		}
		
		// Gorde dictionary
		stwv.setDictionaryFileToSaveTo(new File(direktorioa + "/" + fileName + "_" + errepresentazioa + "_" + bektoreMota + "_dictionary.txt"));
		
		stwv.setInputFormat(dataSet);
		
		dataSet = Filter.useFilter(dataSet, stwv);
		dataSet.setClassIndex(0);
		
		if ("Sparse".equals(bektoreMota)) { 
			// NonSparsetik Sparsera 
			SparseToNonSparse nsts = new SparseToNonSparse();
			nsts.setInputFormat(dataSet);
			dataSet = Filter.useFilter(dataSet, nsts);
		}
		
		dataSet.setRelationName("train_" + errepresentazioa + "_" + bektoreMota);
		
		return dataSet;
	}
}
