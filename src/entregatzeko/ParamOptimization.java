package entregatzeko;

import java.io.FileWriter;

import optimizing.BayesNetObject;
import optimizing.BayesNetParamOpt;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ParamOptimization {
	
	
	// Parametroak  hautatzeko  irizpidea  klase  minoritarioarekiko f-measure-a da
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		parametroakOptimizatu(args[0]);
	}

	public static void parametroakOptimizatu(String arffPath) throws Exception {
		
		BayesNetObject paramsOpt = BayesNetParamOpt.optimizatuParametroak(arffPath);
		
		String[] aux = arffPath.split("/");
		String direktorioa = arffPath.replace(aux[aux.length-1],"");
		FileWriter f = new FileWriter(direktorioa + "BayesNetParamsOpt.txt");
		f.write(paramsOpt.toString());
		f.close();
		
		System.out.println(paramsOpt.toString());
	}
	
}
