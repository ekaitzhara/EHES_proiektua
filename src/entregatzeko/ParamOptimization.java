package entregatzeko;

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
		
		System.out.println(paramsOpt.toString());
	}
	
}
