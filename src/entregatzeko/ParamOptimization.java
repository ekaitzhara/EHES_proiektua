package entregatzeko;

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

	public static void parametroakOptimizatu(String trainPath) throws Exception {
		
		DataSource source = new DataSource(trainPath);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(0);
		
		
	}
	
}
