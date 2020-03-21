package entregatzeko;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FSS_MakeCompatible {
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			System.out.println("LAGUNTZA");
			
			System.exit(0);
		}
		
		make2ArffCompatible(args[0], args[1]);
	}
	
	public static void make2ArffCompatible(String trainArff, String toChangeArff) throws Exception {
		

		DataSource source = new DataSource(trainArff);
		Instances train = source.getDataSet();
		if (train.classIndex() == -1)
			train.setClassIndex(train.numAttributes() - 1);
		
		DataSource sourceChange = new DataSource(trainArff);
		Instances toChange = sourceChange.getDataSet();
		if (toChange.classIndex() == -1)
			toChange.setClassIndex(0);
		
		
	}

}
