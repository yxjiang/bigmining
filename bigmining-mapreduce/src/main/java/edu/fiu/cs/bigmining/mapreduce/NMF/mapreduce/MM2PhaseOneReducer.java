package edu.fiu.cs.bigmining.mapreduce.NMF.mapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VectorWritable;

/**
 * compute column vector of A times row vector of B
 * get a matrix
 * @author zhouwubai
 *
 */
public class MM2PhaseOneReducer extends
		Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

	Map<IntWritable,VectorWritable> pairs;
	Configuration conf;
	int rowNumOfA, colNumOfB;
	
	@Override
	protected void setup(Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		pairs = new HashMap<IntWritable, VectorWritable>();
		conf = context.getConfiguration();
		rowNumOfA = Integer.parseInt(conf.get("A.row.num"));
		colNumOfB = Integer.parseInt(conf.get("B.col.num"));
	}
	
	
	@Override
	protected void reduce(IntWritable key, Iterable<VectorWritable> values,
			Context context) throws IOException, InterruptedException {
		
		for(VectorWritable value : values){
			//first part arrives
			if(!pairs.containsKey(key)){
				pairs.put(key, value);
			}else{
				
				//compute multiplication of two vectors, partition it to VectorWritable again
				
			}
		}
		
	}
}
