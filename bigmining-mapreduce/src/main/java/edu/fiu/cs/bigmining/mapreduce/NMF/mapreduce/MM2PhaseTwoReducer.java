package edu.fiu.cs.bigmining.mapreduce.NMF.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class MM2PhaseTwoReducer extends
		Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

	private Vector rowSum = null;

	@Override
	protected void reduce(IntWritable key, Iterable<VectorWritable> values,
			Context context) throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		for(VectorWritable value : values){
			if(rowSum == null){
				rowSum = value.get();
			}else{
				rowSum = rowSum.plus(value.get());
			}
		}
		
		context.write(key, new VectorWritable(rowSum));
	}

}
