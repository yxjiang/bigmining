package edu.fiu.cs.bigmining.mapreduce.NMF.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.VectorWritable;


/**
 * mapper for calculating A transpose * A, reducer is same as
 * MM2PhaseTwoReducer
 * @author zhouwubai
 *
 */
public class MM3Mapper extends
		Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

	@Override
	protected void map(IntWritable key, VectorWritable value, Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		
		Matrix crossProd = value.get().cross(value.get());
		for(int i = 0; i < crossProd.rowSize(); i++){
			context.write(new IntWritable(i), new VectorWritable(crossProd.viewRow(i)));
		}
		
	}

}
