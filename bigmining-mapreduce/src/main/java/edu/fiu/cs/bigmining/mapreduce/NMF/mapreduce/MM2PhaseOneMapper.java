package edu.fiu.cs.bigmining.mapreduce.NMF.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.VectorWritable;

/**
 * There are two input sequence files. One is matrix A partitioned in column vector,
 * the other is B partitioned in row vector 
 * @author zhouwubai
 *
 */
public class MM2PhaseOneMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>{
	
	
	@Override
	protected void map(IntWritable key, VectorWritable value,Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		context.write(key, value);
	}
	
}
