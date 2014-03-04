package edu.fiu.cs.bigmining.mapreduce.NMF.mapreduce;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

public class MM2PhaseTwoMapper extends
		Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

	@Override
	protected void map(IntWritable key, VectorWritable value, Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		context.write(key, value);
	}
	
}
