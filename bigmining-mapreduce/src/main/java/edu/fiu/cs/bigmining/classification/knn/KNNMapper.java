package edu.fiu.cs.bigmining.classification.knn;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

/**
 * Retrieve the local top-k nearest neighbor.
 *
 */
public class KNNMapper extends Mapper<LongWritable, VectorWritable, NullWritable, VectorWritable> {

  public void map(LongWritable key, VectorWritable vec, Context context) {
    
  }
}
