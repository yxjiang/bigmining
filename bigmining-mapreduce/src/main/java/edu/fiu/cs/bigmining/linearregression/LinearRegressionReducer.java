package edu.fiu.cs.bigmining.linearregression;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class LinearRegressionReducer extends
    Reducer<LongWritable, VectorWritable, NullWritable, NullWritable> {
  
  private String modelPath;
  
  private LinearRegressionModel model;
  
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.modelPath = conf.get("model.path");
    
    // load model
    try {
      model = new LinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  public void reduce(LongWritable key, Iterable<VectorWritable> vecList, Context context) {
    int count = 0;
    
    Vector globalUpdates = null;
    Iterator<VectorWritable> itr = vecList.iterator();
    
    while (itr.hasNext()) {
      if (globalUpdates == null) {
        globalUpdates = itr.next().get();
      }
      else {
        globalUpdates.plus(itr.next().get());
      }
      ++count;
    }
    
    globalUpdates.divide(count);
    model.updateWeights(globalUpdates);
  }
  
  /**
   * Write the model to specified location.
   */
  public void cleanup(Context context) {
    try {
      model.writeToFile(modelPath, context.getConfiguration());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
}