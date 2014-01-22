package edu.fiu.cs.bigmining.mapreduce.linearregression;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class LinearRegressionReducer extends
    Reducer<NullWritable, VectorWritable, NullWritable, NullWritable> {
  
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
  
  public void reduce(NullWritable key, Iterable<VectorWritable> vecList, Context context) {
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
      FileSystem fs = FileSystem.get(context.getConfiguration());
      Path path = new Path(modelPath);
      if (fs.exists(path)) {
        fs.delete(path, true);
      }
      model.writeToFile(modelPath, context.getConfiguration());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
}