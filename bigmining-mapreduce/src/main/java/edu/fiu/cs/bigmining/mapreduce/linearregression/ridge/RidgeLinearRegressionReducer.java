package edu.fiu.cs.bigmining.mapreduce.linearregression.ridge;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;

import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;


public class RidgeLinearRegressionReducer extends
    Reducer<NullWritable, PairWritable, NullWritable, NullWritable> {
  
  private String modelPath;
  
  private RidgeLinearRegressionModel model;
  
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.modelPath = conf.get("model.path");
    
    // load model
    try {
      model = new RidgeLinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  public void reduce(NullWritable key, Iterable<PairWritable> vecList, Context context) {
    long count = 0;
    
    Vector globalUpdates = null;
    Iterator<PairWritable> itr = vecList.iterator();
    
    while (itr.hasNext()) {
      PairWritable pair = itr.next();
      if (globalUpdates == null) {
        globalUpdates = pair.getValue().get();
      }
      else {
        globalUpdates.plus(pair.getValue().get());
      }
      count += pair.getKey().get();
    }
    
    model.updateAllWeights(globalUpdates.divide(count));
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