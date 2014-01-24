package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

public class LassoLinearRegressionReducer extends
    Reducer<NullWritable, PairWritable, NullWritable, NullWritable> {

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

  public void reduce(NullWritable key, Iterable<PairWritable> vecList, Context context) {

  }

  /**
   * Write the model to specified location.
   */
  public void cleanup(Context context) {

  }

}
