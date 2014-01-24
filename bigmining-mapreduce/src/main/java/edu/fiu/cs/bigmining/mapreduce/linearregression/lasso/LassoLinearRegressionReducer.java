package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

public class LassoLinearRegressionReducer extends
    Reducer<BooleanWritable, PairWritable, NullWritable, NullWritable> {

  private double learningRate;
  private double regularizationRate;

  private String modelPath;

  private long countPositive;
  private long countNegative;

  private Vector globalUpdatesPositive;
  private Vector globalUpdatesNegative;

  private LinearRegressionModel model;

  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.modelPath = conf.get("model.path");
    this.learningRate = Double.parseDouble(conf.get("learning.rate"));
    this.regularizationRate = Double.parseDouble(conf.get("regularization.rate"));

    this.countPositive = 0;
    this.countNegative = 0;

    this.globalUpdatesPositive = null;
    this.globalUpdatesNegative = null;

    // load model
    try {
      model = new LinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void reduce(BooleanWritable key, Iterable<PairWritable> vecList, Context context) {
    if (key.get()) { // receive negative updates
      Iterator<PairWritable> itr = vecList.iterator();
      while (itr.hasNext()) {
        PairWritable pair = itr.next();
        if (this.globalUpdatesNegative == null) {
          this.globalUpdatesNegative = pair.getValue().get();
          this.countNegative += pair.getKey().get();
        } else {
          this.globalUpdatesPositive = pair.getValue().get();
          this.countPositive += pair.getKey().get();
        }
      }
    }
  }

  /**
   * Write the model to specified location.
   */
  public void cleanup(Context context) {
    this.globalUpdatesPositive = this.globalUpdatesPositive.divide(countPositive);
    this.globalUpdatesNegative = this.globalUpdatesNegative.divide(countNegative);

    // find and update the feature with most negative value
    int candidateIndex = -1;
    double minDifferential = Double.MAX_VALUE;
    for (int i = 0; i < this.globalUpdatesPositive.size(); ++i) {
      double differential = Math.min(this.globalUpdatesPositive.get(i),
          this.globalUpdatesNegative.get(i));
      if (differential < 0 && differential < minDifferential) {
        minDifferential = differential;
        candidateIndex = i;
      }
    }

    if (candidateIndex == -1) { // no further update is needed
      return;
    }

    Vector weights = model.getFeatureWeights();

    double delta = -this.learningRate
        * (minDifferential + weights.get(candidateIndex) >= 0 ? regularizationRate
            : -regularizationRate);
    
    model.updateWeight(candidateIndex, delta);
    try {
      model.writeToFile(modelPath, context.getConfiguration());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
