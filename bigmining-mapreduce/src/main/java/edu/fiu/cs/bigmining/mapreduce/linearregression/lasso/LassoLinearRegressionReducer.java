package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

public class LassoLinearRegressionReducer extends
    Reducer<IntWritable, PairWritable, NullWritable, NullWritable> {

  private double learningRate;
  private double regularizationRate;
  private int featureDimension;

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
    this.featureDimension = conf.getInt("feature.dimension", 0);

    this.countPositive = 0;
    this.countNegative = 0;

    this.globalUpdatesPositive = new DenseVector(this.featureDimension);
    this.globalUpdatesNegative = new DenseVector(this.featureDimension);

    // load model
    try {
      model = new LinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Aggregate the positive and negative weight updates.
   */
  public void reduce(IntWritable key, Iterable<PairWritable> vecList, Context context) {
    Iterator<PairWritable> itr = vecList.iterator();
    context.getCounter("c", "non").increment(1);
    if (key.get() == 0) {
      context.getCounter("c", "negative").increment(1);
    }
    else {
      context.getCounter("c", "positive").increment(1);
    }
    
    if (key.get() == 0) { // receive negative updates
      while (itr.hasNext()) {
        PairWritable pair = itr.next();
        if (this.globalUpdatesNegative == null) {
          this.globalUpdatesNegative = pair.getValue().get();
        } else {
          this.globalUpdatesNegative = this.globalUpdatesNegative.plus(pair.getValue().get());
        }
        this.countNegative += pair.getKey().get();
      }
    }
    else { // receive positive updates
      while (itr.hasNext()) {
        PairWritable pair = itr.next();
        if (this.globalUpdatesPositive == null) {
          this.globalUpdatesPositive = pair.getValue().get();
        }
        else {
          this.globalUpdatesPositive = this.globalUpdatesPositive.plus(pair.getValue().get());
        }
        this.countPositive += pair.getKey().get();
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
