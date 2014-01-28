package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.math.Constants;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

public class LassoLinearRegressionReducer extends
    Reducer<NullWritable, PairWritable, NullWritable, NullWritable> {

  private static final double EPSILON = Constants.EPSILON * 100;
  
  private double learningRate;
  private double regularizationRate;
  private int featureDimension;

  private String modelPath;

  private long count;

  private Vector globalUpdatesPositive;
  private Vector globalUpdatesNegative;

  private LinearRegressionModel model;

  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.modelPath = conf.get("model.path");
    this.learningRate = Double.parseDouble(conf.get("learning.rate"));
    this.regularizationRate = Double.parseDouble(conf.get("regularization.rate"));
    this.featureDimension = conf.getInt("feature.dimension", 0);

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
    
    // aggregate all local vector into global vector
    while (itr.hasNext()) {
      PairWritable pair = itr.next();
      this.count += pair.getKey().get();
      if (this.globalUpdatesPositive == null) {
        this.globalUpdatesPositive = pair.getValue().get().clone();
        this.globalUpdatesNegative = pair.getValue().get().times(-1);
      }
      else {
        this.globalUpdatesPositive = this.globalUpdatesPositive.plus(pair.getValue().get());
        this.globalUpdatesPositive = this.globalUpdatesNegative.minus(pair.getValue().get());
      }
    }
    
  }

  /**
   * Write the model to specified location.
   */
  public void cleanup(Context context) {
    int candidateIndex = -1;
    double mostNegative = 0;
    
    this.globalUpdatesPositive = this.globalUpdatesPositive.divide(count);
    this.globalUpdatesNegative = this.globalUpdatesNegative.divide(count);
    
    for (int i = 0; i < this.globalUpdatesPositive.size(); ++i) {
      double weight = 0;
      if (i == 0) {
        weight = this.model.getBias();
      }
      else {
        weight = this.model.getFeatureWeight(i);
      }
      
      if (weight >= EPSILON) { // w_i is positive
        this.globalUpdatesPositive.set(i, this.globalUpdatesPositive.get(i) + this.regularizationRate);
        this.globalUpdatesNegative.set(i, this.globalUpdatesNegative.get(i) - this.regularizationRate);
      }
      else if (-EPSILON < weight && weight < EPSILON){ // w_i is 0
        this.globalUpdatesPositive.set(i, this.globalUpdatesPositive.get(i) + this.regularizationRate);
        this.globalUpdatesNegative.set(i, this.globalUpdatesNegative.get(i) + this.regularizationRate);
      }
      else { // w_i is negative
        this.globalUpdatesPositive.set(i, this.globalUpdatesPositive.get(i) - this.regularizationRate);
        this.globalUpdatesNegative.set(i, this.globalUpdatesNegative.get(i) + this.regularizationRate);
      }
      
      double negativeValue = Math.min(this.globalUpdatesPositive.get(i), this.globalUpdatesNegative.get(i));
      if (negativeValue < mostNegative) {
        mostNegative = negativeValue;
        candidateIndex = i;
      }
      
    }
    
    if (candidateIndex != -1) {
      double delta = this.learningRate * mostNegative;
      if (candidateIndex == 0) {
        model.setBiasWeight(model.getBias() + delta);
      }
      else {
        model.updateWeight(candidateIndex, delta);
      }
      
      if (model.getBias() < EPSILON) {
        model.setBiasWeight(0.0);
      }
      
      for (int i = 0; i < model.getFeatureDimension(); ++i) {
        if (model.getFeatureWeight(i) < EPSILON) {
          model.setWeight(i, 0.0);
        }
      }
      
      try {
        model.writeToFile(modelPath, context.getConfiguration());
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    
  }

}
