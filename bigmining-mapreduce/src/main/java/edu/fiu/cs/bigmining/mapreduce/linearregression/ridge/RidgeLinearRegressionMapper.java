package edu.fiu.cs.bigmining.mapreduce.linearregression.ridge;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

/**
 * The Mapper for linear regression. Two kinds of keys (boolean) are generated,
 * the 'FALSE' key denotes the count, the 'TRUE' key denotes the aggregated
 * vector.
 * 
 */
public class RidgeLinearRegressionMapper extends
    Mapper<NullWritable, VectorWritable, NullWritable, PairWritable> {

  /* a sparse vector contains the weight updates */
  private long count;
  private int featureDimension;

  private double learningRate;
  private double regularizationRate;

  private double biasUpdate;
  private Vector weightUpdates;

  private LinearRegressionModel model;

  @Override
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.count = 0;
    this.featureDimension = conf.getInt("feature.dimension", 0);
    this.learningRate = Double.parseDouble(conf.get("learning.rate") != null ? conf
        .get("learning.rate") : "0.01");
    this.regularizationRate = Double.parseDouble(conf.get("regularization.rate") != null ? conf
        .get("learning.rate") : "0.01");

    String modelPath = conf.get("model.path");

    this.biasUpdate = 0;

    if (this.featureDimension <= LinearRegressionModel.DIMENSION_THRESHOLD) {
      this.weightUpdates = new DenseVector(this.featureDimension);
    } else {
      this.weightUpdates = new RandomAccessSparseVector(this.featureDimension);
    }

    try { // load the model into memory
      model = new LinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  /**
   * Train the model with full-batch stochastic regression.
   */
  public void map(NullWritable key, VectorWritable value, Context context) throws IOException {
    ++count;
    Vector vec = value.get();
    double expected = vec.get(featureDimension);
    double actual = model.predict(vec).get(0);

    // update bias
    double biasDelta = learningRate * (actual - expected) + regularizationRate * model.getBias();
    biasUpdate -= biasDelta;

    // update each weight
    for (int i = 0; i < featureDimension; ++i) {
      double delta = 0;
      delta = learningRate * ((actual - expected) * vec.get(i) + regularizationRate
          * model.getFeatureWeight(i)); // regularization term
      weightUpdates.set(i, weightUpdates.get(i) - delta);
    }

  }

  /**
   * Write local updates to reducer. Local update: \delta w = learningRate *
   * \frac{1}{count} \sigma_{count} (y - t) * x
   */
  public void cleanup(Context context) throws IOException, InterruptedException {
    Vector vec = new DenseVector(1 + this.featureDimension);
    vec.set(0, this.biasUpdate);
    for (int i = 0; i < featureDimension; ++i) {
      vec.set(i + 1, this.weightUpdates.get(i));
    }
    // the output contains the bias
    context.write(NullWritable.get(), new PairWritable(new LongWritable(count), new VectorWritable(
        vec)));
  }

}
