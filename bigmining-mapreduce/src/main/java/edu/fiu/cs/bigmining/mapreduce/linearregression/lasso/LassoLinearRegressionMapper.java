package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.jet.math.Constants;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

/**
 * The Mapper of Lasso linear regression.
 * 
 */
public class LassoLinearRegressionMapper extends
    Mapper<NullWritable, VectorWritable, BooleanWritable, PairWritable> {

  /* a sparse vector contains the weight updates */
  private long count;
  private int featureDimension;

  private double learningRate;
  private double regularizationRate;

  private double biasUpdate;
  private Vector weightUpdatesPositive;
  private Vector weightUpdatesNegative;

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
      this.weightUpdatesPositive = new DenseVector(this.featureDimension);
      this.weightUpdatesNegative = new DenseVector(this.featureDimension);
    } else {
      this.weightUpdatesPositive = new RandomAccessSparseVector(this.featureDimension);
      this.weightUpdatesNegative = new RandomAccessSparseVector(this.featureDimension);
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

    // compute local d_{e_k} f(w) and d_{-e_k} f(w)
    for (int i = 0; i < weightUpdatesPositive.size(); ++i) {
      double xi = vec.get(i);
      double wpos = weightUpdatesPositive.getQuick(i);
      double wneg = weightUpdatesNegative.getQuick(i);
      if (expected - actual > Constants.EPSILON) {
        weightUpdatesPositive.set(i, wpos - xi);
        weightUpdatesNegative.set(i, wneg + xi);
      }
      else if (expected - actual < -Constants.EPSILON) {
        weightUpdatesPositive.set(i, wpos + xi);
        weightUpdatesNegative.set(i, wneg - xi);
      }
      else {
        double absXi = Math.abs(xi);
        weightUpdatesPositive.set(i, wpos + absXi);
        weightUpdatesNegative.set(i, wneg + absXi);
      }
    }
  }

  /**
   * Write local updates to reducer. Local update: \delta w = learningRate *
   * \frac{1}{count} \sigma_{count} (y - t) * x
   */
  public void cleanup(Context context) throws IOException, InterruptedException {
    // the output vector does not contains the bias
    BooleanWritable weightUpdatesPositiveKey = new BooleanWritable(false); // false denotes the positive
    BooleanWritable weightUpdatesNegativeKey = new BooleanWritable(true);  // true denotes the negative
    LongWritable countWritable = new LongWritable(count);
    
    PairWritable positivePair = new PairWritable(countWritable, new VectorWritable(weightUpdatesPositive));
    PairWritable negativePair = new PairWritable(countWritable, new VectorWritable(weightUpdatesNegative));
    context.write(weightUpdatesPositiveKey, positivePair);
    context.write(weightUpdatesNegativeKey, negativePair);
  }

}
