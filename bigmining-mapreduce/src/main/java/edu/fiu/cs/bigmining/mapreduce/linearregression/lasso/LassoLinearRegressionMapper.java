package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
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
    Mapper<NullWritable, VectorWritable, NullWritable, PairWritable> {
  
  private static final double EPSILON = Constants.EPSILON * 100;

  /* a sparse vector contains the weight updates */
  private long count;
  private int featureDimension;

  private Vector weightUpdates;

  private LinearRegressionModel model;

  @Override
  public void setup(Context context) {
    Configuration conf = context.getConfiguration();
    this.count = 0;
    this.featureDimension = conf.getInt("feature.dimension", 0);

    String modelPath = conf.get("model.path");

    if (this.featureDimension <= LinearRegressionModel.DIMENSION_THRESHOLD) {
      this.weightUpdates = new DenseVector(this.featureDimension + 1);
    } else {
      this.weightUpdates = new RandomAccessSparseVector(this.featureDimension + 1);
    }

    try { // load the model into memory
      model = new LinearRegressionModel(modelPath, conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  /**
   * Train the model with full-batch greedy coordinate regression.
   * The last dimension of the value is the label.
   */
  public void map(NullWritable key, VectorWritable value, Context context) throws IOException {
    ++count;
    Vector vec = value.get();
    double expected = vec.get(featureDimension);
    double actual = model.predict(vec).get(0);
    
    // update bias
    weightUpdates.setQuick(0, weightUpdates.getQuick(0) - (expected - actual));
    
    // compute local d_{e_k} f(w) and d_{-e_k} f(w)
    for (int i = 1; i < weightUpdates.size(); ++i) {
      double xi = vec.get(i);
      
      double w = weightUpdates.getQuick(i);
      weightUpdates.setQuick(i, w - (expected - actual) * xi);
    }

  }

  /**
   * Write local updates to reducer. 
   */
  public void cleanup(Context context) throws IOException, InterruptedException {
    // the output vector does not contains the bias
    LongWritable countWritable = new LongWritable(count);
    VectorWritable vecWrtiable = new VectorWritable(this.weightUpdates);
    context.write(NullWritable.get(), new PairWritable(countWritable, vecWrtiable));
  }

}
