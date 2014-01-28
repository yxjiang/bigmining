package edu.fiu.cs.bigmining.mapreduce.linearregression;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleFunction;

import com.google.common.io.Closeables;

import edu.fiu.cs.bigmining.mapreduce.PredictiveModel;

public class LinearRegressionModel extends PredictiveModel implements Writable {

  public static final int DIMENSION_THRESHOLD = 10000;

  private Map<String, String> modelMetadata;

  private double bias;

  private Vector features;

  public static LinearRegressionModel getCopy(LinearRegressionModel model) {
    LinearRegressionModel copy = new LinearRegressionModel(model.getFeatureWeights().size(),
        model.getMetadata());

    copy.setBiasWeight(model.getBias());
    copy.features = model.getFeatureWeights().clone();

    return copy;
  }

  /**
   * 
   * @param dimension The number of dimensions of the features (bias in the
   *          model is excluded).
   * @param modelMetadata
   * @return
   */
  public static LinearRegressionModel initializeModel(int dimension,
      Map<String, String> modelMetadata) {
    return new LinearRegressionModel(dimension, modelMetadata);
  }

  private LinearRegressionModel(int dimension, Map<String, String> modelMetadata) {
    final Random rnd = new Random();
    this.bias = rnd.nextDouble();
    if (dimension <= DIMENSION_THRESHOLD) {
      this.features = new DenseVector(dimension);
    } else {
      this.features = new RandomAccessSparseVector(dimension);
    }

    features = features.assign(new DoubleFunction() {
      @Override
      public double apply(double x) {
        return rnd.nextDouble();
      }
    });

    this.modelMetadata = new HashMap<String, String>();
  }

  public LinearRegressionModel(String modelPath, Configuration conf) throws IOException {
    this.readFromFile(modelPath, conf);
  }

  public double getFeatureWeight(int index) {
    return this.features.getQuick(index);
  }

  public double getBias() {
    return this.bias;
  }

  /**
   * Get all the weights, bias excluded.
   * 
   * @return
   */
  public Vector getFeatureWeights() {
    return this.features;
  }
  
  public int getFeatureDimension() {
    return this.features.size();
  }

  /**
   * Set the weight of a specific index, starting from 0 and bias is excluded.
   * @param index
   * @param weight
   */
  public void setWeight(int index, double weight) {
    this.features.set(index, weight);
  }

  public void setBiasWeight(double weight) {
    this.bias = weight;
  }

  /**
   * Update the weight for a specific index. The updated weight will becomes
   * w_{index} + delta.
   * 
   * @param index
   * @param delta
   */
  public void updateWeight(int index, double delta) {
    features.set(index, features.getQuick(index) + delta);
  }

  /**
   * Update all the weights including the bias (the 0th element).
   * 
   * @param updates
   */
  public void updateAllWeights(Vector updates) {
    this.bias += updates.get(0);
    for (int i = 0; i < features.size(); ++i) {
      features.set(i, features.getQuick(i) + updates.get(i + 1));
    }
  }

  public Map<String, String> getMetadata() {
    return this.modelMetadata;
  }

  /**
   * Evaluate the y = w^T x
   * 
   * @param values
   * @return
   */
  @Override
  public Vector predict(Vector values) {
    double value = bias;

    if (values.size() == features.size()) {
      value += features.dot(values);
    } else {
      value += features.dot(values.viewPart(0, features.size()));
    }

    double[] result = new double[] { value };
    return new DenseVector(result);
  }

  public void write(DataOutput out) throws IOException {
    int metaDataSize = this.modelMetadata.size();
    out.writeInt(metaDataSize);
    for (Map.Entry<String, String> entry : this.modelMetadata.entrySet()) {
      WritableUtils.writeString(out, entry.getKey());
      WritableUtils.writeString(out, entry.getValue());
    }
    out.writeDouble(bias);
    VectorWritable.writeVector(out, features);
  }

  public void readFields(DataInput in) throws IOException {
    int metaDataSize = in.readInt();
    this.modelMetadata = new HashMap<String, String>();
    for (int i = 0; i < metaDataSize; ++i) {
      this.modelMetadata.put(WritableUtils.readString(in), WritableUtils.readString(in));
    }
    this.bias = in.readDouble();
    this.features = VectorWritable.readVector(in);
  }

  /**
   * Write the model to specified location.
   * 
   * @param modelPath
   * @param conf
   * @throws IOException
   */
  public void writeToFile(String modelPath, Configuration conf) throws IOException {
    FSDataOutputStream os = null;
    try {
      URI uri = new URI(modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      os = fs.create(new Path(modelPath), true);
      this.write(os);
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

    Closeables.close(os, false);
  }

  /**
   * Read the model from specified location.
   * 
   * @param modelPath
   * @param conf
   * @throws IOException
   */
  public void readFromFile(String modelPath, Configuration conf) throws IOException {
    FSDataInputStream is = null;
    try {
      URI uri = new URI(modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      is = new FSDataInputStream(fs.open(new Path(modelPath)));
      this.readFields(is);
    } catch (URISyntaxException e) {
      e.printStackTrace();
    } finally {
      Closeables.close(is, false);
    }
  }

  /**
   * Check whether two linear regression model are identical.
   * 
   * @param otherModel
   * @param epsilon
   * @return
   */
  public boolean isIdentical(LinearRegressionModel otherModel, double epsilon) {
    // The difference of all features must be smaller than epsilon
    return features.minus(otherModel.features).norm(Double.POSITIVE_INFINITY) <= epsilon
        && Math.abs(bias - otherModel.bias) <= epsilon;
  }

}
