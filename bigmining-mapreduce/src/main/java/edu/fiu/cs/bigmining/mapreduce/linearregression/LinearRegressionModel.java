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
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

import edu.fiu.cs.bigmining.mapreduce.PredictiveModel;

public class LinearRegressionModel extends PredictiveModel implements Writable {

  private Map<String, String> modelMetadata;

  private double bias;

  private double[] features; 

  public static LinearRegressionModel getCopy(LinearRegressionModel model) {
    double[] features = model.getFeatureWeights();
    LinearRegressionModel copy = new LinearRegressionModel(features.length, model.getMetadata());

    copy.setBiasWeight(model.getBias());
    for (int i = 0; i < features.length; ++i) {
      copy.setWeight(i, features[i]);
    }

    return copy;
  }
  
  /**
   * 
   * @param dimension   The number of dimensions of the features (bias in the model is excluded).
   * @param modelMetadata
   * @return
   */
  public static LinearRegressionModel initializeModel(int dimension, Map<String, String> modelMetadata) {
    return new LinearRegressionModel(dimension, modelMetadata);
  }

  private LinearRegressionModel(int dimension, Map<String, String> modelMetadata) {
    Random rnd = new Random();
    this.bias = rnd.nextDouble();
    this.features = new double[dimension];
    for (int i = 0; i < features.length; ++i) {
      this.features[i] = rnd.nextDouble();
    }
    this.modelMetadata = new HashMap<String, String>();
  }

  public LinearRegressionModel(String modelPath, Configuration conf) throws IOException {
    this.readFromFile(modelPath, conf);
  }

  public double getFeatureWeight(int index) {
    return this.features[index];
  }

  public double getBias() {
    return this.bias;
  }

  public double[] getFeatureWeights() {
    return this.features;
  }

  public void setWeight(int index, double weight) {
    this.features[index] = weight;
  }

  public void setBiasWeight(double weight) {
    this.bias = weight;
  }

  /**
   * Update all the weights including the bias (the 0th element).
   * @param updates
   */
  public void updateAllWeights(Vector updates) {
    this.bias += updates.get(0);
    for (int i = 0; i < features.length; ++i) {
      this.features[i] += updates.get(i + 1);
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
    for (int i = 0; i < features.length; ++i) {
      value += features[i] * values.get(i);
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
    out.writeInt(features.length);
    for (int i = 0; i < features.length; ++i) {
      out.writeDouble(features[i]);
    }
  }

  public void readFields(DataInput in) throws IOException {
    int metaDataSize = in.readInt();
    this.modelMetadata = new HashMap<String, String>();
    for (int i = 0; i < metaDataSize; ++i) {
      this.modelMetadata.put(WritableUtils.readString(in), WritableUtils.readString(in));
    }
    this.bias = in.readDouble();
    this.features = new double[in.readInt()];
    for (int i = 0; i < features.length; ++i) {
      this.features[i] = in.readDouble();
    }
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
    double[] otherFeatures = otherModel.features;
    if (features.length != otherFeatures.length) {
      return false;
    }

    for (int i = 0; i < features.length; ++i) {
      if (Math.abs(features[i] - otherFeatures[i]) > epsilon) {
        return false;
      }
    }

    return Math.abs(bias - otherModel.bias) <= epsilon;
  }

}
