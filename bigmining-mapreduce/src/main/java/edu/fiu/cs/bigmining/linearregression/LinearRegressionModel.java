package edu.fiu.cs.bigmining.linearregression;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.mahout.math.Vector;

import com.google.common.io.Closeables;

public class LinearRegressionModel implements Writable {
  
  private Map<String, String> modelMetadata;
  
  private double bias;
  
  private double[] features;
  
  public LinearRegressionModel(int dimension, Map<String, String> modelMetadata) {
    this.bias = 0;
    this.features = new double[dimension];
    this.modelMetadata = new HashMap<String, String>();
  }
  
  public LinearRegressionModel(String modelPath, Configuration conf) throws IOException {
    this.readFromFile(modelPath, conf);
  }
  
  public double getWeight(int index) {
    return this.features[index];
  }
  
  public double getBias() {
    return this.bias;
  }
  
  public void setWeight(int index, double weight) {
    this.features[index] = weight;
  }
  
  public void setBiasWeight(double weight) { 
    this.bias = weight;
  }
  
  public void updateWeights(Vector updates) {
    this.bias -= updates.get(0);
    for (int i = 0; i < features.length; ++i) {
      this.features[i] = updates.get(i + 1);
    }
  }
  
  /**
   * Evaluate the y = w^T x
   * @param values
   * @return
   */
  public double predict(Vector values) {
    double value = bias;
    for (int i = 0; i < features.length; ++i) {
      value += features[i] * values.get(i);
    }
    return value;
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
   * @param modelPath
   * @param conf
   * @throws IOException
   */
  public void writeToFile(String modelPath, Configuration conf) throws IOException {
    FSDataOutputStream is = null;
    try {
      URI uri = new URI(modelPath);
      FileSystem fs = FileSystem.get(uri, conf);
      is = fs.create(new Path(modelPath), true);
      this.write(is);
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

    Closeables.close(is, false);
  }
  
  /**
   * Read the model from specified location.
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

}
