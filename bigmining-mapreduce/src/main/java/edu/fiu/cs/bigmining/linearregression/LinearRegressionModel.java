package edu.fiu.cs.bigmining.linearregression;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

public class LinearRegressionModel implements Writable {
  
  private Map<String, String> modelMetadata;
  
  private double bias;
  
  private double[] features;
  
  public LinearRegressionModel(int dimension, Map<String, String> modelMetadata) {
    this.bias = 0;
    this.features = new double[dimension];
    this.modelMetadata = new HashMap<String, String>();
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

}
