package edu.fiu.cs.bigmining.mapreduce.util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VectorWritable;

/**
 * A composite writable data structure that stores a boolean key  vector writable value pair. 
 *
 */
public class PairWritable implements Writable {
  
  private LongWritable key;
  private VectorWritable value;
  
  public PairWritable() {
    key = new LongWritable();
    value = new VectorWritable();
  }
  
  public PairWritable(LongWritable key, VectorWritable value) {
    this.key = key;
    this.value = value;
  }
  
  public void setKey(LongWritable key) {
    this.key = key;
  }
  
  public void setValue(VectorWritable val) {
    this.value = val;
  }
  
  public LongWritable getKey() {
    return this.key;
  }
  
  public VectorWritable getValue() {
    return this.value;
  }

  public void write(DataOutput out) throws IOException {
    key.write(out);
    value.write(out);
  }

  public void readFields(DataInput in) throws IOException {
    key.readFields(in);
    value.readFields(in);
  }

}