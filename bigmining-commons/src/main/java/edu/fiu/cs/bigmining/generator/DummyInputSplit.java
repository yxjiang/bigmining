package edu.fiu.cs.bigmining.generator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.mapreduce.InputSplit;

public class DummyInputSplit extends InputSplit implements Writable {
  
  @Override
  public long getLength() throws IOException, InterruptedException {
    return 0;
  }

  @Override
  public String[] getLocations() throws IOException, InterruptedException {
    return new String[] {};
  }

  public void write(DataOutput out) throws IOException {
    WritableUtils.writeString(out, "dummy");
  }

  public void readFields(DataInput in) throws IOException {
    WritableUtils.readString(in);
  }

}
