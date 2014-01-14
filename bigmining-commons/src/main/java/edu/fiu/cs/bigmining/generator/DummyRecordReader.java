package edu.fiu.cs.bigmining.generator;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

/**
 * For each split, the dummy record reader will only create on dummy key value pair. 
 *
 * @param <K>
 * @param <V>
 */
public class DummyRecordReader<K extends WritableComparable<K>, V extends Writable>
    extends RecordReader<K, V> {

  protected boolean done;

  @Override
  public void initialize(InputSplit split, TaskAttemptContext context)
      throws IOException, InterruptedException {
    // does nothing
  }

  
  @Override
  public boolean nextKeyValue() throws IOException, InterruptedException {
    if (done) {
      return false;
    }
    done = true;
    return true;
  }
  
  @Override
  public K getCurrentKey() {
    return null;
  }

  @Override
  public V getCurrentValue() {
    return null;
  }

  @Override
  public float getProgress() throws IOException {
    if (done) {
      return 1;
    }
    return 0;
  }
  
  @Override
  public void close() {
    done = true;
  }

}