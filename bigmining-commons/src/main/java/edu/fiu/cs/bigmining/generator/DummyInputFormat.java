package edu.fiu.cs.bigmining.generator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

/**
 * The input format that feeds only feeds on data to each mapper.
 * 
 */
public class DummyInputFormat<K extends WritableComparable<K>, V extends Writable>
    extends InputFormat<K, V> {

  @Override
  public List<InputSplit> getSplits(JobContext context) throws IOException,
      InterruptedException {
    List<InputSplit> splits = new ArrayList<InputSplit>();
    splits.add(new DummyInputSplit());
    return splits;
  }

  @Override
  public RecordReader<K, V> createRecordReader(InputSplit split,
      TaskAttemptContext context) throws IOException, InterruptedException {
    // TODO Auto-generated method stub
    return new DummyRecordReader<K, V>();
  }

}
