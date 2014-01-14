package edu.fiu.cs.bigmining.generator.vector;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VectorWritable;

public class DummyReducer extends
    Reducer<NullWritable, VectorWritable, NullWritable, VectorWritable> {

  @Override
  public void reduce(NullWritable key, Iterable<VectorWritable> values,
      Context context) throws IOException, InterruptedException {
    Iterator<VectorWritable> itr = values.iterator();
    while (itr.hasNext()) {
      context.write(key, itr.next());
    }
  }
}
