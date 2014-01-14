package edu.fiu.cs.bigmining.generator.vector;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * The Mapper of {@link RandomVectorGenerator}
 * 
 */
public class RandomVectorGeneratorMapper extends
    Mapper<NullWritable, NullWritable, NullWritable, VectorWritable> {

  private Random rnd;

  /* The number of vectors to generate */
  private int numberVectorsPerMapper;

  /* Feature dimensions */
  private int featureDimension;

  /* Label dimensions */
  private int labelDimension;

  /* Whether the label is for regression or classification */
  private boolean isRegression;

  private NullWritable nullWritable = NullWritable.get();

  @Override
  public void setup(Context context) {
    rnd = new Random();
    Configuration conf = context.getConfiguration();
    this.numberVectorsPerMapper = Math.max(100, conf.getInt("number.vectors.per.mapper", 100));
    this.featureDimension = Math.max(1, conf.getInt("feature.dimension", 1));
    this.labelDimension = Math.max(1, conf.getInt("label.dimension", 1));
    this.isRegression = conf.getBoolean("regression", true);
  }

  @Override
  public void map(NullWritable key, NullWritable value, Context context)
      throws IOException, InterruptedException {

    for (int i = 0; i < numberVectorsPerMapper; ++i) {
      double[] vector = new double[featureDimension + labelDimension];
      for (int j = 0; j < featureDimension; ++j) {
        vector[j] = rnd.nextDouble();
      }
      if (this.isRegression) {
        for (int j = 0; j < this.labelDimension; ++j) {
          vector[featureDimension + j] = rnd.nextDouble();
        }
      } else { // classification
        int classIndex = rnd.nextInt(labelDimension);
        vector[featureDimension + classIndex] = 1;
      }
      Vector vec = new DenseVector(vector);
      context.write(nullWritable, new VectorWritable(vec));
    }

  }

}
