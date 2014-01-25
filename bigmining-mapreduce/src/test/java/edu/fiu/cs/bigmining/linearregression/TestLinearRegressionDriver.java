package edu.fiu.cs.bigmining.linearregression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Arrays;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import edu.fiu.cs.bigmining.mapreduce.linearregression.ridge.RidgeLinearRegressionDriver;
import edu.fiu.cs.bigmining.util.Normalizer;
import edu.fiu.cs.bigmining.util.TestBase;

public abstract class TestLinearRegressionDriver extends TestBase {
  
  protected final Logger log = LoggerFactory.getLogger(TestLinearRegressionDriver.class);

  protected int featureDimension = 15;
  protected String rawDataStr = "../bigmining-commons/src/test/resources/raw-linear-regression.txt";
  protected String modelPathStr = String.format("/tmp/%s", "linear-regression-model.model");
  protected String trainingDataStr = String.format("/tmp/%s", "linear-regression-training.data");
  
  private void generateTrainingData() throws IOException {
    log.info("Generate training data...");

    Path trainingDataPath = new Path(trainingDataStr);

    if (fs.exists(trainingDataPath)) {
      fs.delete(trainingDataPath, true);
    }

    BufferedReader br = new BufferedReader(new FileReader(rawDataStr));
    List<double[]> unnormalizedData = Lists.newArrayList();
    String line = null;
    while ((line = br.readLine()) != null) {
      if (line.trim().isEmpty() || line.startsWith("#")) {
        continue;
      }
      String[] tokens = line.split(" ");
      double[] values = new double[featureDimension + 1];
      for (int i = 0; i < tokens.length; ++i) {
        values[i] = Double.parseDouble(tokens[i]);
      }
      unnormalizedData.add(values);
    }
    Closeables.close(br, true);

    List<double[]> normalizedData = Normalizer.zeroOneNormalization(unnormalizedData);

//    printData(normalizedData);

    SequenceFile.Writer out = new SequenceFile.Writer(fs, conf, trainingDataPath,
        NullWritable.class, VectorWritable.class);
    try {
      for (double[] values : normalizedData) {
        Vector vector = new DenseVector(values);
        VectorWritable vectorWritable = new VectorWritable(vector);
        out.append(NullWritable.get(), vectorWritable);
      }
    } finally {
      Closeables.close(out, true);
    }

    log.info("Finished generating data.");
  }

  private void printData(List<double[]> data) {
    for (double[] arr : data) {
      System.out.println(Arrays.toString(arr));
    }
  }
  
  @Test
  public void testLinearRegressionDriver() throws Exception {
    this.generateTrainingData();
    this.run();
  }
  
  /**
   * Set the parameters can run.
   */
  protected abstract void run() throws Exception;

  /**
   * Delete temporal data.
   */
  public void cleanup() {
    Path trainingDataPath = new Path(this.trainingDataStr);
    try {
      if (fs.exists(trainingDataPath)) {
        fs.delete(trainingDataPath, true);
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
