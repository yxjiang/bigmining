package edu.fiu.cs.bigmining.linearregression;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.util.Normalizer;
import edu.fiu.cs.bigmining.util.TestBase;

public class TestLinearRegressionDriver extends TestBase {

  private static final Logger log = LoggerFactory.getLogger(TestLinearRegressionDriver.class);

  private Configuration conf;
  
  private int featureDimension = 15;
  private String rawDataStr = "../bigmining-commons/src/test/resources/raw-linear-regression.txt";
  private String modelPathStr = String.format("/tmp/%s", "linear-regression-model.model");
  private String trainingDataStr = String.format("/tmp/%s", "linear-regression-training.data");
  private FileSystem fs;

  @Before
  /**
   * Write data to specified location.
   */
  public void setup() throws IOException {
    conf = new Configuration();
  }

  private void generateTrainingData() throws IOException {
    log.info("Generate training data...");
    try {
      fs = FileSystem.get(conf);
    } catch (IOException e) {
      e.printStackTrace();
    }

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

  @Test
  public void testLinearRegressionDriver() throws Exception {
    this.generateTrainingData();

    String[] args = { "-i", trainingDataStr, "-m", modelPathStr, "-d", "" + featureDimension, "-itr", "10" };
    LinearRegressionDriver.main(args);
    
    this.evaluate();
  }
  
  /**
   * Evaluate the performance of linear regression.
   * @throws IOException 
   */
  private void evaluate() throws IOException {
    log.info("Begin to evaluate the model...");
    
    LinearRegressionModel model = new LinearRegressionModel(modelPathStr, conf);
    
    
    log.info("End of evaluation.");
  }

  @After
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
