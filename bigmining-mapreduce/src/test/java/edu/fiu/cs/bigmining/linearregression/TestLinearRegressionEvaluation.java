package edu.fiu.cs.bigmining.linearregression;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionModel;
import edu.fiu.cs.bigmining.math.ErrorMeasure;
import edu.fiu.cs.bigmining.math.RMSE;
import edu.fiu.cs.bigmining.util.TestBase;

public class TestLinearRegressionEvaluation extends TestBase {
  
  private static final Logger log = LoggerFactory.getLogger(TestLinearRegressionEvaluation.class);
  
  private String modelPathStr = String.format("/tmp/%s", "linear-regression-model.model");
  private String trainingDataStr = String.format("/tmp/%s", "linear-regression-training.data");
  
  @Override
  public void extraSetup() {
  }
  
  /**
   * Evaluate the performance of linear regression.
   * @throws IOException 
   */
  @Test
  public void evaluate() throws IOException {
    log.info("Begin to evaluate the model...");
    
    LinearRegressionModel model = new LinearRegressionModel(modelPathStr, conf);
    
    Path path = new Path(trainingDataStr);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    
    NullWritable key = NullWritable.get();
    VectorWritable val = new VectorWritable();
    
    ErrorMeasure rmse = new RMSE();
    
    while (reader.next(key, val)) {
      Vector instance = val.get();
      Vector label = model.predict(instance);
      rmse.accumulate(instance.getQuick(instance.size() - 1), label.get(0));
    }
    
    System.out.printf("RMSE error: %f\n", rmse.getError());
    
    Closeables.close(reader, true);
    
    printWeights(model);
    
    log.info("End of evaluation.");
  }
  
  private void printWeights(LinearRegressionModel model) {
    double bias = model.getBias();
    Vector vec = model.getFeatureWeights();
    
    System.out.printf("%f\t%s\n", bias, vec);
  }

}
