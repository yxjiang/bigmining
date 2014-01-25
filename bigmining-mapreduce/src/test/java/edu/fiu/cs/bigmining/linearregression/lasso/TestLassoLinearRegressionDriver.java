package edu.fiu.cs.bigmining.linearregression.lasso;

import edu.fiu.cs.bigmining.linearregression.TestLinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.linearregression.lasso.LassoLinearRegressionDriver;

public class TestLassoLinearRegressionDriver extends TestLinearRegressionDriver {

  @Override
  protected void run() throws Exception {
    String[] args = { "-i", trainingDataStr, "-m", modelPathStr, "-d", "" + featureDimension,
        "-itr", "10", "-l", "0.5", "-r", "0.01" };
    LassoLinearRegressionDriver.main(args);
  }

  @Override
  public void extraSetup() {
  }

}
