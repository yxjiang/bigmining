package edu.fiu.cs.bigmining.linearregression.lasso;

import edu.fiu.cs.bigmining.linearregression.TestLinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.linearregression.lasso.LassoLinearRegressionDriver;

public class TestLassoLinearRegressionDriver extends TestLinearRegressionDriver {

  @Override
  protected void run() throws Exception {
    String[] args = { "-i", trainingDataStr, "-m", modelPathStr, "-d", "" + featureDimension,
        "-itr", "50", "-l", "0.2", "-r", "0.8" };
    LassoLinearRegressionDriver.main(args);
  }

  @Override
  public void extraSetup() {
  }

}
