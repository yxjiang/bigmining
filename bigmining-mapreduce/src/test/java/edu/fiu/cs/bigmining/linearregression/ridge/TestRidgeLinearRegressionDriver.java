package edu.fiu.cs.bigmining.linearregression.ridge;

import edu.fiu.cs.bigmining.linearregression.TestLinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.linearregression.ridge.RidgeLinearRegressionDriver;

public class TestRidgeLinearRegressionDriver extends TestLinearRegressionDriver {

  @Override
  protected void run() throws Exception {
    String[] args = { "-i", trainingDataStr, "-m", modelPathStr, "-d", "" + featureDimension,
        "-itr", "10", "-l", "0.2", "-r", "0.1" };
    RidgeLinearRegressionDriver.main(args);
  }

  @Override
  public void extraSetup() {
  }

}
