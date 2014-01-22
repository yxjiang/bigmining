package edu.fiu.cs.bigmining.math;

public class RMSE extends ErrorMeasure {
  
  private int count = 0;

  @Override
  public void accumulate(double target, double actual) {
    ++count;
    error += Math.pow(target - actual, 2);
  }

  @Override
  public double getError() {
    return Math.sqrt(error / count);
  }

}
