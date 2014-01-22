package edu.fiu.cs.bigmining.math;

public abstract class ErrorMeasure {
  
  protected double error;
  
  public abstract void accumulate(double target, double actual);
  
  public abstract double getError();

}
