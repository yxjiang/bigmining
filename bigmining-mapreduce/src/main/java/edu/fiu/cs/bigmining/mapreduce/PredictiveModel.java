package edu.fiu.cs.bigmining.mapreduce;

import org.apache.mahout.math.Vector;

public abstract class PredictiveModel {
  
  public abstract Vector predict(Vector features);

}
