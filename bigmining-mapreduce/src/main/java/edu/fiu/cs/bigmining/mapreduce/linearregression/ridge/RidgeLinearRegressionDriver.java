package edu.fiu.cs.bigmining.mapreduce.linearregression.ridge;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

/**
 * The driver to run the linear regression.
 * 
 */
public class RidgeLinearRegressionDriver extends LinearRegressionDriver {

  public RidgeLinearRegressionDriver(Class driverClass, Class<? extends Mapper> mapperClass,
      Class<? extends Reducer> reducerClass, Class mapperOutputKey, Class mapperOutputValue) {
    super(driverClass, mapperClass, reducerClass, mapperOutputKey, mapperOutputValue);
  }

  public static void main(String[] args) throws Exception {
    int exit = ToolRunner.run(new RidgeLinearRegressionDriver(RidgeLinearRegressionDriver.class,
        RidgeLinearRegressionMapper.class, RidgeLinearRegressionReducer.class, NullWritable.class,
        PairWritable.class), args);
    System.exit(exit);
  }

}
