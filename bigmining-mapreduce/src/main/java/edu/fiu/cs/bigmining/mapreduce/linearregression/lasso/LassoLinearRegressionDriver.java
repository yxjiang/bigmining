package edu.fiu.cs.bigmining.mapreduce.linearregression.lasso;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;

import edu.fiu.cs.bigmining.mapreduce.linearregression.LinearRegressionDriver;
import edu.fiu.cs.bigmining.mapreduce.util.PairWritable;

/**
 * The driver of lasso linear regression that parses the input arguments and run
 * the job.
 * 
 */
public class LassoLinearRegressionDriver extends LinearRegressionDriver {

  public LassoLinearRegressionDriver(Class driverClass, Class<? extends Mapper> mapperClass,
      Class<? extends Reducer> reducerClass, Class mapperOutputKey, Class mapperOutputValue) {
    super(driverClass, mapperClass, reducerClass, mapperOutputKey, mapperOutputValue);
  }

  public static void main(String[] args) throws Exception {
    int exit = ToolRunner.run(new LassoLinearRegressionDriver(LassoLinearRegressionDriver.class,
        LassoLinearRegressionMapper.class, LassoLinearRegressionReducer.class,
        NullWritable.class, PairWritable.class), args);
    System.exit(exit);
  }

}
