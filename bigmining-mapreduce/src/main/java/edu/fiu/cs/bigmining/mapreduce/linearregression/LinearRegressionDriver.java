package edu.fiu.cs.bigmining.mapreduce.linearregression;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.fiu.cs.bigmining.util.ParserUtil;

/**
 * The driver to run the linear regression.
 * 
 */
public class LinearRegressionDriver extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(LinearRegressionDriver.class);

  private static final double EPSILON = 0.01;

  private String trainingDataPath;
  private String modelPath;
  private int dimension;
  private int maxIterations;
  private double learningRate;
  private Map<String, String> metaData;

  private LinearRegressionModel model;

  /**
   * Initialize the model.
   * 
   * @param metaData
   * @throws IOException
   */
  private void initializeModel() throws IOException {
    model = LinearRegressionModel.initializeModel(dimension, metaData);
    model.writeToFile(modelPath, getConf());
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    GroupBuilder groupBuilder = new GroupBuilder();

    Option trainingDataPathOption = optionBuilder.withShortName("i").withLongName("training-data")
        .withDescription("The training data path to put the model")
        .withArgument(argumentBuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withRequired(true).create();

    Option modelPathOption = optionBuilder.withShortName("m").withLongName("model-path")
        .withDescription("The path to put the model")
        .withArgument(argumentBuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withRequired(true).create();

    Option modelDimensionOption = optionBuilder
        .withShortName("d")
        .withLongName("dimension")
        .withDescription("the feature dimension")
        .withArgument(
            argumentBuilder.withName("dimension-size").withMinimum(1).withMaximum(1).create())
        .withRequired(true).create();

    Option iterationsOption = optionBuilder
        .withShortName("itr")
        .withLongName("iteration")
        .withDescription("the maximum iterations")
        .withArgument(
            argumentBuilder.withName("iteration-number").withMinimum(1).withMaximum(1)
                .withDefault(1).create()).withRequired(false).create();

    Option learningRateOption = optionBuilder
        .withShortName("l")
        .withLongName("learning-rate")
        .withDescription("the learning rate for training")
        .withArgument(
            argumentBuilder.withName("learning-rate").withMinimum(1).withMaximum(1)
                .withDefault(0.1).create()).withRequired(false).create();

    // the key value pairs of meta data
    Option metaDataOption = optionBuilder.withShortName("meta").withLongName("metadata")
        .withDescription("the key=value pairs")
        .withArgument(argumentBuilder.withName("pairs").create()).withRequired(false).create();

    Group normalGroup = groupBuilder.withOption(trainingDataPathOption).withOption(modelPathOption)
        .withOption(modelDimensionOption).withOption(iterationsOption)
        .withOption(learningRateOption).withOption(metaDataOption).create();

    Parser parser = new Parser();
    parser.setGroup(normalGroup);
    parser.setHelpFormatter(new HelpFormatter());
    parser.setHelpTrigger("--help");

    CommandLine cli = parser.parseAndHelp(args);
    if (cli == null) {
      return false;
    }

    this.trainingDataPath = ParserUtil.getString(cli, trainingDataPathOption);
    this.modelPath = ParserUtil.getString(cli, modelPathOption);
    this.dimension = ParserUtil.getInteger(cli, modelDimensionOption);
    this.maxIterations = ParserUtil.getInteger(cli, iterationsOption);
    this.learningRate = ParserUtil.getDouble(cli, learningRateOption);
    this.learningRate = Math.max(0.0001, learningRate);
    this.metaData = ParserUtil.getMap(cli, metaDataOption, "=");

    return true;
  }

  public int run(String[] args) throws Exception {
    log.info("Parsing input arguments...");
    if (!parseArgs(args)) {
      return -1;
    }

    log.info("Initializing model...");
    Configuration conf = getConf();
    conf.set("model.path", this.modelPath);
    conf.setInt("feature.dimension", this.dimension);
    initializeModel();

    Path trainingDataPath = new Path(this.trainingDataPath);
    LinearRegressionModel prevModel = null;

    int curIteration = 0;
    // loop until model converges or exceeds maximal iteration
    do {
      log.info(String.format("Iteration %d.", curIteration));
      ++curIteration;
      prevModel = LinearRegressionModel.getCopy(this.model);
      Job job = new Job(conf, String.format("Linear Regression: iteration %d", curIteration));
      job.setJarByClass(LinearRegressionDriver.class);
      job.setMapperClass(LinearRegressionMapper.class);
      job.setReducerClass(LinearRegressionReducer.class);

      // configure input and output
      FileInputFormat.addInputPath(job, trainingDataPath);
      job.setMapOutputKeyClass(NullWritable.class);
      job.setMapOutputValueClass(PairWritable.class);
      
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputKeyClass(NullWritable.class);
      job.setOutputValueClass(NullWritable.class);
      job.setOutputFormatClass(NullOutputFormat.class);
      job.setNumReduceTasks(1);

      job.waitForCompletion(true);
      model = new LinearRegressionModel(this.modelPath, conf);
    } while (curIteration < maxIterations || !prevModel.isIdentical(model, EPSILON));

    return 0;
  }

  public static void main(String[] args) throws Exception {
    int exit = ToolRunner.run(new LinearRegressionDriver(), args);
    System.exit(exit);
  }

}
