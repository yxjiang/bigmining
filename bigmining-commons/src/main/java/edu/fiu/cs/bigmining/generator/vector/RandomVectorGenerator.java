package edu.fiu.cs.bigmining.generator.vector;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.VectorWritable;

import edu.fiu.cs.bigmining.util.ParserUtil;

/**
 * Randomly generate a set of double vectors.
 * 
 */
public class RandomVectorGenerator extends Configured implements Tool {

  /* The directory path of the output */
  private String outputPathStr;

  /* Whether to overwrite the existing directory */
  private boolean isOverwrite;

  /* The number of vectors to generate */
  private long numInstances;

  /* The size of dimension of the features */
  private int featureDimension;

  /* Whether the label for regression or for classification */
  private boolean isRegression;

  /* The size of dimension of the labels */
  private int labelDimension;

  /**
   * Parse the arguments.
   * 
   * @param args
   * @return
   */
  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    GroupBuilder groupBuilder = new GroupBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    
    Option overwriteOption = optionBuilder.withLongName("overwrite")
        .withShortName("ow")
        .withDescription("Whether overwrites the existing files.")
        .withRequired(false).create();

    Group overwriteGroup = groupBuilder.withOption(overwriteOption).create();

    Option outputPathOption = optionBuilder.withLongName("output")
        .withShortName("o")
        .withDescription("The directory path of output files.")
        .withRequired(true).withChildren(overwriteGroup).create();
    
    Option featureDimensionOption = optionBuilder
        .withLongName("featureDimension").withShortName("fd")
        .withDescription("The number of dimensions for the features.")
        .withRequired(true)
        .withArgument(argumentBuilder.withMinimum(1).withMaximum(1).create())
        .create();
    
    Option labelsDimensionOption = optionBuilder
        .withLongName("featureDimension").withShortName("ld")
        .withDescription("The number of dimensions for the labels.")
        .withRequired(true)
        .withArgument(argumentBuilder.withMinimum(1).withMaximum(1).create())
        .create();
    
    Option vectorTypeOption = optionBuilder
        .withLongName("type")
        .withShortName("t")
        .withDescription("The task type the generated data used for.")
        .withArgument(
            argumentBuilder.withDefault("regression").withMinimum(1)
                .withMaximum(1).create()).create();
    
    Parser parser = new Parser();
    Group normalGroup = groupBuilder.withOption(outputPathOption)
        .withOption(overwriteOption).withOption(featureDimensionOption)
        .withOption(labelsDimensionOption).withOption(vectorTypeOption).create();
    
    parser.setGroup(normalGroup);
    
    CommandLine cli = parser.parseAndHelp(args);
    if (cli == null) {
      return false;
    }
    
    this.outputPathStr = ParserUtil.getString(cli, outputPathOption);
    this.isOverwrite = ParserUtil.getBoolean(cli, overwriteOption);
    this.featureDimension = ParserUtil.getInteger(cli, featureDimensionOption);
    this.labelDimension = ParserUtil.getInteger(cli, labelsDimensionOption);
    this.isRegression = ParserUtil.getString(cli, vectorTypeOption).equalsIgnoreCase("regression")? true : false;
    
    return true;
  }
  
  public int run(String[] args) throws Exception {

    if (!parseArgs(args)) {
      return -1;
    }

    Configuration conf = new Configuration(getConf());
    conf.setInt("feature.dimension", this.featureDimension);
    conf.setInt("label.dimension", this.labelDimension);

    Job job = new Job(conf);
    job.setJarByClass(RandomVectorGenerator.class);

    // validate the output path
    Path outputPath = new Path(this.outputPathStr);
    FileSystem fs = outputPath.getFileSystem(conf);
    if (fs.exists(outputPath)) {
      if (!this.isOverwrite) {
        return -1;
      } else {
        fs.delete(outputPath, true);
      }
    }

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.setMapperClass(RandomVectorGeneratorMapper.class);
    job.setNumReduceTasks(0);

    FileOutputFormat.setOutputPath(job, outputPath);

    job.waitForCompletion(true);

    return 0;
  }
  
  public static void main(String[] args) throws Exception {
    int exitCode = ToolRunner.run(new RandomVectorGenerator(), args);
    System.exit(exitCode);
  }

}
