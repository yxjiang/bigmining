package edu.fiu.cs.bigmining.util;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.junit.Before;

public abstract class TestBase {
  
  protected String base = "/tmp";
  protected Random rnd = new Random();
  
  protected Configuration conf;
  protected FileSystem fs;
  
  @Before
  public void setup() {
    conf = new Configuration();
    try {
      fs = FileSystem.get(conf);
    } catch (IOException e) {
      e.printStackTrace();
    }
    extraSetup();
  }
  
  public abstract void extraSetup();
  
  /**
   * Get a temporal test directory.
   * @return
   */
  public String getTestTempDir() {
    return String.format("%s/%d-%d/", base, rnd.nextInt(), rnd.nextInt());
  }
  
}
