package edu.fiu.cs.bigmining.util;

import java.util.Random;

public abstract class TestBase {
  
  private String base = "/tmp";
  private Random rnd = new Random();
  
  /**
   * Get a temporal test directory.
   * @return
   */
  public String getTestTempDir() {
    return String.format("%s/%d-%d/", base, rnd.nextInt(), rnd.nextInt());
  }

}
