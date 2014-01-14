package edu.fiu.cs.bigmining.generator.vector;

import org.junit.Test;

public class TestRandomVectorGenerator {

  @Test
  public void testRandomVectorGenerator() throws Exception {
    String[] args = { "-o",
        "/user/yjian004/random/vector", "-ow",
        "-h", "bigdata-node01.cs.fiu.edu",
        "-fd", "10", "-ld", "3" };

    RandomVectorGenerator.main(args);
  }

}
