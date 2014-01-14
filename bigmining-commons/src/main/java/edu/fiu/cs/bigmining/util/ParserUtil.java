package edu.fiu.cs.bigmining.util;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Option;

/**
 * Facilitate the command line argument parsing.
 * 
 */
public class ParserUtil {
  
  /**
   * Parse and return the string parameter.
   * @param cli
   * @param option
   * @return
   */
  public static String getString(CommandLine cli, Option option) {
    Object val = cli.getValue(option);
    if (val != null) {
      return val.toString();
    }
    return null;
  }
  
  /**
   * Parse and return the integer parameter.
   * @param cli
   * @param option
   * @return
   */
  public static Integer getInteger(CommandLine cli, Option option) {
    Object val = cli.getValue(option);
    if (val != null) {
      return Integer.parseInt(val.toString());
    }
    return null;
  }
  
  /**
   * Parse and return the long parameter.
   * @param cli
   * @param option
   * @return
   */
  public static Long getLong(CommandLine cli, Option option) {
    Object val = cli.getValue(option);
    if (val != null) {
      return Long.parseLong(val.toString());
    }
    return null;
  }

  /**
   * Parse and return the double parameter.
   * @param cli
   * @param option
   * @return
   */
  public static Double getDouble(CommandLine cli, Option option) {
    Object val = cli.getValue(option);
    if (val != null) {
      return Double.parseDouble(val.toString());
    }
    return null;
  }
  
  /**
   * Parse and return the boolean parameter. If the parameter is set, it is true, otherwise it is false.
   * @param cli
   * @param option
   * @return
   */
  public static boolean getBoolean(CommandLine cli, Option option) {
    return cli.hasOption(option);
  }
}
