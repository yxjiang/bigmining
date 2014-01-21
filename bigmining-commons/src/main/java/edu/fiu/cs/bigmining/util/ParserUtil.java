package edu.fiu.cs.bigmining.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Option;

/**
 * Facilitate the command line argument parsing.
 * 
 */
public class ParserUtil {

  /**
   * Parse and return the string parameter.
   * 
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
   * 
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
   * 
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
   * 
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
   * Parse and return the boolean parameter. If the parameter is set, it is
   * true, otherwise it is false.
   * 
   * @param cli
   * @param option
   * @return
   */
  public static boolean getBoolean(CommandLine cli, Option option) {
    return cli.hasOption(option);
  }

  /**
   * Parse and return the map parameter. If the parameter is set, the map is
   * filled, otherwise return an empty map.
   * 
   * @param cli
   * @param option
   * @param separator   the separator that splits the key and value.
   * @return
   */
  public static Map<String, String> getMap(CommandLine cli, Option option, String separator) {
    Map<String, String> map = new HashMap<String, String>();
    List<?> list = cli.getValues(option);
    if (list == null) {
      return map;
    }
    for (Object obj : list) {
      String[] pair = obj.toString().split("=");
      map.put(pair[0], pair[1]);
    }
    
    return map;
  }

}
