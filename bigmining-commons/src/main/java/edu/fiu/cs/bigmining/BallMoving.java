package edu.fiu.cs.bigmining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BallMoving {

  public static String[] getSteps(int speed, String chamber) {
    List<String> steps = new ArrayList<String>();
    
    // generate the inital chamber
    int len = chamber.length();
    char[] initChamber = new char[len];
    Arrays.fill(initChamber, '.');
    
    int[] directions = new int[len];    // -1: Left, 1: Right
    int[] locations = new int[len];     // -1 denotes not on the board
    Arrays.fill(locations, -1);
    
    int[] count = new int[] {0};
    for (int i = 0; i < len; ++i) {
      if (chamber.charAt(i) == 'L') {
        directions[i] = -1;
        initChamber[i] = 'X';
        ++count[0];
      } else if (chamber.charAt(i) == 'R') {
        directions[i] = 1;
        initChamber[i] = 'X';
        ++count[0];
      }
      if (chamber.charAt(i) != '.') {
        locations[i] = i;
      }
    }
    
    steps.add(new String(initChamber));
    while (count[0] != 0) {
      String str = oneMove(speed, count, locations, directions);
      steps.add(str);
    }
    
    String[] res = new String[steps.size()];
    steps.toArray(res);
    
    return res;
  }
  
  /**
   * Generate a string with one move.
   * 
   * @return
   */
  private static String oneMove(int speed, int[] count, int[] locations, int[] directions) {
    int newCount = 0;

    char[] newBoard = new char[locations.length];
    Arrays.fill(newBoard, '.');

    for (int i = 0; i < locations.length; ++i) {
      if (locations[i] >= 0 && locations[i] < locations.length) { // still on
                                                                  // the chamber
        locations[i] += directions[i] * speed;
      }
      if (locations[i] >= 0 && locations[i] < locations.length) {
        newBoard[locations[i]] = 'X';
        ++newCount;
      }
    }
    count[0] = newCount;

    return new String(newBoard);
  }
  
  public static void main(String[] args) {
    int speed = 1;
    String chamber = "LRRL.LR.LRR.R.LRRL.";
    String[] res = BallMoving.getSteps(speed, chamber);
    for (String str : res) {
      System.out.println(str);
    }
  }

}