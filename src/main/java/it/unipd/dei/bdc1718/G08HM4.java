package it.unipd.dei.bdc1718;

/*
* HOMEWORK 4 BIG DATA 2017/18
* Davide Masiero
* Riccardo Castelletto
* Teo Spadotto
* */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

public class G08HM4 {

  public static void main(String[] args) throws Exception {
    if (args.length != 3)
      throw new IllegalArgumentException("Expecting 3 parameters (file name, k, numBlocks)");

    //Some variables
    int k, numBlocks;
    long start, end;

    //Take the value k and numBlocks form the command line
    try {
      k = Integer.valueOf(args[1]);
      numBlocks = Integer.valueOf(args[2]);
    }
    catch (Exception e) {
      throw new IllegalArgumentException("k and numBlocks must be integer");
    }

    //Creation of the Spark Configuration
    SparkConf configuration = new SparkConf(true);
    configuration.setAppName("Homework 4");

    //Now we can create the Spark Context
    JavaSparkContext sc = new JavaSparkContext(configuration);

    //Creation of the JavaRDD from the text file passed in input
    JavaRDD<Vector> points = sc.textFile(args[0]).map(G08HM4::strToVector).repartition(numBlocks).cache();

    points.count();

    ArrayList<Vector> x = runMapReduce(points, k, numBlocks);
    System.out.println(x.size());
    for (Vector v : x) {
      System.out.println(v);
    }

    System.in.read();
  }

  //runMapReduce method
  private static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsrdd, int k, int numBlocks){
    /*JavaPairRDD<Long, Vector> y = pointsrdd.mapToPair((z) -> {
      return new Tuple2<>((long) (Math.random() * pointsrdd.count()/numBlocks), z);
    });*/


    ArrayList<Vector> coreset = pointsrdd.mapToPair((z) -> {
      return new Tuple2<>((long) (Math.random() * numBlocks), z);
    }).groupByKey().mapValues((its) ->{
      ArrayList<Vector> list = new ArrayList<>(0);
      for (Vector it : its) {
        list.add(it);
      }
      return  list;
    }).mapToPair((z) -> {
      ArrayList<Vector> centers = kcenter(z._2(), k);
      return new Tuple2<>(0L, centers);
    }).groupByKey().mapValues((its) -> {
      ArrayList<Vector> list = new ArrayList<>(0);
      for (ArrayList<Vector> it : its) {
        for (Vector v : it) {
          list.add(v);
        }
      }
      return list;
    }).map((list) -> (list._2())).take(1).get(0);
/*
    ArrayList<Vector> y = pointsrdd.foreachPartition((its) -> {
      ArrayList<Vector> list = new ArrayList<>();
      for (Vector it : its) {
        list.add(it);
      }
      return list;
    }).map((list) -> {
      ArrayList<Vector> centers = kcenter(list, k);
      return centers;
    })*/

    return coreset;
  }

  //Function for converting string to vector (The same as the Homework 3)
  private static Vector strToVector(String str) {
    String[] tokens = str.split(" ");
    double[] data = new double[tokens.length];
    for (int i=0; i<tokens.length; i++) {
      data[i] = Double.parseDouble(tokens[i]);
    }
    return Vectors.dense(data);
  }

  //Method Farthest-First-Traversal for k-center with time complexity O( |P| * k )
  private static ArrayList<Vector> kcenter(ArrayList<Vector> P, int k) {
    ArrayList<Vector> centers = new ArrayList<>();
    //storing the centers
    Vector max = Vectors.zeros(1);
    //storing the various distance
    ArrayList<Double> dist = new ArrayList<>();
    double distance = 0.0;
    //storing the indices of the center
    int index = 0;
    //first center decided at random
    int r = (int)(Math.random() * P.size());
    //add it to the set S of centers and remove it
    centers.add(P.remove(r));
    for (int i = 0; i < k - 1; i++) {               //k-1 because the first center is yet determined
      for (int j = 0; j < P.size(); j++) {
        if (i==0) {                             //first round i have to compute the distance to the center for every point
          dist.add(Vectors.sqdist(centers.get(i), P.get(j)));
          if (dist.get(j) > distance) { //find the farther point
            max = P.get(j);
            distance = dist.get(j);
            index = j;
          }
        }
        else {
          if (dist.get(j) > Vectors.sqdist(centers.get(i), P.get(j))) {       //find the closest center from every point of P
            dist.set(j, Vectors.sqdist(centers.get(i), P.get(j)));
          }
          if (dist.get(j) > distance) {                                       //find the value that maximize d(Ci,S)
            max = P.get(j);
            distance = dist.get(j);
            index = j;
          }
        }
      }//end for P
      centers.add(max);         //once i found the centers we add it to the set S of centers
      P.remove(index);          //and we can remove it from the set of point P
      dist.remove(index);       //we remove also the value of the distance to maintain the accuracy of the array
      distance=0.0;             //reset the distance
    }//fine for k
    return centers;
  }

  //Sequential approximation algorithm based on matching.
  public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {
    final int n = points.size();
    if (k >= n) {
      return points;
    }

    ArrayList<Vector> result = new ArrayList<>(k);
    boolean[] candidates = new boolean[n];
    Arrays.fill(candidates, true);
    for (int iter=0; iter<k/2; iter++) {
      // Find the maximum distance pair among the candidates
      double maxDist = 0;
      int maxI = 0;
      int maxJ = 0;
      for (int i = 0; i < n; i++) {
        if (candidates[i]) {
          for (int j = i+1; j < n; j++) {
            if (candidates[j]) {
              double d = Math.sqrt(Vectors.sqdist(points.get(i), points.get(j)));
              if (d > maxDist) {
                maxDist = d;
                maxI = i;
                maxJ = j;
              }
            }
          }
        }
      }
      // Add the points maximizing the distance to the solution
      result.add(points.get(maxI));
      result.add(points.get(maxJ));
      // Remove them from the set of candidates
      candidates[maxI] = false;
      candidates[maxJ] = false;
    }
    // Add an arbitrary point to the solution, if k is odd.
    if (k % 2 != 0) {
      for (int i = 0; i < n; i++) {
        if (candidates[i]) {
          result.add(points.get(i));
          break;
        }
      }
    }
    if (result.size() != k) {
      throw new IllegalStateException("Result of the wrong size");
    }
    return result;
  }

}
