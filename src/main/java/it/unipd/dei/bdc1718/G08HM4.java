package it.unipd.dei.bdc1718;

/*
* HOMEWORK 4 BIG DATA 2017/18
* Davide Masiero
* Riccardo Castelletto
* Teo Spadotto
* */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class G08HM4 {

  public static void main(String[] args) throws FileNotFoundException {
    if (args.length != 3)
      throw new IllegalArgumentException("Expecting 3 parameters (file name, k, numBlocks)");

    //Some variables
    int k, numBlocks;

    //Take the value k and numBlocks form the command line
    try {
      k = Integer.valueOf(args[1]);
      numBlocks = Integer.valueOf(args[2]);
    }
    catch (Exception e) {
      throw new IllegalArgumentException("k and numBlocks must be integer");
    }

  }
}
