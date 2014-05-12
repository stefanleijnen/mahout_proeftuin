package performancetests;

import static performancetests.DynamicRecommenderBuilder.RecommenderName.*;
import static performancetests.DynamicRecommenderBuilder.SimilarityMeasure.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.util.Date;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;



/**
 * This class evaluates recommendation algorithms (currently two, hard coded).
 * Evaluation metrics are average absolute difference, precision and recall.
 */
public class EvaluationRunner
{

  public static void main(String[] args) throws IOException, TasteException
  {
    String dataDirectory = "data";
    String[] dataSets = {"dataset32.csv", "ml-10k.csv", "ml-100k.csv", "ml-1M.csv", "ml-10M.csv"};
    int[] repeats = { 1, 1, 1, 1 };
//    RecommenderBuilder[] recommenderBuilders = {new UserBasedPearson(), new ItemBasedTanimoto()};
    
    Object[][] recommenders = { 
        { "Random", Random, None }, 
        { "ItemAverage", ItemAverage, None }, 
        { "ItemUserAverage", ItemUserAverage, None }, 
        { "UB Pearson", GenericUserBased, Pearson }, 
        { "UB PearsonW", GenericUserBased, PearsonW }, 
        { "UB Euclidian", GenericUserBased, Euclidian },
        { "UB EuclidianW", GenericUserBased, EuclidianW },
        { "UB Spearman", GenericUserBased, Spearman },
        { "IB Pearson", GenericItemBased, Pearson },
        { "IB PearsonW", GenericItemBased, PearsonW },
        { "IB Euclidian", GenericItemBased, Euclidian },
        { "IB EuclidianW", GenericItemBased, EuclidianW },
        { "IB Tanimoto", GenericItemBased, Tanimoto },
        { "IB LogLikelihood", GenericItemBased, LogLikelihood },
    };
    
    String today = DateFormat.getDateTimeInstance().format(new Date()).replace(':', '_');
    BufferedWriter writer = new BufferedWriter(new FileWriter(
      "results/evaluation-results-" + today + ".csv"));

    writer.write( "Java version: " + System.getProperty("java.version") );
    writer.newLine();
    writer.write( "JVM version: " + System.getProperty("java.vm.version") );
    writer.newLine();
    writer.write( "Bitness: " + System.getProperty("sun.arch.data.model") );
    writer.newLine();
    writer.write( "Heap space: " + (Runtime.getRuntime().maxMemory()/1024/1024) + " Mb" );
    writer.newLine();
    writer.newLine();

    writer.write( "max memory (mb),data set,algorithm,run,av. abs. diff.,duration (sec),precision,recall,duration (sec)" ); 
    writer.write( "," ); 
    writer.newLine();
    writer.flush();

    for (int i=0; i<dataSets.length; i++) 
    {
      
      String dataSet = dataSets[i];
      System.out.println();
      System.out.println("Using data set: " + dataSet);
      System.out.println();
      System.gc();
      DataModel model = new FileDataModel(new File(dataDirectory + "/" + dataSet));

//      for (RecommenderBuilder recommenderBuilder : recommenderBuilders)
      for (Object[] configuration : recommenders)
      {
        
        DynamicRecommenderBuilder recommenderBuilder = new DynamicRecommenderBuilder(configuration);
        
//        System.out.println("Testing " + recommenderBuilder.getClass().getSimpleName());
        System.out.println("Testing " + recommenderBuilder.name);

        for (int j=0; j<repeats[i]; j++) 
        {
          
          System.gc();
          long start = System.nanoTime();
          System.out.println("run " + j);
          
          RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
//          RecommenderBuilder recommenderBuilder = new UserBasedPearson();
          double avAbsDif = evaluator.evaluate(recommenderBuilder, null, model, 0.9, 1.0);
          
          System.out.println("AvAbsDiff: " + avAbsDif);
          long lapse1 = System.nanoTime();
          
          RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
          IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, model, null, 10,
              GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
          
          System.out.println("precision: " + stats.getPrecision());
          System.out.println("recall: " + stats.getRecall());
          
          long lapse2 = System.nanoTime();
          long millis1 = (lapse1 - start) / 1000000;
          long millis2 = (lapse2 - lapse1) / 1000000;
          Period period1 = new Period(millis1).normalizedStandard();
          Period period2 = new Period(millis2).normalizedStandard();
          System.out.println("Duration1: " + PeriodFormat.getDefault().print(period1));
          System.out.println("Duration2: " + PeriodFormat.getDefault().print(period2));
          
          writer.write((int)(Runtime.getRuntime().maxMemory()/1024/1024) + ",");
          writer.write(dataSet + ",");
          writer.write(recommenderBuilder.name + ",");
          writer.write(j + ",");
          writer.write(avAbsDif + ",");
          writer.write(millis1/1000.0 + ",");
//          writer.write(millis1/1000.0/repeats[i] + ",");
          writer.write(stats.getPrecision() + ",");
          writer.write(stats.getRecall() + ",");
          writer.write(millis2/1000.0 + ",");
          writer.newLine();
          writer.flush();
        }
      }
    }
    writer.newLine();
    writer.write("Done.");
    writer.newLine();
    writer.close();
  }

}