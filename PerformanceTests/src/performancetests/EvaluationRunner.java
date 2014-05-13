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
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
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
    
    Object[][] recommenders = { 
        { "Random", Random, None }, 
        { "ItemAverage", ItemAverage, None }, 
        { "ItemUserAverage", ItemUserAverage, None }, 
        { "UB Pearson", GenericUserBased, Pearson }, 
        { "UB PearsonW", GenericUserBased, PearsonW }, 
        { "UB Euclidian", GenericUserBased, Euclidian },
        { "UB EuclidianW", GenericUserBased, EuclidianW },
        { "UB Spearman", GenericUserBased, Spearman },
        { "UB Spearman", GenericUserBased, Spearman },
        { "IB Pearson", GenericItemBased, Pearson },
        { "IB PearsonW", GenericItemBased, PearsonW },
        { "IB Euclidian", GenericItemBased, Euclidian },
        { "IB EuclidianW", GenericItemBased, EuclidianW },
        { "IB Tanimoto", GenericItemBased, Tanimoto },
        { "IB LogLikelihood", GenericItemBased, LogLikelihood },
        { "SVG", SVG, None },
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

    writer.write( "data set,algorithm,run,used mem,time 1st rec.,used mem,time 2nd rec.,used mem,"
        + "av. abs. diff.,duration (sec),precision,recall,duration (sec)" ); 
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
      DataModel dataModel = new FileDataModel(new File(dataDirectory + "/" + dataSet));

      for (Object[] configuration : recommenders)
      {
        
        DynamicRecommenderBuilder recommenderBuilder = new DynamicRecommenderBuilder(configuration);
        
        System.out.println("Testing " + recommenderBuilder.name);

        writer.write(dataSet + ",");
        writer.write(recommenderBuilder.name + ",");

        for (int j=0; j<repeats[i]; j++) 
        {
          
          System.gc();
          writer.write(j + ",");
          writer.write(getUsedMemory() + ",");
          writer.flush();
          System.out.println("run " + j);
          long start = System.nanoTime();

          // do single recommendation and measure time and memory
          Recommender recommender = recommenderBuilder.buildRecommender(dataModel);
          LongPrimitiveIterator userIDs = dataModel.getUserIDs();
          Long user1 = userIDs.next();
          recommender.recommend(user1, 5);

          long split = System.nanoTime();
          long millis = (split - start) / 1000000;
          Period period = new Period(millis).normalizedStandard();
          writer.write(millis/1000.0 + ",");
          writer.write(getUsedMemory() + ",");
          writer.flush();
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
          
          // do another recommendation and measure time and memory
          Long user2 = userIDs.next();
          recommender.recommend(user2, 5);

          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.write(millis/1000.0 + ",");
          writer.write(getUsedMemory() + ",");
          writer.flush();
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
          
          // evaluate recommender
          RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
          double avAbsDif = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
          
          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.write(avAbsDif + ",");
          writer.write(millis/1000.0 + ",");
          writer.flush();
          System.out.println("AvAbsDiff: " + avAbsDif);
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));

          // calculate IR statistics
          RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
          IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, dataModel, null, 10,
              GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
          
          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.write(stats.getPrecision() + ",");
          writer.write(stats.getRecall() + ",");
          writer.write(millis/1000.0 + ",");
          System.out.println("precision: " + stats.getPrecision());
          System.out.println("recall: " + stats.getRecall());
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));

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

  static int getUsedMemory()
  {
    return (int) ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024);
  }
}