package performancetests;

import static performancetests.DynamicRecommenderBuilder.RecommenderName.*;
import static performancetests.DynamicRecommenderBuilder.SimilarityMeasure.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.util.Date;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.LoadEvaluator;
import org.apache.mahout.cf.taste.impl.eval.LoadStatistics;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * This class evaluates recommendation algorithms (currently two, hard coded).
 * Evaluation metrics are average absolute difference, precision and recall.
 */
public class EvaluationRunner
{
  static Logger logger = LoggerFactory.getLogger(EvaluationRunner.class);
  
  public static void main(String[] args) throws IOException, TasteException
  {
    String dataDirectory = "data";
    String[] dataSets = {"dataset32.csv", "ml-10k.csv", "ml-100k.csv", "ml-1M.csv", "ml-10M.csv"};
    int[] repeats = { 1, 1, 1, 1 };
    
    Object[][] recommenders = { 
        { "Random", Random, None }, 
        { "ItemAverage", ItemAverage, None }, 
        { "ItemUserAverage", ItemUserAverage, None }, 
        { "UB Pearson 1nn", GenericUserBased, Pearson, 1 }, 
        { "UB Pearson 2nn", GenericUserBased, Pearson, 2 }, 
        { "UB Pearson 4nn", GenericUserBased, Pearson, 4 }, 
        { "UB Pearson th.8", GenericUserBased, Pearson, .8 }, 
        { "UB Pearson th.9", GenericUserBased, Pearson, .9 }, 
        { "UB Pearson th.95", GenericUserBased, Pearson, .95 }, 
        { "UB PearsonW 1nn", GenericUserBased, PearsonW, 1 }, 
        { "UB PearsonW 2nn", GenericUserBased, PearsonW, 2 }, 
        { "UB PearsonW 4nn", GenericUserBased, PearsonW, 4 }, 
        { "UB PearsonW th.8", GenericUserBased, PearsonW, .8 }, 
        { "UB PearsonW th.9", GenericUserBased, PearsonW, .9 }, 
        { "UB PearsonW th.95", GenericUserBased, PearsonW, .95 }, 
        { "UB Euclidian 1nn", GenericUserBased, Euclidian, 1 }, 
        { "UB Euclidian 2nn", GenericUserBased, Euclidian, 2 }, 
        { "UB Euclidian 4nn", GenericUserBased, Euclidian, 4 }, 
        { "UB Euclidian th.8", GenericUserBased, Euclidian, .8 }, 
        { "UB Euclidian th.9", GenericUserBased, Euclidian, .9 }, 
        { "UB Euclidian th.95", GenericUserBased, Euclidian, .95 }, 
        { "UB EuclidianW 1nn", GenericUserBased, EuclidianW, 1 }, 
        { "UB EuclidianW 2nn", GenericUserBased, EuclidianW, 2 }, 
        { "UB EuclidianW 4nn", GenericUserBased, EuclidianW, 4 }, 
        { "UB EuclidianW th.8", GenericUserBased, EuclidianW, .8 }, 
        { "UB EuclidianW th.9", GenericUserBased, EuclidianW, .9 }, 
        { "UB EuclidianW th.95", GenericUserBased, EuclidianW, .95 }, 
        { "UB Spearman", GenericUserBased, Spearman, .1 },
        { "IB Pearson", GenericItemBased, Pearson },
        { "IB PearsonW", GenericItemBased, PearsonW },
        { "IB Euclidian", GenericItemBased, Euclidian },
        { "IB EuclidianW", GenericItemBased, EuclidianW },
        { "IB Tanimoto", GenericItemBased, Tanimoto },
        { "IB LogLikelihood", GenericItemBased, LogLikelihood },
        { "SlopeOne", SlopeOne, None }, // not in Mahout 0.9
        { "SlopeOneMem", SlopeOneMem, None }, // not in Mahout 0.9
        { "SVG", SVG, None },
        { "KnnItemBased", KnnItemBased, LogLikelihood }, // not in Mahout 0.9
        { "TreeClustering", TreeClustering, LogLikelihood }, // not in Mahout 0.9
    };
    
    String today = DateFormat.getDateTimeInstance().format(new Date()).replace(':', '_');
    // using Printwriter for easy formatting and auto-flushing
    PrintWriter writer = new PrintWriter(
      new FileOutputStream("results/evaluation-results-" + today + ".csv"), true);

    writer.println( "Java version," + System.getProperty("java.version") );
    writer.println( "JVM version," + System.getProperty("java.vm.version") );
    writer.println( "Bitness," + System.getProperty("sun.arch.data.model") );
    writer.println( "Max heap," + (Runtime.getRuntime().maxMemory()/1024/1024) + " Mb" );
    writer.println();

    writer.println("data set,algorithm,run,mem,1st rec (s),mem,2nd rec (s),mem,"
        + "avg time,ct,"
        + "av abs dif,dur (s),prec,recall,fallout,F1 msr,nDCG,dur (s)");

    for (int i=0; i<dataSets.length; i++) 
    {
      
      String dataSet = dataSets[i];
      System.out.println();
      System.out.println("Using data set: " + dataSet);
      System.out.println();
      DataModel dataModel = new FileDataModel(new File(dataDirectory + "/" + dataSet));

      for (Object[] configuration : recommenders)
      {
        
        DynamicRecommenderBuilder recommenderBuilder = new DynamicRecommenderBuilder(configuration);
        
        System.out.println();
        System.out.println("Testing " + recommenderBuilder.name);
        logger.info("Testing " + recommenderBuilder.name);

        writer.printf(dataSet + ",");
        writer.printf(recommenderBuilder.name + ",");

        for (int j=0; j<repeats[i]; j++) 
        {
          
          System.gc();
          
          writer.printf(j + ",");
          writer.printf(getUsedMemory() + ",");
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
          writer.printf("%.3f,", millis/1000.0);
          writer.printf("%d,", getUsedMemory());
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
          
          // do another recommendation and measure time and memory
          Long user2 = userIDs.next();
          recommender.recommend(user2, 5);

          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.printf("%.3f,", millis/1000.0);
          writer.printf("%d,", getUsedMemory());
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
          
          // run load evaluator
          LoadStatistics loadStats = LoadEvaluator.runLoad(recommender);
          RunningAverage timing = loadStats.getTiming();
          System.out.println("LoadEvaluator: av: " + timing.getAverage());
          System.out.println("LoadEvaluator: ct: " + timing.getCount());
          writer.printf("%.2f,", timing.getAverage());
          writer.printf("%d,", timing.getCount());

          // evaluate recommender
          RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
          double avAbsDif = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.9, 1.0);
          
          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.printf("%.2f,", avAbsDif);
          writer.printf("%f,", millis/1000.0);
          System.out.println("AvAbsDiff: " + avAbsDif);
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));

          // calculate IR statistics
          RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
          IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, dataModel, null, 
              10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
          
          split = System.nanoTime();
          millis = (split - start) / 1000000;
          period = new Period(millis).normalizedStandard();
          writer.printf("%.3f,", stats.getPrecision());
          writer.printf("%.3f,", stats.getRecall());
          writer.printf("%.4f,", stats.getFallOut());
          writer.printf("%.3f,", stats.getF1Measure());
          writer.printf("%.3f,", stats.getNormalizedDiscountedCumulativeGain());
          writer.printf("%.2f,", millis/1000.0);
          System.out.println("precision: " + stats.getPrecision());
          System.out.println("recall: " + stats.getRecall());
          System.out.println("fallout: " + stats.getFallOut());
          System.out.println("F1 measure: " + stats.getF1Measure());
          System.out.println("nDCG: " + stats.getNormalizedDiscountedCumulativeGain());
          System.out.println("Duration: " + PeriodFormat.getDefault().print(period));

          writer.println();
        }
      }
      writer.println();
    }
    writer.println();
    writer.println("Done.");
    writer.close();
  }

  static int getUsedMemory()
  {
    return (int) ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024);
  }
}