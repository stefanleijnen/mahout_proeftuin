package performancetests;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.util.Date;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;


public class HoldOutTestRunner
{

  public static void main(String[] args) throws IOException, TasteException
  {
    String dataDirectory = "data";
    String[] dataSets = {"dataset32.csv", "ml-100k.csv", "ml-1M.csv", "ml-10M.csv"};
    int[] repeats = { 1, 1, 1, 1 };
    String today = DateFormat.getDateTimeInstance().format(new Date()).replace(':', '_');
    BufferedWriter writer = new BufferedWriter(new FileWriter(
      "results/holdout-test-results-" + today + ".csv"));

    writer.write( "Java version: " + System.getProperty("java.version") );
    writer.newLine();
    writer.write( "JVM version: " + System.getProperty("java.vm.version") );
    writer.newLine();
    writer.write( "Bitness: " + System.getProperty("sun.arch.data.model") );
    writer.newLine();
    writer.write( "Heap space: " + (Runtime.getRuntime().maxMemory()/1024/1024) + " Mb" );
    writer.newLine();
    writer.newLine();

    writer.write( "data set" ); 
    writer.write( "," ); 
    writer.newLine();
    writer.flush();

    for (int i=0; i<dataSets.length; i++) 
    {
      String dataSet = dataSets[i];
      System.out.println();
      System.out.println("Using data set: " + dataSet);
      System.out.println();
      writer.write(dataSet + ",");
      System.gc();
      DataModel model = new FileDataModel(new File(dataDirectory + "/" + dataSet));

      {
        System.out.println("Testing GenericUserBasedRecommender");
        writer.write("GenericUserBasedRecommender,");
        long start = System.nanoTime();

        for (int j=0; j<repeats[i]; j++) 
        {
          System.out.println("run " + j);
          RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
          RecommenderBuilder recommenderBuilder = new GenericUserBasedRecommenderBuilder();
          double result = evaluator.evaluate(recommenderBuilder, null, model, 0.9, 1.0);
          System.out.println("AvAbsDiff: " + result);
          writer.write(result + ",");
          writer.flush();
          RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
          IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, model, null, 10,
              GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
          System.out.println("precision: " + stats.getPrecision());
          System.out.println("recall: " + stats.getRecall());

        }

        long finish = System.nanoTime();
        long millis = (finish - start) / 1000000;
        Period period = new Period(millis).normalizedStandard();
        System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
        writer.write("," + millis/1000.0 + ",");
        writer.write(repeats[i] + ",");
        writer.write(millis/1000.0/repeats[i] + ",");
        writer.flush();
      }
      
      writer.newLine();

      {
        System.out.println("Testing GenericItemBasedRecommender");
        writer.write(",GenericItemBasedRecommender,");
        long start = System.nanoTime();

        for (int j=0; j<repeats[i]; j++) 
        {
          RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
          RecommenderBuilder builder = new GenericItemBasedRecommenderBuilder();
          double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
          System.out.println(result);
          writer.write(result + ",");
          writer.flush();
        }

        long finish = System.nanoTime();
        long millis = (finish - start) / 1000000;
        Period period = new Period(millis).normalizedStandard();
        System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
        writer.write(millis/1000.0 + ",");
        writer.flush();
      }
      writer.newLine();
    }
    writer.newLine();
    writer.write("Done.");
    writer.newLine();
    writer.close();
  }

}

class GenericUserBasedRecommenderBuilder implements RecommenderBuilder
{
  @Override
  public Recommender buildRecommender(DataModel dataModel) throws TasteException
  {
    UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
    UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
    return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
  }
}

class GenericItemBasedRecommenderBuilder implements RecommenderBuilder
{
  @Override
  public Recommender buildRecommender(DataModel dataModel) throws TasteException
  {
    TanimotoCoefficientSimilarity sim = new TanimotoCoefficientSimilarity(dataModel);
    return new GenericItemBasedRecommender(dataModel, sim);
  }
}