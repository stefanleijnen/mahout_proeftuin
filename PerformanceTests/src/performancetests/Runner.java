package performancetests;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.util.Date;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;


public class Runner
{

  public static void main(String[] args) throws IOException, TasteException
  {
    String dataDirectory = "data";
    String[] dataSets = {"dataset32.csv", "ml-100k.csv", "ml-1M.csv", "ml-10M.csv"};
    String today = DateFormat.getDateTimeInstance().format(new Date()).replace(':', '_');
    BufferedWriter writer = new BufferedWriter(new FileWriter("results/results" + today + ".csv"));
    
    writer.write( "Java version: " + System.getProperty("java.version") );
    writer.newLine();
    writer.write( "JVM version: " + System.getProperty("java.vm.version") );
    writer.newLine();
    writer.write( "Bitness: " + System.getProperty("sun.arch.data.model") );
    writer.newLine();
    writer.write( "Heap space: " + (Runtime.getRuntime().maxMemory()/1024/1024) + " Mb" );
    writer.newLine();
    writer.newLine();

    writer.write( "(seconds),GenericUserBased,GenericItemBased" ); 
    writer.newLine();
    writer.flush();

    for (String dataSet : dataSets) 
    {
      System.out.println();
      System.out.println("Using data set: " + dataSet);
      System.out.println();
      writer.write(dataSet + ",");
      System.gc();
      DataModel model = new FileDataModel(new File(dataDirectory + "/" + dataSet));

      {
        System.out.println("Running GenericUserBasedRecommender");
        long start = System.nanoTime();

        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
        System.out.print(".");
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
        System.out.print(".");
        UserBasedRecommender recommender =
            new GenericUserBasedRecommender(model, neighborhood, similarity);
        System.out.print(".");

        List<RecommendedItem> recommendations = recommender.recommend(2, 3);
        for (RecommendedItem recommendation : recommendations) {
//          System.out.println(recommendation);
          recommendation.getItemID();
        }
        System.out.print(".");

        long finish = System.nanoTime();
        long millis = (finish - start) / 1000000;
        Period period = new Period(millis).normalizedStandard();
        System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
        writer.write(millis/1000.0 + ",");
        writer.flush();
      }

      {
        System.out.println("Running GenericItemBasedRecommender");
        long start = System.nanoTime();

        TanimotoCoefficientSimilarity sim = new TanimotoCoefficientSimilarity(model);
        GenericItemBasedRecommender recommender = new GenericItemBasedRecommender(model, sim);

        int x=1;
        for (LongPrimitiveIterator items = model.getItemIDs(); items.hasNext();) {
          long itemId = items.nextLong();
          List<RecommendedItem> recommendations = recommender.mostSimilarItems(itemId, 5);
          for (RecommendedItem recommendation : recommendations) {
            System.out.println(itemId + "," + recommendation.getItemID() + ","
                + recommendation.getValue());
            recommendation.getItemID();
          }
          x++;
          if (x > 10) break;
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
