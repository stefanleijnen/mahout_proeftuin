package userbasedrecommender;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.ItemUserAverageRecommender;
import org.apache.mahout.cf.taste.impl.similarity.CachingUserSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.joda.time.Period;
import org.joda.time.format.PeriodFormat;

// Results on GJ machine
// bitness / Algorithm / dataset / AbsAvDiff, time / precision, recall, time
// 64bit / Random / 100k / 1.38, 1 sec
// 64bit / ItemAv / 100k / 0.81, 1 sec
// 64bit / ItemUAv / 100k / 0.75, 1 sec / 0.00013, 0.00018, 2min16
// 64bit / GUBR+Pearson / 100k / 0.78, 12 sec / 0.013, 0.015, 1min34
// 64bit / GUBR+PearsonW / 100k / 0.79, 13 sec / 0.013, 0.015, 1min36
// 64bit / GUBR+Euclidian / 100k / 0.80, 13 sec / 0.0073, 0.0094, 2min54
// 64bit / GUBR+EuclidianW / 100k / 0.81, 13 sec
// 32bit / GUBR+Spearman / 100k / 0.80, 13 min
// 64bit / GUBR+Spearman / 100k / 0.80, 3 min
// 64bit / GUBR+Tanimoto / 100k / 0.81, 50 sec
// 64bit / GUBR+LogLikelh / 100k / 0.81, 50 sec

public class EvaluateSingleRecommender
{
  void runHoldOutTest() throws IOException, TasteException
  {
    long start = System.nanoTime();

    DataModel model = new FileDataModel(new File("data/ml-100k.csv"));
    RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
    RecommenderBuilder builder = new MyRecommenderBuilder();
    double result = evaluator.evaluate(builder, null, model, 0.9, 1.0);
    
    System.out.println(result);
    long finish = System.nanoTime();
    long millis = (finish - start) / 1000000;
    Period period = new Period(millis).normalizedStandard();
    System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
    
    RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
    IRStatistics stats = statsEvaluator.evaluate(builder, null, model, null, 10,
        GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
    
    System.out.println("precision: " + stats.getPrecision());
    System.out.println("recall: " + stats.getRecall());
    finish = System.nanoTime();
    millis = (finish - start) / 1000000;
    period = new Period(millis).normalizedStandard();
    System.out.println("Duration: " + PeriodFormat.getDefault().print(period));
  }

  class MyRecommenderBuilder implements RecommenderBuilder
  {
    @Override
    public Recommender buildRecommender(DataModel dataModel) throws TasteException
    {
      UserSimilarity similarity = new CachingUserSimilarity(
//          new PearsonCorrelationSimilarity(dataModel, Weighting.UNWEIGHTED), dataModel
          new EuclideanDistanceSimilarity(dataModel, Weighting.UNWEIGHTED), dataModel
//          new SpearmanCorrelationSimilarity(dataModel), dataModel
//          new TanimotoCoefficientSimilarity(dataModel), dataModel
//          new LogLikelihoodSimilarity(dataModel), dataModel
      );
      UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
//      return new RandomRecommender(dataModel);
//      return new ItemAverageRecommender(dataModel);
//      return new ItemUserAverageRecommender(dataModel);
      return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
//      return new SlopeOneRecommender(dataModel); // Doesn't exist anymore in Mahout. I searched hard but couldn't find any discussion on this, except that SlopeOne was one of a number of "unused" algorithms that was first deprecated and then removed
    }
  }

  public static void main(String[] args) throws IOException, TasteException
  {
    new EvaluateSingleRecommender().runHoldOutTest();
  }
}

