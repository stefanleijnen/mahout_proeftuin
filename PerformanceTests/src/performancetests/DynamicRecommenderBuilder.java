package performancetests;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.ItemAverageRecommender;
import org.apache.mahout.cf.taste.impl.recommender.ItemUserAverageRecommender;
import org.apache.mahout.cf.taste.impl.recommender.RandomRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.SpearmanCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;


public class DynamicRecommenderBuilder implements RecommenderBuilder
{
  enum Type { UserBased, ItemBased }
  enum RecommenderName { Random, ItemAverage, ItemUserAverage, GenericUserBased, GenericItemBased }
  enum SimilarityMeasure { None, Pearson, PearsonW, Euclidian, EuclidianW, Spearman, Tanimoto,
    LogLikelihood }

  String name;
  RecommenderName recommenderName;
  SimilarityMeasure similarityMeasure;
  
  DynamicRecommenderBuilder(Object[] conf) 
  {
    name = (String) conf[0];
    recommenderName = (RecommenderName) conf[1];
    similarityMeasure = (SimilarityMeasure) conf[2];
  }

  @Override
  public Recommender buildRecommender(DataModel dataModel) throws TasteException
  {
    UserSimilarity similarity; 
    switch (similarityMeasure) {
      case None: 
        similarity = null;
        break;
      case Pearson:
        similarity = new PearsonCorrelationSimilarity(dataModel);
        break;
      case PearsonW:
        similarity = new PearsonCorrelationSimilarity(dataModel, Weighting.WEIGHTED);
        break;
      case Euclidian:
        similarity = new EuclideanDistanceSimilarity(dataModel);
        break;
      case EuclidianW:
        similarity = new EuclideanDistanceSimilarity(dataModel, Weighting.WEIGHTED);
        break;
      case Spearman:
        similarity = new SpearmanCorrelationSimilarity(dataModel);
        break;
      case Tanimoto:
        similarity = new TanimotoCoefficientSimilarity(dataModel);
        break;
      case LogLikelihood:
        similarity = new LogLikelihoodSimilarity(dataModel);
        break;
      default:
        throw new RuntimeException("No similarity measure set.");
    }    
    
    Recommender recommender;
    switch (recommenderName) {
      case Random:
        recommender = new RandomRecommender(dataModel);
        break;
      case ItemAverage:
        recommender = new ItemAverageRecommender(dataModel);
        break;
      case ItemUserAverage:
        recommender = new ItemUserAverageRecommender(dataModel);
        break;
      case GenericUserBased:
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dataModel);
        recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
        break;
      case GenericItemBased:
        recommender = new GenericItemBasedRecommender(dataModel, (ItemSimilarity) similarity);
        break;
      default: 
        throw new RuntimeException("No recommender measure set.");
    }
    return recommender;
  };

}
