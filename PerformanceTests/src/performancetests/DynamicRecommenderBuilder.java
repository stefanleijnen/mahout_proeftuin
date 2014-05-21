package performancetests;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.*;
import org.apache.mahout.cf.taste.impl.recommender.knn.KnnItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.knn.NonNegativeQuadraticOptimizer;
import org.apache.mahout.cf.taste.impl.recommender.knn.Optimizer;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.MemoryDiffStorage;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.mortbay.log.Log;


@SuppressWarnings("deprecation")
public class DynamicRecommenderBuilder implements RecommenderBuilder
{
  enum RecommenderName { Random, ItemAverage, ItemUserAverage, GenericUserBased, GenericItemBased,
    SlopeOne, SlopeOneMem, SVG, KnnItemBased, TreeClustering }
  enum SimilarityMeasure { None, Pearson, PearsonW, Euclidian, EuclidianW, Spearman, Tanimoto,
    LogLikelihood }

  String name;
  RecommenderName recommenderName;
  SimilarityMeasure similarityMeasure;
  double nearestN = -1; // if smaller than 1, used as threshold for ThresholdUserNeighborhood
  
  DynamicRecommenderBuilder(Object[] conf) 
  {
    name = (String) conf[0];
    recommenderName = (RecommenderName) conf[1];
    similarityMeasure = (SimilarityMeasure) conf[2];
    if (conf.length > 3)
      if (conf[3] instanceof Integer)
        nearestN = ((Integer)conf[3]).doubleValue();
      else
        nearestN = (double) conf[3];
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
    
    UserNeighborhood userNeighborhood = null;
    if (nearestN != -1) {
      if (nearestN < 1) {
        Log.info("using ThresholdUserNeighborhood with threshold " + nearestN);
        userNeighborhood = new ThresholdUserNeighborhood(nearestN, similarity, dataModel);
      }
      else {
        Log.info("using NearestNUserNeighborhood with N " + nearestN);
        userNeighborhood = new NearestNUserNeighborhood((int) nearestN, similarity, dataModel);
      }
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
        similarity = new CachingUserSimilarity(similarity, dataModel);
        if (userNeighborhood == null)
          throw new RuntimeException("UserNeighborhood should be defined with GenericUserBasedRecommender");
        recommender = new GenericUserBasedRecommender(dataModel, userNeighborhood, similarity);
        break;
      case GenericItemBased:
        ItemSimilarity iSimilarity = new CachingItemSimilarity((ItemSimilarity) similarity, dataModel);
        recommender = new GenericItemBasedRecommender(dataModel, iSimilarity);
        break;
//    not in Mahout 0.9
      case SlopeOne:
        recommender = new SlopeOneRecommender(dataModel);
        break;
//    not in Mahout 0.9
      case SlopeOneMem:
        DiffStorage diffStorage = new MemoryDiffStorage(dataModel, Weighting.WEIGHTED, 10000000L);
        recommender = new SlopeOneRecommender(dataModel, Weighting.WEIGHTED, Weighting.WEIGHTED, diffStorage);
        break;
      case SVG:
        recommender = new SVDRecommender(dataModel, new ALSWRFactorizer(dataModel, 10, 0.05, 10));
        break;
//    not in Mahout 0.9
      case KnnItemBased:
        Optimizer optimizer = new NonNegativeQuadraticOptimizer();
        recommender = new KnnItemBasedRecommender(dataModel, (ItemSimilarity) similarity, optimizer, 10);
        break;
//    not in Mahout 0.9
      case TreeClustering:
        ClusterSimilarity clusterSimilarity = new FarthestNeighborClusterSimilarity(similarity);
        recommender = new TreeClusteringRecommender(dataModel, clusterSimilarity, 10);
        break;
      default:
        throw new RuntimeException("No recommender measure set.");
    }
    return recommender;
  };

}
