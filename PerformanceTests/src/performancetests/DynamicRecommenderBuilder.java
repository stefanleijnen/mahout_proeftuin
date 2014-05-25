package performancetests;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.example.kddcup.track1.svd.ParallelArraysSGDFactorizer;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.*;
import org.apache.mahout.cf.taste.impl.recommender.knn.KnnItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.knn.NonNegativeQuadraticOptimizer;
import org.apache.mahout.cf.taste.impl.recommender.knn.Optimizer;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.MemoryDiffStorage;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.*;
import org.apache.mahout.cf.taste.impl.similarity.*;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@SuppressWarnings("deprecation")
public class DynamicRecommenderBuilder implements RecommenderBuilder
{
  static Logger logger = LoggerFactory.getLogger(DynamicRecommenderBuilder.class);

  enum RecommenderName { Random, ItemAverage, ItemUserAverage, GenericUserBased, GenericItemBased,
    BiasedItemBased, SlopeOne, SlopeOneMem, 
    SVD_ALS, SVD_FUNK, SVD_ILR, SVD_PSGD, SVD_RSGD, SVD_PlusPlus,  
    KnnItemBased, TreeClustering, TreeClustering2,
    BookCrossing, KddCupTrack1 } // recommenders from Mahout examples. 
  // Note that KddCupTrack2 is too complex to include here.
  enum SimilarityMeasure { None, Pearson, PearsonW, UncenteredCosine, Euclidian, EuclidianW, 
    Spearman, Tanimoto, LogLikelihood }

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
      case UncenteredCosine:
        similarity = new UncenteredCosineSimilarity(dataModel);
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
        logger.info("using ThresholdUserNeighborhood with threshold " + nearestN);
        userNeighborhood = new ThresholdUserNeighborhood(nearestN, similarity, dataModel);
      }
      else {
        logger.info("using NearestNUserNeighborhood with N " + nearestN);
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
          throw new RuntimeException("UserNeighborhood should be defined when using "
              + "GenericUserBasedRecommender");
        recommender = new GenericUserBasedRecommender(dataModel, userNeighborhood, similarity);
        break;
      case GenericItemBased:
        ItemSimilarity iSimilarity = new CachingItemSimilarity((ItemSimilarity) similarity, 
            dataModel);
        recommender = new GenericItemBasedRecommender(dataModel, iSimilarity);
        break;
      case BiasedItemBased:
        ItemSimilarity iSimilarity2 = new CachingItemSimilarity((ItemSimilarity) similarity, 
          dataModel);
        recommender = new BiasedItemBasedRecommender(dataModel, iSimilarity2);
        break;
      case SlopeOne: // not in Mahout 0.9
        recommender = new SlopeOneRecommender(dataModel);
        break;
      case SlopeOneMem: // not in Mahout 0.9
        DiffStorage diffStorage = new MemoryDiffStorage(dataModel, Weighting.WEIGHTED, 10000000L);
        recommender = new SlopeOneRecommender(dataModel, Weighting.WEIGHTED, Weighting.WEIGHTED, 
            diffStorage);
        break;
      case SVD_ALS:
        recommender = new SVDRecommender(dataModel, new ALSWRFactorizer(dataModel, 10, 0.05, 10));
        break;
      case SVD_FUNK: // not in Mahout 0.9
        recommender = new SVDRecommender(dataModel, new FunkSVDFactorizer(dataModel, 10, 10));
        break;
      case SVD_ILR: // not in Mahout 0.9
        recommender = new SVDRecommender(dataModel, new ImplicitLinearRegressionFactorizer(
          dataModel, 10, 10, 0.1));
        break;
      case SVD_PlusPlus: // not in Mahout 0.9
        recommender = new SVDRecommender(dataModel, new SVDPlusPlusFactorizer(dataModel, 10, 10));
        break;
      case SVD_PSGD: // not in Mahout 0.9
        recommender = new SVDRecommender(dataModel, new ParallelArraysSGDFactorizer(
          dataModel, 10, 10));
        break;
      case SVD_RSGD: // not in Mahout 0.9
        recommender = new SVDRecommender(dataModel, new RatingSGDFactorizer(dataModel, 10, 10));
        break;
      case KnnItemBased: // not in Mahout 0.9
        Optimizer optimizer = new NonNegativeQuadraticOptimizer();
        recommender = new KnnItemBasedRecommender(dataModel, (ItemSimilarity) similarity, optimizer,
            10);
        break;
      case TreeClustering: // not in Mahout 0.9
        ClusterSimilarity clusterSimilarity = new FarthestNeighborClusterSimilarity(similarity);
        recommender = new TreeClusteringRecommender(dataModel, clusterSimilarity, 10);
        break;
      case TreeClustering2: // not in Mahout 0.9
        ClusterSimilarity clusterSimilarity2 = new FarthestNeighborClusterSimilarity(similarity);
        recommender = new TreeClusteringRecommender2(dataModel, clusterSimilarity2, 10);
        break;
      case BookCrossing:
        similarity = new CachingUserSimilarity(similarity, dataModel);
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, 0.2, similarity, dataModel,
            0.2);
        recommender = new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
        break;
      case KddCupTrack1:
        recommender = new GenericItemBasedRecommender(dataModel, (ItemSimilarity) similarity);
        break;
      default:
        throw new RuntimeException("No recommender measure set.");
    }
    return recommender;
  };

}
