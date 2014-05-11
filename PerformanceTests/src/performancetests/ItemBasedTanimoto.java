package performancetests;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

class ItemBasedTanimoto implements RecommenderBuilder
{
  @Override
  public Recommender buildRecommender(DataModel dataModel) throws TasteException
  {
    TanimotoCoefficientSimilarity sim = new TanimotoCoefficientSimilarity(dataModel);
    return new GenericItemBasedRecommender(dataModel, sim);
  }
}