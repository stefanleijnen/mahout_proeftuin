package org.apache.mahout.cf.taste.impl.eval;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

public class RankBasedRecommenderEvaluatorTest
{

  private static final double EPSILON = 0;

  @Before
  public void setUp() throws Exception
  {}

  @Test
  public void testCalculateScore()
  {
    RankBasedRecommenderEvaluator evaluator = new RankBasedRecommenderEvaluator();
    evaluator.reset();
    long userId = 1;
    ArrayList<Object[]> allPrefs = new ArrayList<Object[]>();
    double score = evaluator.calculateScore(userId, allPrefs);
    
//  pref = [itemId, realPref, realRank, estPref, estRank]
    evaluator.reset();
    allPrefs.clear();
    allPrefs.add( new Object[] {1, 5f, 1, 5f, 1} );
    score = evaluator.calculateScore(userId, allPrefs);   
    assertEquals(1, score, EPSILON);
    
    evaluator.reset();
    allPrefs.clear();
    allPrefs.add( new Object[] {1, 5f, 1, 5f, 1} );
    allPrefs.add( new Object[] {1, 4f, 2, 4f, 2} );
    allPrefs.add( new Object[] {1, 3f, 3, 3f, 3} );
    score = evaluator.calculateScore(userId, allPrefs);   
    assertEquals(1, score, EPSILON);
    
    evaluator.reset();
    allPrefs.clear();
    allPrefs.add( new Object[] {1, 5f, 1, 1f, 5} );
    allPrefs.add( new Object[] {1, 4f, 2, 2f, 4} );
    allPrefs.add( new Object[] {1, 3f, 3, 3f, 3} );
    allPrefs.add( new Object[] {1, 2f, 4, 4f, 2} );
    allPrefs.add( new Object[] {1, 1f, 5, 5f, 1} );
    score = evaluator.calculateScore(userId, allPrefs);   
    assertEquals(0, score, EPSILON);
    
    evaluator.reset();
    allPrefs.clear();
    allPrefs.add( new Object[] {1, 5f, 1, 5f, 1} );
    allPrefs.add( new Object[] {1, 4f, 2, 2f, 4} );
    allPrefs.add( new Object[] {1, 3f, 3, 3f, 3} );
    allPrefs.add( new Object[] {1, 2f, 4, 4f, 2} );
    allPrefs.add( new Object[] {1, 1f, 5, 1f, 5} );
    score = evaluator.calculateScore(userId, allPrefs);   
    assertEquals(.66666666666666666, score, EPSILON);
    
  }

}
