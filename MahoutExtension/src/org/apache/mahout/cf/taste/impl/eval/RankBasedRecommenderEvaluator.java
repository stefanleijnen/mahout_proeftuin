/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.Preference;

/**
 * <p>
 * Evaluation based on ranking of recommendations. It is inspired by the Normalized Distance-based 
 * Performance Measure (NDPM) of Yao, Y.Y.: Measuring retrieval effectiveness based on user 
 * preference of documents. (1995), which was rewritten in Evaluating Recommender Systems Guy Shani 
 * and Asela Gunawardana November 2009, albeit with an error 
 * (http://math.stackexchange.com/questions/673624/given-its-definition-the-formula-seems-incorrect-normalized-distance-based-per)
 * </p><p>
 * Resulting score is 1 for a recommender that rates items such that the resulting order is correct,
 * and 0 for a recommender that rates item such that the resulting order is completely reversed. 
 * </p><p>
 * This algorithm uses a part of a users preferences as training set and a part as test set. For the
 * items in the test set, the predicted rating is requested. 
 * The test set is then sorted by real rating and ranked. Then it is sorted by predicted rating and
 * ranked. These two rankings are compared. For an item, the penalty is abs(rank1-rank2).
 * All penalties are added up and then divided by the worst possible score, which is the one 
 * corresponding with the items being in the completely reversed order.  
 * </p><p>
 * One thing the algorithm does not do (yet), contrary to NDPM, is taking ties into account.
 * </p>
 */
public final class RankBasedRecommenderEvaluator extends AbstractRankBasedRecommenderEvaluator {
  
  RunningAverage average;
  
  @Override
  protected void reset() {
    average = new FullRunningAverage();
  }
  
  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) 
  {
    System.out.println("Unimplemented");
  }
  
  @Override
  protected void processOneUser(long userId, ArrayList<Object[]> allPrefs)
  {
    double score = calculateScore(userId, allPrefs);
    average.addDatum(score);
  }

  double calculateScore(long userId, ArrayList<Object[]> allPrefs)
  {
    if (allPrefs.size() <= 1)
      return 1;
    
    // pref = [itemId, realPref, realRank, estPref, estRank]

    // sort by real prefs and set rankings
    Collections.sort(allPrefs, new Comparator<Object[]>() {
      @Override
      public int compare(Object[] o1, Object[] o2)
      {
        return Float.compare( (float)o1[1], (float)o2[1] );
      }
    });
    for (int i=0; i<allPrefs.size(); i++)
    {
      Object[] objs = allPrefs.get(i);
      objs[2] = i;
//      System.out.printf("user %d p %d item %d real %.1f rank %d est %.1f rank %d \n", userId, i, 
//        objs[0], objs[1], objs[2], objs[3], objs[4]); 
    }
//    System.out.println();
    
    // sort by estimated prefs and set rankings
    Collections.sort(allPrefs, new Comparator<Object[]>() {
      @Override
      public int compare(Object[] o1, Object[] o2)
      {
        return Float.compare( (float)o1[3], (float)o2[3] );
      }
    });
    for (int i=0; i<allPrefs.size(); i++)
    {
      Object[] objs = allPrefs.get(i);
      objs[4] = i;
//      System.out.printf("user %d p %d item %d real %.1f rank %d est %.1f rank %d \n", userId, i, 
//        objs[0], objs[1], objs[2], objs[3], objs[4]); 
    }
//    System.out.println();
    
    // now measure and cummulate all differences
    int totalDiff=0;
    for (int i=0; i<allPrefs.size(); i++)
    {
      Object[] objs = allPrefs.get(i);
      int realRank = (int) objs[2];
      int estRank = (int) objs[4];
      int diff = Math.abs(realRank - estRank);
      totalDiff += diff;      
//      System.out.printf("user %d p %d item %d real %.1f rank %d est %.1f rank %d diff %d\n", 
//        userId, i, objs[0], objs[1], realRank, objs[3], estRank, diff); 
    }
    // what would be the cumulative difference if the list was exactly in reversed order?
    int n = allPrefs.size();
    double worstCase = (n*(n+1)) / 2 - Math.ceil(n/2.0);
    double score = (worstCase - totalDiff) / worstCase;
//    System.out.println(totalDiff);
//    System.out.println(worstCase);
//    System.out.println("score: " + score);
    return score;
  }

  @Override
  protected double computeFinalEvaluation() 
  {
//    System.out.println();
//    System.out.println("compute final");
//    System.out.println(average.getAverage());
    return average.getAverage();
  }
  
  @Override
  public String toString() {
    return "RankBasedRecommenderEvaluator";
  }
  
}
