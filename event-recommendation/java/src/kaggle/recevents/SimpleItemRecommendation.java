package kaggle.recevents;

import java.io.File;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.knn.KnnItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.knn.NonNegativeQuadraticOptimizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

public class SimpleItemRecommendation {
	
	private static String dataFile = "../data/train_user_events_pref.csv";
	
	public static void run() throws Exception {
		// data model
		DataModel dataModel = new FileDataModel(new File(dataFile));
		// recommender builder
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
			
			@Override
			public Recommender buildRecommender(DataModel dataModel)
					throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(dataModel);
				//return new GenericItemBasedRecommender(dataModel, similarity);
				//return new SVDRecommender(dataModel, new ALSWRFactorizer(dataModel, 5, 0.05, 10));
				return new KnnItemBasedRecommender(dataModel, similarity, 
						new NonNegativeQuadraticOptimizer(), 100);
			}
		};
		// evaluator
		RecommenderEvaluator evaluator = 
				new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderIRStatsEvaluator irEvaluator = 
				new GenericRecommenderIRStatsEvaluator();
		// try to make recommendations
		Recommender recommender = recommenderBuilder.buildRecommender(dataModel);
		Iterator uids = dataModel.getUserIDs();
		Long numWithoutRec = 0L;
		while (uids.hasNext()) {
			Long uid = (Long) uids.next();

			List<RecommendedItem> items = recommender.recommend(uid, 10);
			//System.out.println(uid + ":" + items.size());
			if (items.size() == 0) {
				numWithoutRec += 1;
			}
		}
		System.out.println(numWithoutRec*100./dataModel.getNumUsers() 
				+ "% users cannot get recommendations");

		// profiling score
		double score = evaluator.evaluate(recommenderBuilder, null, dataModel, 0.8, 1.0);
		System.out.println("average absolute difference: " + score);
		// profiling precision and recall
		/*
		IRStatistics stats = irEvaluator.evaluate(recommenderBuilder, null, 
				dataModel, null, 1, 
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
		 
		System.out.println("Precision: " + stats.getPrecision());
		System.out.println("Recall: " + stats.getRecall());
		System.out.println("Reach: " + stats.getReach());
		System.out.println("Fallout: " + stats.getFallOut());
		*/
	}
}
