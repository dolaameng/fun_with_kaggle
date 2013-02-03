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
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.FarthestNeighborClusterSimilarity;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TreeClusteringRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;



public class SimpleUserRecommendation {
	
	private static String dataFile = "../data/train_user_events_pref.csv";
	
	public static void run() throws Exception {
		//data model
		DataModel data = new FileDataModel(new File(dataFile));
		// evaluator
		RecommenderEvaluator evaluator = 
				new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderIRStatsEvaluator irEvaluator = 
				new GenericRecommenderIRStatsEvaluator();
		// model builder
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {

			@Override
			public Recommender buildRecommender(DataModel dataModel)
					throws TasteException {
				//UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
				//UserSimilarity similarity = new EuclideanDistanceSimilarity(dataModel);
				//UserSimilarity similarity = new TanimotoCoefficientSimilarity(dataModel);
				UserSimilarity similarity = new LogLikelihoodSimilarity(dataModel);
				UserNeighborhood neighborhood = 
						new NearestNUserNeighborhood(100, similarity, dataModel);
				//return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
				//return new SlopeOneRecommender(dataModel);
				//return new SVDRecommender(dataModel, new ALSWRFactorizer(dataModel, 15, 0.05, 10));
				return new TreeClusteringRecommender(dataModel, 
						new FarthestNeighborClusterSimilarity(similarity), 
						10);
			}
		};
		// try to make recommendations
		Recommender recommender = recommenderBuilder.buildRecommender(data);
		Iterator uids = data.getUserIDs();
		Long numWithoutRec = 0L;
		while (uids.hasNext()) {
			Long uid = (Long) uids.next();
			
			List<RecommendedItem> items = recommender.recommend(uid, 10);
			//System.out.println(uid + ":" + items.size());
			if (items.size() == 0) {
				numWithoutRec += 1;
			}
		}
		System.out.println(numWithoutRec*100./data.getNumUsers() 
				+ "% users cannot get recommendations");
		
		// profiling score
		double score = evaluator.evaluate(recommenderBuilder, null, data, 0.8, 1.0);
		System.out.println("average absolute difference: " + score);
		
		// profiling precision and recall
		/*
		IRStatistics stats = irEvaluator.evaluate(recommenderBuilder, null, 
						data, null, 1, 
						GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
		*/
		//System.out.println("Precision: " + stats.getPrecision());
		//System.out.println("Recall: " + stats.getRecall());
		//System.out.println("Reach: " + stats.getReach());
		//System.out.println("Fallout: " + stats.getFallOut());
	}
}
