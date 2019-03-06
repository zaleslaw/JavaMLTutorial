package Apache_Math;

import java.util.Arrays;
import java.util.List;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Ex_1_Stat {
    public static void main(String[] args) {
        DescriptiveStatistics stats1 = new DescriptiveStatistics();
        DescriptiveStatistics stats2 = new DescriptiveStatistics();
        DescriptiveStatistics stats3 = new DescriptiveStatistics();
        DescriptiveStatistics stats4 = new DescriptiveStatistics();

        // Add the data from the array
        for (double[] aData : data) {
            stats1.addValue(aData[1]);
            stats2.addValue(aData[2]);
            stats3.addValue(aData[3]);
            stats4.addValue(aData[4]);
        }

        System.out.println("Feature #1");
        System.out.println("Mean " +  stats1.getMean());
        System.out.println("Std " +  stats1.getStandardDeviation());
        System.out.println("Median " +  stats1.getPercentile(50));

        System.out.println("Feature #2");
        System.out.println("Mean " +  stats2.getMean());
        System.out.println("Std " +  stats2.getStandardDeviation());
        System.out.println("Median " +  stats2.getPercentile(50));


        System.out.println("Feature #3");
        System.out.println("Mean " +  stats3.getMean());
        System.out.println("Std " +  stats3.getStandardDeviation());
        System.out.println("Median " +  stats3.getPercentile(50));


        System.out.println("Feature #4");
        System.out.println("Mean " +  stats4.getMean());
        System.out.println("Std " +  stats4.getStandardDeviation());
        System.out.println("Median " +  stats4.getPercentile(50));

        System.out.println("Cov b/w f1 and f2 " +  new PearsonsCorrelation().correlation(stats1.getValues(), stats2.getValues()));
        System.out.println("Cov b/w f1 and f3 " +  new PearsonsCorrelation().correlation(stats1.getValues(), stats3.getValues()));
        System.out.println("Cov b/w f1 and f4 " +  new PearsonsCorrelation().correlation(stats1.getValues(), stats4.getValues()));


    }


    /** The Iris dataset. */
    private static final double[][] data = {
        {1, 5.1, 3.5, 1.4, 0.2},
        {1, 4.9, 3, 1.4, 0.2},
        {1, 4.7, 3.2, 1.3, 0.2},
        {1, 4.6, 3.1, 1.5, 0.2},
        {1, 5, 3.6, 1.4, 0.2},
        {1, 5.4, 3.9, 1.7, 0.4},
        {1, 4.6, 3.4, 1.4, 0.3},
        {1, 5, 3.4, 1.5, 0.2},
        {1, 4.4, 2.9, 1.4, 0.2},
        {1, 4.9, 3.1, 1.5, 0.1},
        {1, 5.4, 3.7, 1.5, 0.2},
        {1, 4.8, 3.4, 1.6, 0.2},
        {1, 4.8, 3, 1.4, 0.1},
        {1, 4.3, 3, 1.1, 0.1},
        {1, 5.8, 4, 1.2, 0.2},
        {1, 5.7, 4.4, 1.5, 0.4},
        {1, 5.4, 3.9, 1.3, 0.4},
        {1, 5.1, 3.5, 1.4, 0.3},
        {1, 5.7, 3.8, 1.7, 0.3},
        {1, 5.1, 3.8, 1.5, 0.3},
        {1, 5.4, 3.4, 1.7, 0.2},
        {1, 5.1, 3.7, 1.5, 0.4},
        {1, 4.6, 3.6, 1, 0.2},
        {1, 5.1, 3.3, 1.7, 0.5},
        {1, 4.8, 3.4, 1.9, 0.2},
        {1, 5, 3, 1.6, 0.2},
        {1, 5, 3.4, 1.6, 0.4},
        {1, 5.2, 3.5, 1.5, 0.2},
        {1, 5.2, 3.4, 1.4, 0.2},
        {1, 4.7, 3.2, 1.6, 0.2},
        {1, 4.8, 3.1, 1.6, 0.2},
        {1, 5.4, 3.4, 1.5, 0.4},
        {1, 5.2, 4.1, 1.5, 0.1},
        {1, 5.5, 4.2, 1.4, 0.2},
        {1, 4.9, 3.1, 1.5, 0.1},
        {1, 5, 3.2, 1.2, 0.2},
        {1, 5.5, 3.5, 1.3, 0.2},
        {1, 4.9, 3.1, 1.5, 0.1},
        {1, 4.4, 3, 1.3, 0.2},
        {1, 5.1, 3.4, 1.5, 0.2},
        {1, 5, 3.5, 1.3, 0.3},
        {1, 4.5, 2.3, 1.3, 0.3},
        {1, 4.4, 3.2, 1.3, 0.2},
        {1, 5, 3.5, 1.6, 0.6},
        {1, 5.1, 3.8, 1.9, 0.4},
        {1, 4.8, 3, 1.4, 0.3},
        {1, 5.1, 3.8, 1.6, 0.2},
        {1, 4.6, 3.2, 1.4, 0.2},
        {1, 5.3, 3.7, 1.5, 0.2},
        {1, 5, 3.3, 1.4, 0.2},
        {2, 7, 3.2, 4.7, 1.4},
        {2, 6.4, 3.2, 4.5, 1.5},
        {2, 6.9, 3.1, 4.9, 1.5},
        {2, 5.5, 2.3, 4, 1.3},
        {2, 6.5, 2.8, 4.6, 1.5},
        {2, 5.7, 2.8, 4.5, 1.3},
        {2, 6.3, 3.3, 4.7, 1.6},
        {2, 4.9, 2.4, 3.3, 1},
        {2, 6.6, 2.9, 4.6, 1.3},
        {2, 5.2, 2.7, 3.9, 1.4},
        {2, 5, 2, 3.5, 1},
        {2, 5.9, 3, 4.2, 1.5},
        {2, 6, 2.2, 4, 1},
        {2, 6.1, 2.9, 4.7, 1.4},
        {2, 5.6, 2.9, 3.6, 1.3},
        {2, 6.7, 3.1, 4.4, 1.4},
        {2, 5.6, 3, 4.5, 1.5},
        {2, 5.8, 2.7, 4.1, 1},
        {2, 6.2, 2.2, 4.5, 1.5},
        {2, 5.6, 2.5, 3.9, 1.1},
        {2, 5.9, 3.2, 4.8, 1.8},
        {2, 6.1, 2.8, 4, 1.3},
        {2, 6.3, 2.5, 4.9, 1.5},
        {2, 6.1, 2.8, 4.7, 1.2},
        {2, 6.4, 2.9, 4.3, 1.3},
        {2, 6.6, 3, 4.4, 1.4},
        {2, 6.8, 2.8, 4.8, 1.4},
        {2, 6.7, 3, 5, 1.7},
        {2, 6, 2.9, 4.5, 1.5},
        {2, 5.7, 2.6, 3.5, 1},
        {2, 5.5, 2.4, 3.8, 1.1},
        {2, 5.5, 2.4, 3.7, 1},
        {2, 5.8, 2.7, 3.9, 1.2},
        {2, 6, 2.7, 5.1, 1.6},
        {2, 5.4, 3, 4.5, 1.5},
        {2, 6, 3.4, 4.5, 1.6},
        {2, 6.7, 3.1, 4.7, 1.5},
        {2, 6.3, 2.3, 4.4, 1.3},
        {2, 5.6, 3, 4.1, 1.3},
        {2, 5.5, 2.5, 4, 1.3},
        {2, 5.5, 2.6, 4.4, 1.2},
        {2, 6.1, 3, 4.6, 1.4},
        {2, 5.8, 2.6, 4, 1.2},
        {2, 5, 2.3, 3.3, 1},
        {2, 5.6, 2.7, 4.2, 1.3},
        {2, 5.7, 3, 4.2, 1.2},
        {2, 5.7, 2.9, 4.2, 1.3},
        {2, 6.2, 2.9, 4.3, 1.3},
        {2, 5.1, 2.5, 3, 1.1},
        {2, 5.7, 2.8, 4.1, 1.3},
        {3, 6.3, 3.3, 6, 2.5},
        {3, 5.8, 2.7, 5.1, 1.9},
        {3, 7.1, 3, 5.9, 2.1},
        {3, 6.3, 2.9, 5.6, 1.8},
        {3, 6.5, 3, 5.8, 2.2},
        {3, 7.6, 3, 6.6, 2.1},
        {3, 4.9, 2.5, 4.5, 1.7},
        {3, 7.3, 2.9, 6.3, 1.8},
        {3, 6.7, 2.5, 5.8, 1.8},
        {3, 7.2, 3.6, 6.1, 2.5},
        {3, 6.5, 3.2, 5.1, 2},
        {3, 6.4, 2.7, 5.3, 1.9},
        {3, 6.8, 3, 5.5, 2.1},
        {3, 5.7, 2.5, 5, 2},
        {3, 5.8, 2.8, 5.1, 2.4},
        {3, 6.4, 3.2, 5.3, 2.3},
        {3, 6.5, 3, 5.5, 1.8},
        {3, 7.7, 3.8, 6.7, 2.2},
        {3, 7.7, 2.6, 6.9, 2.3},
        {3, 6, 2.2, 5, 1.5},
        {3, 6.9, 3.2, 5.7, 2.3},
        {3, 5.6, 2.8, 4.9, 2},
        {3, 7.7, 2.8, 6.7, 2},
        {3, 6.3, 2.7, 4.9, 1.8},
        {3, 6.7, 3.3, 5.7, 2.1},
        {3, 7.2, 3.2, 6, 1.8},
        {3, 6.2, 2.8, 4.8, 1.8},
        {3, 6.1, 3, 4.9, 1.8},
        {3, 6.4, 2.8, 5.6, 2.1},
        {3, 7.2, 3, 5.8, 1.6},
        {3, 7.4, 2.8, 6.1, 1.9},
        {3, 7.9, 3.8, 6.4, 2},
        {3, 6.4, 2.8, 5.6, 2.2},
        {3, 6.3, 2.8, 5.1, 1.5},
        {3, 6.1, 2.6, 5.6, 1.4},
        {3, 7.7, 3, 6.1, 2.3},
        {3, 6.3, 3.4, 5.6, 2.4},
        {3, 6.4, 3.1, 5.5, 1.8},
        {3, 6, 3, 4.8, 1.8},
        {3, 6.9, 3.1, 5.4, 2.1},
        {3, 6.7, 3.1, 5.6, 2.4},
        {3, 6.9, 3.1, 5.1, 2.3},
        {3, 5.8, 2.7, 5.1, 1.9},
        {3, 6.8, 3.2, 5.9, 2.3},
        {3, 6.7, 3.3, 5.7, 2.5},
        {3, 6.7, 3, 5.2, 2.3},
        {3, 6.3, 2.5, 5, 1.9},
        {3, 6.5, 3, 5.2, 2},
        {3, 6.2, 3.4, 5.4, 2.3},
        {3, 5.9, 3, 5.1, 1.8}
    };
};
