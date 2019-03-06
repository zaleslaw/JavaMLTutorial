package Apache_Math;

import org.apache.commons.math3.stat.regression.SimpleRegression;

/**
 * y = intercept + slope * x
 */
public class Ex_3_Regression {
    public static void main(String[] args) {

        double[][] data = { { 1, 3 }, {2, 5 }, {3, 7 }, {4, 14 }, {5, 11 }};
        SimpleRegression regression = new SimpleRegression(false);
        //the argument, false, tells the class not to include a constant
        regression.addData(data);

        System.out.println(regression.getIntercept());
// displays intercept of regression line, since we have constrained the constant, 0.0 is returned

        System.out.println(regression.getSlope());
// displays slope of regression line

        System.out.println(regression.getSlopeStdErr());
// displays slope standard error

        System.out.println(regression.getInterceptStdErr() );

        System.out.println(regression.predict(1.5));

    }
};
