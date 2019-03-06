package Step_1_ndArrays_tutorial;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * --- Nd4j Example 9: Functions ---
 *
 * In this example, we'll see how apply some mathematical functions to a matrix
 *
 * Created by cvn on 9/7/14.
 */

public class Nd4jEx9_Functions {

    public static void main(String[] args) {

        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
        INDArray ndv; // a placeholder variable to print out and leave the original data unchanged
        System.out.println(nd);

        //this normalizes data and helps activate artificial neurons in deep-learning nets and assigns it to var ndv
        ndv = sigmoid(nd);
        System.out.println("Sigmoid");
        System.out.println(ndv);

        //this gives you absolute value
        System.out.println("Abs");
        ndv = abs(nd);
        System.out.println(ndv);

        //a hyperbolic function to transform data much like sigmoid.
        System.out.println("Tanh");
        ndv = tanh(nd);
        System.out.println(ndv);


        //exponentiation
        System.out.println("Exp");
        ndv = exp(nd);
        System.out.println(ndv);

        //square root
        System.out.println("Sqrt");
        ndv = sqrt(nd);
        System.out.println(ndv);
    }
}
