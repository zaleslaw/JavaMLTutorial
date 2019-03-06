package Step_5_onSpark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTOnSparkSpeedUp {

    private static int batchSizePerWorker = 16;
    private static int numEpochs = 3;

    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, false, 12345);
        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        trainData = trainData.sample(true, 0.1);
        testData = testData.sample(true, 0.1);


        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.02))// To configure: .updater(Nesterovs.builder().momentum(0.9).build())
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(100).build())
                .layer(1, new DenseLayer.Builder().nIn(100).nOut(50).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(50).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        //Configuration for Spark training: see https://deeplearning4j.org/distributed for explanation of these configuration options
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()

                /**
                 * This can be any port, but it should be open for IN/OUT comms on all Spark nodes
                 */
                .unicastPort(40123)

                /**
                 * if you're running this example on Hadoop/YARN, please provide proper netmask for out-of-spark comms
                 */
                //.networkMask("10.0.0.0/16")

                /**
                 * However, if you're running this example on Spark standalone cluster, you can rely on Spark internal addressing via $SPARK_PUBLIC_DNS env variables announced on each node
                 */
                .controllerAddress("127.0.0.1")
                .build();

        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSizePerWorker)
                // encoding threshold. Please check https://deeplearning4j.org/distributed for details
                .updatesThreshold(1e-3)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .batchSizePerWorker(batchSizePerWorker)
                //.storageLevel()

                // this option will enforce exactly 4 workers for each Spark node
                .workersPerNode(4)
                .build();

        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        //Execute training:
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            System.out.println("Completed Epoch " + i);
        }

        //Perform evaluation (distributed)
//        Evaluation evaluation = sparkNet.evaluate(testData);
        Evaluation evaluation = sparkNet.doEvaluation(testData, 64, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
        System.out.println(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);



    }
}
