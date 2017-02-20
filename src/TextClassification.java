/**
 * Created by Hao Xiong on 2/10/2017.
 * Copyright belongs to Hao Xiong, Email: haoxiong@outlook.com
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class TextClassification {
    private static final String[] STOP_WORDS = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"};
    public static final int HAM = 0;
    public static final int SPAM = 1;
    private static final String[] TRAIN_PATH = {"dataset/train/ham", "dataset/train/spam"};
    private static final String[] TEST_PATH = {"dataset/test/ham", "dataset/test/spam"};


    private static List<String> stop_words;
    private static Map<String, double[]> dictionary;
    private static double[] W;

    private static int[] total_words = {0, 0};
    private static double[] prior = {0, 0};
    private static int[] numberOfTestFiles = {0, 0};

    public static void main(String args[]) {
        dictionary = new LinkedHashMap<>();

        //-------------------------------Naive Bayes Classifier---------------------------
        long start = System.nanoTime();
        TrainMultinomialNB(TRAIN_PATH, false);
        System.out.println("Accuracy of Naive Bayes: " + testNBAccuracy(TEST_PATH, false));
        System.out.println("Time consumption(s): " + (System.nanoTime() - start) * 1.0e-9);

        //-----------------------------Logistic Regression Classifier---------------------
        start = System.nanoTime();
        incrementalGradient(vectorNormalize(toVectors(TRAIN_PATH, false)), 0.01, 0.35, 1);
        System.out.println("Accuracy of Logistic Regression: " + testLRAccuracy(toVectors(TEST_PATH, false)));
        System.out.println("Time consumption(s): " + (System.nanoTime() - start) * 1.0e-9);

        //---------------------------------Remove the Stop Words---------------------------
        initialize();
        stop_words = Arrays.asList(STOP_WORDS);
        TrainMultinomialNB(TRAIN_PATH, true);
        System.out.println("Naive Bayes without S.W.: " + testNBAccuracy(TEST_PATH, true));

        incrementalGradient(vectorNormalize(toVectors(TRAIN_PATH, true)), 0.01, 0.35, 1);
        System.out.println("Logistic Regression without S.W.: " + testLRAccuracy(toVectors(TEST_PATH, true)));

        testMemory();
    }

    private static void initialize() {
        dictionary = new LinkedHashMap<>();
        W = null;
        total_words[HAM] = 0;
        total_words[SPAM] = 0;
        prior[HAM] = 0;
        prior[SPAM] = 0;
        numberOfTestFiles[HAM] = 0;
        numberOfTestFiles[SPAM] = 0;
    }

    /***===============================Methods that used for Naive Bayes===========================**/

    private static double testNBAccuracy(String[] folders, boolean filter) {
        int ham = testNB(folders, HAM, filter);
        int spam = testNB(folders, SPAM, filter);
        return (double) (ham + spam) / (numberOfTestFiles[HAM] + numberOfTestFiles[SPAM]);
    }

    private static int testNB(String[] folders, int type, boolean filter) {
        int[] detected = {0, 0};
        numberOfTestFiles[type] = countFileNumbers(folders[type]);
        try (Stream<Path> paths = Files.walk(Paths.get(folders[type]))) {
            paths.forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    if (ApplyMultinomialNB(filePath.toString(), filter) == type) {
                        detected[type]++;
                    }
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return detected[type];
    }

    /**
     * Implement a multinomial Naive Bayes training algorithm
     */
    private static void TrainMultinomialNB(String[] folders, boolean filter) {
        //get the number of each type of text file
        int hamFiles = countFileNumbers(folders[HAM]);
        int spamFiles = countFileNumbers(folders[SPAM]);


        //count words in ham examples
        countWords(folders, HAM, filter);
        //count words in spam examples
        countWords(folders, SPAM, filter);

        //calculate the prior probability for each text type
        double total_files = (double) (hamFiles + spamFiles);
        prior[HAM] = hamFiles / total_files;
        prior[SPAM] = spamFiles / total_files;

        //update the estimated probability for each distinct word in the dictionary
        int vocabulary = dictionary.size();
        for (Map.Entry<String, double[]> entry : dictionary.entrySet()) {
            double[] prob = entry.getValue();
            //use Laplace smoothing to estimateType probability (parameters)
            prob[HAM] = (prob[HAM] + 1) / (total_words[HAM] + vocabulary);
            prob[SPAM] = (prob[SPAM] + 1) / (total_words[SPAM] + vocabulary);
        }
    }

    /**
     * Record the frequency of each word in the given text type.
     */
    private static void countWords(String[] folders, int type, boolean filter) {
        try (Stream<Path> paths = Files.walk(Paths.get(folders[type]))) {
            paths.forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    String line;
                    try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
                        while ((line = br.readLine()) != null) {
                            String[] words = line.split(" ");
                            for (String word : words) {
                                if (filter && isStopWord(word)) continue;
                                //count the total number of words in each text type
                                total_words[type]++;
                                //put the word into dictionary and count the number of each word for each text type
                                if (dictionary.containsKey(word)) {
                                    dictionary.get(word)[type]++;
                                } else {
                                    double[] counts = new double[2];
                                    counts[type]++;
                                    dictionary.put(word, counts);
                                }
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Apply the naive bayes algorithm to test data
     *
     * @param filePath the path of the text file to classify the dictionary which contains all the parameters that will be used to estimateType the probability
     * @return 0 if it's ham and 1 if spam
     */
    private static int ApplyMultinomialNB(String filePath, boolean filter) {
        double prob_ham = Math.log(prior[HAM]);
        double prob_spam = Math.log(prior[SPAM]);
        String line;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            while ((line = br.readLine()) != null) {
                String[] words = line.split(" ");
                for (String word : words) {
                    if (filter && isStopWord(word)) continue;
                    if (dictionary.containsKey(word)) {
                        double[] prob = dictionary.get(word);
                        prob_ham += Math.log(prob[HAM]);
                        prob_spam += Math.log(prob[SPAM]);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return prob_ham > prob_spam ? HAM : SPAM;
    }

    /***
     * Return the number of files in a given folder
     */
    private static int countFileNumbers(String folder) {
        //noinspection ConstantConditions
        return new File(folder).list().length;
    }

    /**
     * To tell if a word is a stop word
     */
    private static boolean isStopWord(String word) {
        return stop_words.contains(word);
    }


    /***===============================Methods that used for Logistic Regression===========================**/

    /**
     * This method and its sub method are used to convert the raw data to vectors
     * Can only be called after dictionary has been constructed
     */
    private static ArrayList<TextVector> toVectors(String[] folders, boolean filter) {
        ArrayList<TextVector> vectors = new ArrayList<>();
        List<String> word_list = new ArrayList<>(dictionary.keySet());
        fillVector(folders, HAM, word_list, vectors, filter);
        fillVector(folders, SPAM, word_list, vectors, filter);
        return vectors;
    }

    private static void fillVector(String[] folders, int type, List<String> word_list, ArrayList<TextVector> vectors, boolean filter) {
        try (Stream<Path> paths = Files.walk(Paths.get(folders[type]))) {
            paths.forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    String line;
                    float[] features = new float[word_list.size() + 1];
                    features[0] = 1;    // set x0=1 for all vectorNormalize X
                    try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
                        while ((line = br.readLine()) != null) {
                            String[] words = line.split(" ");
                            for (String word : words) {
                                if (filter && isStopWord(word)) continue;
                                int index = word_list.indexOf(word) + 1;
                                if (index != 0) {
                                    features[index]++;
                                }
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    vectors.add(new TextVector(features, type));
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static ArrayList<TextVector> vectorNormalize(ArrayList<TextVector> textVectors) {
        int size = dictionary.size() + 1;
        for (int i = 0; i < size; i++) {
            float sum = 0;
            for (TextVector tv : textVectors) sum += tv.features[i];
            for (TextVector tv : textVectors) tv.features[i] /= sum;
        }
        return textVectors;
    }

    /**
     * MAP estimateType the parameters of Logistic Regression through batch gradient ascent
     *
     * @param learningRate the learning rate in gradient ascent
     * @param lambda       the penalty strength
     * @param tolerance    the converge threshold
     */
    private static void batchGradient(ArrayList<TextVector> vectors, double learningRate, double lambda, double tolerance) {
        //vectorNormalize size
        int size = dictionary.size() + 1;
        int cvs = 0;
        W = new double[size];//initially set all w=0
        double[] W_temp = new double[size];//used to store temp Wi value
        boolean[] converged = new boolean[size];
        int i = 0, j = 0;
        //Repeat until W in all dimension are "converged"
        while (j < 200) {
            if (!converged[i]) {
                double derivative = 0;
                for (TextVector tv : vectors) {
                    if (tv.features[i] != 0) derivative += tv.features[i] * tv.predictionError(W);
                }
                derivative -= lambda * W[i];
                W_temp[i] += learningRate * derivative;
                //when the derivative of this dimension is in this range we think it's "converged"
                if (-tolerance < derivative && derivative < tolerance) {
                    converged[i] = true;
                    if (++cvs == size) {
                        System.arraycopy(W_temp, 0, W, 0, size);
                        break;
                    }
                }
            }
            if (++i == size) {
                i = 0;
                j++;
                System.out.println("ROUND: " + j);
                System.arraycopy(W_temp, 0, W, 0, size);
            }
        }
    }

    /**
     * MAP estimateType the parameters of Logistic Regression through incremental gradient ascent
     *
     * @param learningRate the learning rate in gradient ascent
     * @param lambda       the penalty strength
     */
    private static void incrementalGradient(ArrayList<TextVector> vectors, double learningRate, double lambda, int repeat) {
        int size = dictionary.size() + 1;
        W = new double[size];//initially set all w=0
        while (repeat-- > 0) {
            for (TextVector tv : vectors) {
                for (int i = 0; i < size; i++) {
                    double derivative = 0;
                    //update the parameters according to prediction error with respect to this single training example only.
                    if (tv.features[i] != 0) derivative = tv.features[i] * tv.predictionError(W);
                    W[i] += learningRate * (derivative - lambda * W[i]);
                }
            }
            learningRate *= 0.95;
            lambda *= 1 / 0.95;
        }
    }

    /**
     * Test the accuracy of Logistic Regression method
     */
    private static double testLRAccuracy(ArrayList<TextVector> test_vectors) {
        int correct = 0;
        for (TextVector tv : test_vectors) {
            if (tv.estimateType(W) == tv.type) correct++;
        }
        return (double) correct / (numberOfTestFiles[HAM] + numberOfTestFiles[SPAM]);
    }

    private static void testMemory() {

        double mb = 1024 * 1024;

        //Getting the runtime reference from system
        Runtime runtime = Runtime.getRuntime();

        System.out.println("##### Heap utilization statistics [MB] #####");

        //Print used memory
        System.out.println("Used Memory:"
                + (runtime.totalMemory() - runtime.freeMemory()) / mb);

        //Print free memory
        System.out.println("Free Memory:"
                + runtime.freeMemory() / mb);

        //Print total available memory
        System.out.println("Total Memory:" + runtime.totalMemory() / mb);

        //Print Maximum available memory
        System.out.println("Max Memory:" + runtime.maxMemory() / mb);
    }
}

/**
 * Represent each example vector
 */
class TextVector {
    //key: the index of the word in dictionary, value the number of the word
    public float[] features;
    public int type;

    TextVector(float[] fts, int tp) {
        features = fts;
        type = tp;
    }

    double predictionError(double[] W) {
        double sum = 0;
        for (int i = 0; i < W.length; i++) {
            if (features[i] != 0) sum += W[i] * features[i];
        }
        return type - 1 / (1 + Math.exp(sum));
    }

    int estimateType(double[] W) {
        double sum = 0;
        for (int i = 0; i < W.length; i++) {
            if (features[i] != 0) sum += W[i] * features[i];
        }
        return sum < 0 ? TextClassification.HAM : TextClassification.SPAM;
    }

}