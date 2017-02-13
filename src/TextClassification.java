/**
 * Created by Hao Xiong on 2/10/2017.
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
    private static final int HAM = 0;
    private static final int SPAM = 1;
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

        //-----------------Naive Bayes Classifier--------------
        long start = System.nanoTime();
        TrainMultinomialNB(TRAIN_PATH, false);
        System.out.println("Accuracy of Naive Bayes: " + testNBAccuracy(false));
        System.out.println("Time consumption: " + (System.nanoTime() - start) * 1.0e-9);
        //printMap(dictionary);

        //--------------Logistic Regression Classifier--------------
        start = System.nanoTime();
        gradientAscent(initVectors(TRAIN_PATH, false), 0.05, 40.009, 0.01);//gradientAscent(0.0001, 40.01, 0.0001);
        System.out.println("Accuracy of Logistic Regression: " + testLRAccuracy(initVectors(TEST_PATH, false)));
        System.out.println("Time consumption: " + (System.nanoTime() - start) * 1.0e-9);

        //----------------Remove the Stop Words------------------
        initialize();
        stop_words = Arrays.asList(STOP_WORDS);
        TrainMultinomialNB(TRAIN_PATH, true);
        System.out.println("Naive Bayes without S.W.: " + testNBAccuracy(true));

        gradientAscent(initVectors(TRAIN_PATH, true), 0.05, 40.009, 0.01);
        System.out.println("Logistic Regression without S.W.: " + testLRAccuracy(initVectors(TEST_PATH, true)));

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

    private static double testNBAccuracy(boolean filter) {
        int ham = testNB(TEST_PATH, HAM, filter);
        int spam = testNB(TEST_PATH, SPAM, filter);
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
     * Print out the entire map, a debug function
     */
    private static void printMap(Map mp) {
        Iterator it = mp.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry) it.next();
            String word = (String) pair.getKey();
            double[] values = (double[]) pair.getValue();
            System.out.println(word + " = " + values[HAM] + "," + values[SPAM]);
            //it.remove(); // avoids a ConcurrentModificationException
        }
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
            //use Laplace smoothing to estimate probability (parameters)
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
     * @param filePath the path of the text file to classify the dictionary which contains all the parameters that will be used to estimate the probability
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
    private static ArrayList<MailVector> initVectors(String[] folders, boolean filter) {
        ArrayList<MailVector> vectors = new ArrayList<>();
        List<String> word_list = new ArrayList<>(dictionary.keySet());
        fillVector(folders, HAM, word_list, vectors, filter);
        fillVector(folders, SPAM, word_list, vectors, filter);
        return vectors;
    }

    private static void fillVector(String[] folders, int type, List<String> word_list, ArrayList<MailVector> vectors, boolean filter) {
        try (Stream<Path> paths = Files.walk(Paths.get(folders[type]))) {
            paths.forEach(filePath -> {
                if (Files.isRegularFile(filePath)) {
                    String line;
                    Map<Integer, Integer> features = new HashMap<>();
                    features.put(0, 1);// set x0=1 for all vector X
                    try (BufferedReader br = new BufferedReader(new FileReader(filePath.toFile()))) {
                        while ((line = br.readLine()) != null) {
                            String[] words = line.split(" ");
                            for (String word : words) {
                                if (filter && isStopWord(word)) continue;
                                int index = word_list.indexOf(word) + 1;
                                if (index == 0) continue;
                                if (features.containsKey(index)) {
                                    features.put(index, features.get(index) + 1);
                                } else {
                                    features.put(index, 1);
                                }
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    vectors.add(new MailVector(features, type));
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * MAP estimate the parameters of Logistic Regression through gradient ascent
     *
     * @param learningRate the learning rate in gradient ascent
     * @param lambda       the penalty strength
     * @param tolerance    the converge threshold
     */
    private static void gradientAscent(ArrayList<MailVector> vectors, double learningRate, double lambda, double tolerance) {
        //vector size
        int vsize = dictionary.size()+1;
        W = new double[vsize];//initially set all w=0
        double[] W_temp = new double[vsize];//used to store temp Wi value
        boolean[] converge = new boolean[vsize];
        //loop until W in all dimension are "converged"
        for (int i = 0; !converge[i]; ) {
            double derivative = 0;
            for (MailVector mv : vectors) {
                if (mv.features.containsKey(i)) {
                    derivative += mv.features.get(i) * mv.predictError(W_temp);
                }
            }
            derivative -= lambda * W_temp[i];
            //update W[i]
            W_temp[i] += learningRate * derivative;
            //when the derivative of this dimension is in this range we think it's "converged"
            if (derivative < tolerance && derivative > -1 * tolerance) {
                converge[i] = true;
            }
            if (++i == vsize) {
                i = 0;
                //update W
            }
        }
    }

    /**
     * If the prediction error of a vector is less than .5 then it's correctly classified
     * */
    private static double testLRAccuracy(ArrayList<MailVector> test_vectors) {
        int detected = 0;
        for (MailVector mv : test_vectors) {
            if (mv.predictError(W) < 0.5) detected++;
        }
        return (double) detected / (numberOfTestFiles[HAM] + numberOfTestFiles[SPAM]);
    }

    public static void testMemory() {

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
class MailVector {
    //key: the index of the word in dictionary, value the number of the word
    public Map<Integer, Integer> features;
    public int type;

    MailVector(Map<Integer, Integer> fts, int tp) {
        features = fts;
        type = tp;
    }

    public double predictError(double[] W) {
        double linearSum = 0;
        for (int i : features.keySet()) {
            linearSum += W[i] * features.get(i);
        }
        return type - 1 / (1 + Math.exp(linearSum));
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Integer, Integer> entry : features.entrySet()) {
            sb.append("[").append(entry.getKey()).append(":").append(entry.getValue()).append("], ");
        }
        return sb.append(type).toString();
    }
}