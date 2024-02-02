package com.liorkn.embedded;


public class FastTextTrainConf {

    /**
     * The following arguments are mandatory:
     *   -input              training file path
     *   -output             output file path
     *
     * The following arguments are optional:
     *   -verbose            verbosity level [2]
     *
     * The following arguments for the dictionary are optional:
     *   -minCount           minimal number of word occurences [1]
     *   -minCountLabel      minimal number of label occurences [0]
     *   -wordNgrams         max length of word ngram [1]
     *   -bucket             number of buckets [2000000]
     *   -minn               min length of char ngram [0]
     *   -maxn               max length of char ngram [0]
     *   -t                  sampling threshold [0.0001]
     *   -label              labels prefix [__label__]
     *
     * The following arguments for training are optional:
     *   -lr                 learning rate [0.1]
     *   -lrUpdateRate       change the rate of updates for the learning rate [100]
     *   -dim                size of word vectors [100]
     *   -ws                 size of the context window [5]
     *   -epoch              number of epochs [5]
     *   -neg                number of negatives sampled [5]
     *   -loss               loss function {ns, hs, softmax} [softmax]
     *   -thread             number of threads [12]
     *   -pretrainedVectors  pretrained word vectors for supervised learning []
     *   -saveOutput         whether output params should be saved [false]
     *
     * The following arguments for quantization are optional:
     *   -cutoff             number of words and ngrams to retain [0]
     *   -retrain            whether embeddings are finetuned if a cutoff is applied [false]
     *   -qnorm              whether the norm is quantized separately [false]
     *   -qout               whether the classifier is quantized [false]
     *   -dsub               size of each sub-vector [2]
     *
     * Process finished with exit code 1
     */

    /**
     * 迭代次数
     */
    private int epoch = 5;
    private int dim = 100;
    private int neg = 5;
    /**
     * loss function {ns, hs, softmax}
     */
    private String loss = "softmax";
    private int minCount = 1;
    private int wordNgrams = 1;
    private double lr = 0.1;
    private int thread = 12;
    private int ws = 5;

}
