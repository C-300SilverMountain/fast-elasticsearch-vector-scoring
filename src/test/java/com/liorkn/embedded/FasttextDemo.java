package com.liorkn.embedded;

import com.github.jfasttext.JFastText;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * 参考：
 * https://mengbaoliang.cn/archives/27523/
 * <p>
 * https://developer.aliyun.com/article/750481
 * https://elasticsearch.cn/article/13689
 * https://blog.csdn.net/MpkeShell/article/details/133322347
 * https://blog.51cto.com/u_8238263/6022414
 * https://www.jianshu.com/p/b0a5d622a01d
 */
public class FasttextDemo {

    private FastTextTrainConf trainConf;

    private static String[] structTrainArgs() {

        String inFilePath = "G:\\qzd\\JavaProject\\QZD_GROUP\\openSource\\github\\fast-elasticsearch-vector-scoring\\doc\\embedded\\fasttext\\corpus\\fasttext.data";
        String outFilePath = "G:\\qzd\\JavaProject\\QZD_GROUP\\openSource\\github\\fast-elasticsearch-vector-scoring\\doc\\embedded\\fasttext\\model\\model";
        List<String> argList = new ArrayList<>();
        argList.add("supervised");
        argList.add("-input");
        argList.add(inFilePath);
        argList.add("-output");
        argList.add(outFilePath);
//        if (null != trainConf) {
//            argList.add("-epoch");
//            argList.add(String.valueOf(trainConf.getEpoch()));
//            argList.add("-dim");
//            argList.add(String.valueOf(trainConf.getDim()));
//            argList.add("-neg");
//            argList.add(String.valueOf(trainConf.getNeg()));
//            argList.add("-loss");
//            argList.add(String.valueOf(trainConf.getLoss()));
//            argList.add("-minCount");
//            argList.add(String.valueOf(trainConf.getMinCount()));
//            argList.add("-wordNgrams");
//            argList.add(String.valueOf(trainConf.getWordNgrams()));
//            argList.add("-lr");
//            argList.add(String.valueOf(trainConf.getLr()));
//            argList.add("-thread");
//            argList.add(String.valueOf(trainConf.getThread()));
//            argList.add("-ws");
//            argList.add(String.valueOf(trainConf.getWs()));
//        }
        return argList.toArray(new String[0]);
    }

    private static void train() {
        JFastText fastText = new JFastText();
        fastText.runCmd(structTrainArgs());
    }

    public static void main(String[] args) {

//        train();
//
//        JFastText fastText = new JFastText();
//        fastText.loadModel("G:\\qzd\\JavaProject\\QZD_GROUP\\openSource\\github\\fast-elasticsearch-vector-scoring\\doc\\embedded\\fasttext\\model\\model.bin");
//
//        String query = "文化 传媒";
//
//        String[] termList = query.split(" ");
//
//        List<Float> vector = fastText.getVector("文化");
//        System.out.println(fastText.getWords());
//
//        System.out.println(vector.size());
//        vector.forEach(e -> System.out.println(e));
//        Float[] arr = new Float[vector.size()];
//        vector.toArray(arr);
//
//        System.out.println(convertArrayToBase64(arr));


        List<Float> vector = new ArrayList<>();
        vector.add(1F);
        vector.add(2F);
        vector.add(3F);

        List<Float> floats1 = mapDivide(vector, 3F);
        floats1.stream().forEach(e -> {
            System.out.print(e + ",");
        });


        Float sum = sum(vector);
        System.out.println(sum);
        List<Float> floats = selfMultip(vector, 2);

        floats.stream().forEach(e -> {
            System.out.print(e + ",");
        });

        double sqrt = Math.sqrt(sum);
        System.out.println(sqrt);

        String sentence = "中置 三通 集成 单向 阀";
        List<Float> sentenceVector = getSentenceVector(sentence);
        sentenceVector.stream().forEach(e -> {
            System.out.print(e + ",");
        });
        System.out.println();

        Float[] arr = new Float[sentenceVector.size()];
        sentenceVector.toArray(arr);
        String convertArrayToBase64 = convertArrayToBase64(arr);
        System.out.println(convertArrayToBase64);

    }

    /**
     * 获取句子的向量
     *
     * @param sentence 要求分词，term之间并以空格隔开
     * @return
     */
    public static List<Float> getSentenceVector(String sentence) {
        JFastText fastText = new JFastText();
        fastText.loadModel("G:\\qzd\\JavaProject\\QZD_GROUP\\openSource\\github\\fast-elasticsearch-vector-scoring\\doc\\embedded\\fasttext\\model\\model.bin");
        List<Float>sentenceVector = null;
        // 如果句子以空格切分好，则句子向量为每个 (子词/norm) 相加取均值
        String[] termList = sentence.split(" ");
        for(String term: termList){
            List<Float> vector = fastText.getVector(term);
            Double sqrt = Math.sqrt(sum(selfMultip(vector, 2)));
            List<Float>avgVector = mapDivide(vector,sqrt.floatValue());
            if (sentenceVector == null) {
                sentenceVector = avgVector;
                continue;
            }
            sentenceVector = add(sentenceVector,avgVector);
        }
        return sentenceVector;
    }

    /**
     * 向量除以数值
     *
     * @param vector
     * @param n
     * @return
     */
    public static List<Float> mapDivide(List<Float> vector, Float n) {
        List<Float> newVector = new ArrayList<>();
        for (int i = 0; i < vector.size(); i++) {
            Float newVal = vector.get(i) / n;
            newVector.add(newVal);
        }
        return newVector;

    }

    /**
     * 两个向量相加
     *
     * @param vector1
     * @param vector2
     * @return
     */
    public static List<Float> add(List<Float> vector1, List<Float> vector2) {
        if (vector1.size() != vector2.size() || vector1.size() == 0) {
            return new ArrayList<>();
        }
        List<Float> newVector = new ArrayList<>();
        for (int i = 0; i < vector1.size(); i++) {
            Float newVal = vector1.get(i) + vector2.get(i);
            newVector.add(newVal);
        }
        return newVector;
    }

    public static Float sum(List<Float> vector) {
        Float sum = 0F;
        for (Float val : vector) {
            sum += val;
        }
        return sum;
    }

    /**
     * 向量幂运算
     *
     * @return
     */
    public static List<Float> selfMultip(List<Float> vector, int n) {
        List<Float> newVector = new ArrayList<>();
        for (Float val : vector) {
            Float newVal = 1F;
            for (int i = 0; i < n; i++) {
                newVal *= val;
            }
            newVector.add(newVal);
        }
        return newVector;
    }

    public static String convertArrayToBase64(Float[] array) {
        final int capacity = Float.BYTES * array.length;
        final ByteBuffer bb = ByteBuffer.allocate(capacity);
        for (float v : array) {
            bb.putFloat(v);
        }
        bb.rewind();
        final ByteBuffer encodedBB = Base64.getEncoder().encode(bb);

        return new String(encodedBB.array());
    }
}
