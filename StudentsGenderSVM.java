package com.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Random;

public class StudentsGenderSVM {

    public static void main(String[] args) {
        try {
            long start = System.currentTimeMillis(); // ⏱ iniciar cronômetro

            System.out.println("Carregando arquivo limpo...");
            CSVLoader loader = new CSVLoader();
            loader.setFieldSeparator(",");
            loader.setSource(new File("C:\\Java\\busca\\src\\main\\java\\com\\example\\data\\StudentsPerformance_clean.csv"));

            Instances data = loader.getDataSet();
            if (data == null) {
                System.err.println("ERRO: Instances == null");
                return;
            }

            System.out.println("Instâncias: " + data.numInstances());
            System.out.println("Atributos: " + data.numAttributes());

            int targetIdx = detectGenderColumn(data);
            if (targetIdx == -1) {
                System.out.println("Não foi encontrada coluna de gênero.");
                return;
            }
            data.setClassIndex(targetIdx);

            System.out.println("\nTreinando SVM (SMO)...");
            SMO svm = new SMO();
            svm.buildClassifier(data);

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(svm, data, 10, new Random(1));

            long end = System.currentTimeMillis(); // ⏱ parar cronômetro
            double elapsedSec = (end - start) / 1000.0;

            System.out.println("\n=== Estatísticas SVM ===");
            System.out.println("Acurácia: " + (eval.pctCorrect() / 100.0));
            System.out.println("Precisão: " + eval.weightedPrecision());
            System.out.println("Recall:   " + eval.weightedRecall());
            System.out.println("F1:       " + eval.weightedFMeasure());
            System.out.println("Tempo total (s): " + elapsedSec); // ⏱ estatística de tempo

            System.out.println("\n=== Primeiras 20 previsões ===");
            for (int i = 0; i < Math.min(2000, data.numInstances()); i++) {
                double pred = svm.classifyInstance(data.instance(i));
                String p = data.classAttribute().value((int) pred);
                String a = data.instance(i).stringValue(targetIdx);
                System.out.println("Linha " + i + " | Real=" + a + " | Prev=" + p);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static int detectGenderColumn(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            String n = data.attribute(i).name().toLowerCase();
            if (n.contains("gender") || n.contains("sex")) return i;
        }
        return -1;
    }
}
