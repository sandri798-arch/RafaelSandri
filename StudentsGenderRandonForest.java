package com.example;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import java.io.*;
import java.util.*;

public class StudentsGenderRandonForest {

    public static void main(String[] args) {

        long startAll = System.nanoTime();  // tempo total do programa

        String original = "C:\\Java\\busca\\src\\main\\java\\com\\example\\data\\StudentsPerformance.csv";
        String cleaned  = "C:\\Java\\busca\\src\\main\\java\\com\\example\\data\\StudentsPerformance_clean.csv";

        try {
            System.out.println("Limpando arquivo...");
            cleanCSV(original, cleaned);

            System.out.println("Arquivo limpo salvo em:");
            System.out.println(cleaned);

            System.out.println("\nCarregando CSV limpo no WEKA...");
            CSVLoader loader = new CSVLoader();
            loader.setFieldSeparator(",");  // CSV final com vírgula
            loader.setSource(new File(cleaned));

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

            System.out.println("Treinando RandomForest...");

            long startTrain = System.nanoTime(); // tempo só do RF

            RandomForest rf = new RandomForest();
            rf.buildClassifier(data);

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(rf, data, 10, new Random(1));

            long endTrain = System.nanoTime();

            System.out.println("\n=== Estatísticas ===");
            System.out.println("Acurácia: " + (eval.pctCorrect()/100.0));
            System.out.println("Precisão: " + eval.weightedPrecision());
            System.out.println("Recall:   " + eval.weightedRecall());
            System.out.println("F1:       " + eval.weightedFMeasure());

            System.out.println("\n=== Tempo de Execução ===");
            System.out.printf("Tempo treino+validação: %.3f segundos%n",
                    (endTrain - startTrain) / 1_000_000_000.0);

            long endAll = System.nanoTime();
            System.out.printf("Tempo total do programa: %.3f segundos%n",
                    (endAll - startAll) / 1_000_000_000.0);

            System.out.println("\n=== Primeiras 20 previsões ===");
            for (int i = 0; i < Math.min(2000, data.numInstances()); i++) {
                double pred = rf.classifyInstance(data.instance(i));
                String p = data.classAttribute().value((int) pred);
                String a = data.instance(i).stringValue(targetIdx);
                System.out.println("Linha " + i + " | Real=" + a + " | Prev=" + p);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // -------------------------------------------
    // LIMPEZA DO ARQUIVO
    // -------------------------------------------
    private static void cleanCSV(String in, String out) throws Exception {

        List<String[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(in))) {
            String line;
            while ((line = br.readLine()) != null) {

                // remove aspas simples
                line = line.replace("'", "");

                // convert ||| → ,
                line = line.replace("|||", ",");

                // remove ?
                line = line.replace("?", "");

                // converte ; para ,
                line = line.replace(";", ",");

                // remove múltiplas vírgulas seguidas
                line = line.replaceAll(",{2,}", ",");

                // quebra em colunas
                String[] parts = line.split(",", -1);

                rows.add(parts);
            }
        }

        // detectar maior número de colunas
        int maxCols = 0;
        for (String[] r : rows) {
            maxCols = Math.max(maxCols, r.length);
        }

        // padronizar todas as linhas
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(out))) {
            for (String[] r : rows) {
                String[] fixed = Arrays.copyOf(r, maxCols);
                bw.write(String.join(",", fixed));
                bw.newLine();
            }
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
