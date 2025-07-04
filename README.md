# Reconhecimento de Movimentos Isolados da Língua Brasileira de Sinais (LIBRAS)

## Descrição do Projeto

Este projeto visa desenvolver e avaliar modelos de Machine Learning para o reconhecimento de movimentos isolados da Língua Brasileira de Sinais (LIBRAS). Utilizando o dataset "LIBRAS Movement" da UCI, que consiste em séries temporais de coordenadas de pontos-chave de movimentos de mão, exploraremos diversas técnicas de aprendizado supervisionado. Isso inclui desde modelos lineares e não-lineares, até as poderosas Redes Neurais Profundas, tudo conforme o conteúdo programático da disciplina de Métodos de Machine Learning.

O objetivo principal desta fase do projeto é classificar corretamente 15 diferentes movimentos de LIBRAS. Este é um componente essencial e pré-requisito para futuras expansões em direção a um sistema de tradução de LIBRAS em contexto, agindo como um "vocabulário" de movimentos para a construção de frases e significados mais complexos.

## Dataset Utilizado

**Nome:** LIBRAS Movement Database
**Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/181/libras+movement)

**Informações de Origem (conforme `movement_libras.names`):**
*   **Criadores:** Daniel Baptista Dias, Sarajane Marques Peres, Helton Hideraldo Bíscaro (Universidade de São Paulo)
*   **Doador:** Universidade de São Paulo, Escola de Artes, Ciências e Humanidades
*   **Data:** Novembro, 2008

**Uso Anterior (conforme `movement_libras.names`):**
*   DIAS, D. B.; MADEO, R. C. B.; ROCHA, T.; BÍSCARO, H. H.; PERES, S. M.. Hand Movement Recognition for Brazilian Sign Language: A Study Using Distance-Based Neural Networks. In: 2009 International Joint Conference on Neural Networks, 2009, Atlanta, GA. Proceedings of 2009 International Joint Conference on Neural Networks. Eau Claire, WI, USA : Documation LLC, 2009. p. 697-704. Digital Object Identifier 10.1109/IJCNN.2009.5178917

**Descrição Detalhada do Dataset (conforme `movement_libras.names`):**
LIBRAS, acrônimo de "LÍngua BRAsileira de Sinais", é a língua de sinais oficial brasileira. O dataset `movement_libras` contém 15 classes de 24 instâncias cada, onde cada classe se refere a um tipo de movimento de mão em LIBRAS. O movimento da mão é representado como uma curva bidimensional realizada pela mão em um período de tempo.

As curvas foram obtidas a partir de vídeos de movimentos de mão, com a performance de LIBRAS de 4 pessoas diferentes, durante 2 sessões. Cada vídeo corresponde a apenas um movimento de mão e tem cerca de 7 segundos de duração.

No pré-processamento do vídeo, uma normalização temporal é realizada, selecionando 45 frames de cada vídeo, de acordo com uma distribuição uniforme. Em cada frame, os pixels centróides dos objetos segmentados (a mão) são encontrados, compondo a versão discreta da curva com 45 pontos. Todas as curvas são normalizadas no espaço unitário.

Para preparar esses movimentos para serem analisados por algoritmos, foi realizada uma operação de mapeamento, onde cada curva é mapeada em uma representação com 90 features, representando as coordenadas do movimento. Cada instância representa 45 pontos em um espaço bidimensional, que podem ser plotados de forma ordenada (de 1 a 45 como a coordenada X) para desenhar o caminho do movimento.

**Arquivos de Dados Disponíveis:**
O repositório da UCI fornece o dataset principal e vários sub-datasets, que podem ser utilizados para diferentes experimentos:
*   `movement_libras.data`: O dataset completo, contendo 360 instâncias (24 por classe). Este é o dataset principal para a tarefa de classificação geral.
*   `movement_libras_1.data`
*   `movement_libras_5.data`
*   `movement_libras_8.data`
*   `movement_libras_9.data`
*   `movement_libras_10.data`
    
Estes são sub-datasets que foram utilizados para experimentos específicos no artigo de pesquisa original. Eles contêm um subconjunto das instâncias do dataset principal e podem ser úteis para replicar resultados ou analisar o desempenho do modelo com menos dados.

**Número de Instâncias:** 360 (24 em cada uma das quinze classes) para `movement_libras.data`. Os sub-datasets terão um número menor de instâncias.
**Número de Atributos:** 90 numéricos (double) e 1 para a classe (inteiro)
**Valores Ausentes:** Nenhum
**Distribuição de Classes:** 6.66% para cada uma das 15 classes (dataset balanceado) no `movement_libras.data`.

**Informações Detalhadas das Classes (conforme `movement_libras.names`):**
As 15 classes de movimentos de LIBRAS são:
1.  Curved swing
2.  Horizontal swing
3.  Vertical swing
4.  Anti-clockwise arc
5.  Clockwise arc
6.  Circle
7.  Horizontal straight-line
8.  Vertical straight-line
9.  Horizontal zigzag
10. Vertical zigzag
11. Horizontal wavy
12. Vertical wavy
13. Face-up curve
14. Face-down curve
15. Tremble

**Resultados de Baseline (k-Nearest Neighbors - conforme `movement_libras.names`):**
Aplicando um algoritmo simples de k-Nearest Neighbors (com distância Euclidiana) usando o dataset completo (24 instâncias por classe, 15 classes), os seguintes valores foram obtidos:

| Algoritmo                  | Média de Agrupamento Correto (%) | Desvio Padrão | Máx. de Agrupamento Correto / Classe (%) | Mín. de Agrupamento Correto / Classe (%) |
| :------------------------- | :------------------------------- | :------------ | :--------------------------------------- | :--------------------------------------- |
| 24-nearest neighbors       | 0.3918                           | 0.1267        | 0.5556 / 5                               | 0.2587 / 10                              |
| neighbors in a ratio = 1.0 | 0.2245                           | 0.0979        | 0.3957 / 9                               | 0.1181 / 1                               |
| neighbors in a ratio = 2.0 | 0.3514                           | 0.1210        | 0.4514 / 9                               | 0.2500 / 10                              |
| neighbors in a ratio = 3.0 | 0.3848                           | 0.1266        | 0.5347 / 5                               | 0.2587 / 10                              |

Estes resultados fornecem um ponto de referência inicial para o desempenho dos modelos que serão desenvolvidos neste projeto.

## Problema Definido

**Classificação de Séries Temporais:** Dada uma série temporal de 45 frames, onde cada frame é caracterizado por 90 atributos de coordenadas (totalizando 4050 atributos se a série for achatada), o modelo deve prever a qual das 15 classes de movimentos de LIBRAS essa série temporal pertence.

## Metodologia

A metodologia seguirá um pipeline padrão e robusto de Machine Learning, adaptado para lidar com dados sequenciais e de alta dimensionalidade:

1.  **Coleta e Carregamento de Dados:**
    *   Carregar o dataset "LIBRAS Movement" (ou um de seus sub-datasets) do arquivo `.data` especificado.
    *   Separar os atributos de entrada (features) dos rótulos das classes (targets).
    *   Converter os rótulos de 1-indexado para 0-indexado para compatibilidade com bibliotecas de ML.

2.  **Pré-processamento de Dados:**
    *   **Reestruturação:** Os dados serão reestruturados para o formato de série temporal bidimensional `(número_de_frames, número_de_features_por_frame)`, ou seja, `(45, 90)` para cada instância, para uso com RNNs. Para modelos não-sequenciais, manteremos a versão achatada de 4050 atributos.
    *   **Normalização/Escalonamento:** As características numéricas serão escalonadas (padronizadas) usando `StandardScaler`.
    *   **Divisão:** O dataset será dividido em conjuntos de treinamento, validação e teste de forma estratificada para manter a proporção de classes.

3.  **Seleção e Treinamento de Modelos:**
    *   Explorar e treinar diferentes arquiteturas de modelos, selecionadas a partir do seu conteúdo programático.
    *   Cada modelo será treinado no conjunto de treinamento e sua performance será monitorada no conjunto de validação.

4.  **Avaliação de Modelos:**
    *   Medir o desempenho de cada modelo treinado utilizando métricas apropriadas para classificação multiclasse (Acurácia, Precision, Recall, F1-Score) no conjunto de teste.
    *   Gerar relatórios de classificação detalhados e matrizes de confusão, utilizando os nomes reais das classes de LIBRAS para maior clareza.

5.  **Análise de Resultados:**
    *   Comparar o desempenho de todos os modelos com os resultados de baseline fornecidos pelo dataset.
    *   Identificar os modelos mais promissores e discutir as vantagens e desvantagens de cada técnica para este problema específico.

## Técnicas de Treinamento Selecionadas (Conforme Conteúdo Programático)

Para a tarefa de classificação de séries temporais de movimentos de LIBRAS, as seguintes técnicas do seu conteúdo programático são as mais adequadas e serão exploradas:

#### a. Redes Neurais Recorrentes (RNNs)

*   **Justificativa:** Dada a natureza sequencial e a complexidade dos dados (movimentos de LIBRAS), as Redes Neurais Recorrentes (RNNs), especialmente as variantes LSTM (Long Short-Term Memory), são a escolha mais natural e potente. Elas são projetadas para processar dados onde a ordem e as dependências temporais são cruciais, o que é exatamente o caso de uma sequência de movimentos.
*   **Implementação:** Utilizaremos a API Keras (TensorFlow).
*   **Conexão com o Material Anexado:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 15: "Processing Sequences Using RNNs and CNNs", Capítulo 11: "Training Deep Neural Networks") e *DeepLearningArchitetures.pdf* (Section 12.2: "Recurrent Neural Networks", Section 17: "Recurrent Neural Networks", Section 12.2.6: "Long-Term Dependencies and LSTM RNN").

#### b. Transformer

*   **Justificativa:** A arquitetura Transformer revolucionou o processamento de sequências, superando as RNNs em muitas tarefas devido à sua capacidade de processamento paralelo e à eficácia dos mecanismos de atenção para capturar dependências de longo alcance. Embora o dataset LIBRAS seja de coordenadas de movimento (séries temporais) e não texto, a arquitetura Transformer, com sua capacidade de capturar dependências de longo alcance via mecanismos de atenção, é perfeitamente aplicável e tem se mostrado eficaz em diversas modalidades de dados sequenciais. Para o dataset LIBRAS Movement, cada frame (com seus 90 atributos de coordenadas) pode ser tratado como um "token" em uma sequência de 45 "tokens". A habilidade do Transformer de "olhar" para todos os frames simultaneamente (via self-attention) pode ser crucial para entender a totalidade e o contexto de um movimento complexo.
*   **Implementação:** Implementaremos uma arquitetura Transformer simplificada (focando no Encoder) utilizando Keras/TensorFlow. Isso envolverá a criação de uma camada de "embedding" para cada frame (mapeando os 90 atributos para um espaço de maior dimensão) e a adição de codificação posicional para preservar a ordem temporal dos frames.

### 2. Modelos Não-Lineares: Support Vector Machines (SVMs)

*   **Justificativa:** SVMs são classificadores robustos e eficazes, especialmente com o "kernel trick" (kernel RBF), que permite capturar relações não-lineares nos dados de alta dimensionalidade. Servirão como uma poderosa *baseline* não-linear.
*   **Implementação:** Utilizaremos a biblioteca Scikit-learn.
*   **Referência:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 5: "Support Vector Machines"), *Murphy_Machine_Learning.pdf* (Capítulo 14: "Support Vector Machines"), *Statistical Inference and Machine Learning for Big Data.pdf* (Capítulo 11.5.5: "Support Vector Machines (SVM)").

### 3. Modelos Não-Lineares: Trees, Random Forests

*   **Justificativa:** Random Forests são um método de *ensemble* robusto, capaz de lidar com dados de alta dimensionalidade e menos sensíveis ao *overfitting* do que árvores individuais. Oferecem uma excelente *baseline* de *ensemble* não-linear.
*   **Implementação:** Utilizaremos a biblioteca Scikit-learn.
*   **Referência:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 6: "Decision Trees", Capítulo 7: "Ensemble Learning and Random Forests"), *Murphy_Machine_Learning.pdf* (Capítulo 16, Section 16.2: "Random Forests").

### 4. Modelos Lineares: Regressão Logística

*   **Justificativa:** Servirá como um **modelo de *baseline* linear, simples e interpretável**, fornecendo um ponto de comparação fundamental para avaliar o ganho de desempenho dos modelos mais complexos.
*   **Implementação:** Utilizaremos a biblioteca Scikit-learn.
*   **Referência:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 4: "Training Models"), *Statistical Inference and Machine Learning for Big Data.pdf* (Capítulo 11.5: "Logistic Regression"), *Murphy_Machine_Learning.pdf* (Capítulo 8, Section 8.1: "Logistic Regression").

### 5. Modelos Fatoriais e Redução de Dimensão: Análise de Componentes Principais (PCA)

*   **Justificativa:** Dada a alta dimensionalidade dos dados (4050 atributos se achatados), a PCA pode ser aplicada para compactar as características, remover ruído e potencialmente melhorar o desempenho e a eficiência computacional dos modelos subsequentes (especialmente SVMs e Regressão Logística).
*   **Implementação:** Utilizaremos a biblioteca Scikit-learn.
*   **Referência:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 8: "Dimensionality Reduction"), *Murphy_Machine_Learning.pdf* (Capítulo 12, Section 12.3: "Principal Component Analysis"), *Statistical Inference and Machine Learning for Big Data.pdf* (Capítulo 4.3.1: "Principal Component Analysis").

## Métricas de Avaliação

Para a classificação multiclasse dos movimentos de LIBRAS, as seguintes métricas serão utilizadas para avaliar o desempenho dos modelos:

*   **Acurácia (Accuracy)**
*   **Matriz de Confusão (Confusion Matrix)**
*   **Precision (Precisão)**
*   **Recall (Revocação)**
*   **F1-Score**
*   **Report de Classificação Completo**

*   **Conexão com o Material Anexado:** *Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-TensorFlow.pdf* (Capítulo 3: "Performance Measures", "Confusion Matrices", "Precision and Recall").

## Estrutura do Repositório GitHub

```
├── README.md ├── requirements.txt ├── data/ │ ├── movement_libras.data │ ├── movement_libras_1.data │ ├── movement_libras_5.data │ ├── movement_libras_8.data │ ├── movement_libras_9.data │ ├── movement_libras_10.data │ └── movement_libras.names ├── notebooks/ │ ├── 01_data_exploration.ipynb │ └── 02_model_prototyping.ipynb ├── src/ │ ├── init.py │ ├── data_loader.py │ ├── models.py │ ├── train.py │ ├── evaluate.py │ └── utils.py ├── trained_models/ │ ├── rnn_model_movement_libras.h5 │ ├── transformer_model_movement_libras.h5 # Novo exemplo de modelo Transformer │ ├── svm_model_movement_libras_1.pkl │ └── # ... outros modelos e datasets ├── results/ │ ├── confusion_matrix_rnn_movement_libras.png │ ├── confusion_matrix_transformer_movement_libras.png # Novo exemplo de resultado │ └── classification_report_svm_movement_libras_1.txt │ └── # ... outros resultados └── docs/ └── methodology_details.md

