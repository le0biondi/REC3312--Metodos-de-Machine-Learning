# Projeto: Reconhecimento de Movimentos da Língua Brasileira de Sinais (LIBRAS)

## Visão Geral do Projeto

Este projeto tem como objetivo principal desenvolver e comparar diferentes modelos de Machine Learning para a classificação de 15 movimentos distintos da Língua Brasileira de Sinais (LIBRAS). Utilizamos o **LIBRAS (Brazilian Sign Language) Movement Data Set** da UCI, que consiste em séries temporais de coordenadas 3D de movimentos de mão. O foco é explorar a capacidade de modelos tradicionais e, principalmente, de Redes Neurais Recorrentes (RNNs) e Redes Neurais Profundas (DNNs), em aprender padrões temporais complexos para identificar corretamente cada sinal.

Este trabalho serve como um protótipo fundamental para futuros desenvolvimentos de sistemas mais complexos de tradução contextual de LIBRAS.

## Conteúdo Programático Abordado

Este projeto abordará e aplicará ativamente os seguintes tópicos da disciplina de Métodos de Machine Learning:

*   **1 – Aprendizado Supervisionado:** O problema de classificação de movimentos é um exemplo clássico de aprendizado supervisionado, onde o modelo aprende a mapear features (coordenadas 3D) para rótulos (classes de movimento).
*   **1.1 Aprendizado estatístico – previsão, decomposição e classificação:** O foco principal está na **classificação** de movimentos, utilizando métricas estatísticas para avaliar o desempenho.
*   **2 – Modelos Lineares:** Serão utilizados como **modelos baseline** para comparação de desempenho, incluindo a **Regressão Logística**.
*   **2.2 Métodos de Shrinkage – Lasso e Elastic Net:** Podem ser considerados para **regularização** ou **seleção de features** em modelos lineares ou como parte da engenharia de features, caso a dimensionalidade inicial seja um desafio.
*   **3 – Métodos lineares de classificação:**
    *   **Análise Discriminante (LDA/QDA):** Pode ser explorada como um baseline discriminativo.
    *   **Regressão Logística:** Será implementada como um dos baselines para classificação multi-classe.
*   **4 – Métodos Não-paramétricos:**
    *   **4.2 Métodos de séries e kernel:** **Support Vector Machines (SVMs) com kernels não-lineares** serão explorados para capturar relações complexas nos dados, além das **Redes Neurais Recorrentes (RNNs)** que são naturalmente adequadas para séries temporais.
*   **5 – Modelos fatoriais e redução de dimensão:**
    *   **5.2 Análise de Componentes Principais (PCA):** Pode ser aplicada durante a Análise Exploratória de Dados (EDA) para visualização ou para **redução de dimensionalidade** das features, otimizando o tempo de treinamento para alguns modelos.
*   **6 – Modelos não lineares:**
    *   **6.1 Trees, Random Forests:** Serão implementados como **baselines não-lineares** robustos, conhecidos por sua capacidade de lidar com dados complexos.
    *   **6.2 Support Vector Machines:** Serão utilizados com diferentes kernels para explorar sua eficácia em nosso problema de classificação.
*   **7 – Redes Neurais e Deep Learning:** O ponto central do projeto.
    *   **7.1 Redes Neurais (FCNNs):** Implementação de **Redes Neurais Totalmente Conectadas** para verificar seu desempenho em relação aos modelos sequenciais.
    *   **7.2 Deep Learning (RNNs, LSTMs, GRUs):** O foco principal, com implementação de **Redes Neurais Recorrentes**, como **LSTMs (Long Short-Term Memory)** e **GRUs (Gated Recurrent Units)**, que são arquiteturas ideais para processar e aprender padrões a partir de dados sequenciais, como as coordenadas de movimento no tempo.

## Dataset

O dataset utilizado é o **LIBRAS (Brazilian Sign Language) Movement Data Set**, disponível no UCI Machine Learning Repository.

*   **Fonte:** [https://archive.ics.uci.edu/dataset/181/libras+movement](https://archive.ics.uci.edu/dataset/181/libras+movement)
*   **Descrição:** O dataset consiste em 15 classes de movimentos de mão em LIBRAS. Cada classe possui 24 instâncias (totalizando 360 amostras). Cada amostra é uma **série temporal de 45 valores reais**, que representam as coordenadas da mão em um espaço 3D (assumimos que são 15 pontos x,y,z ou 15 frames com x,y,z, estruturados como uma sequência de 15 timesteps com 3 features cada). Os dados já estão pré-normalizados no intervalo `[-1, 1]`. O arquivo `libras-movement.data` contém os 45 atributos de features seguidos pelo rótulo da classe (1 a 15).

## Estrutura do Repositório

-   `README.md`: Este arquivo, fornecendo uma visão geral completa do projeto.
-   `requirements.txt`: Lista todas as dependências Python necessárias para replicar o ambiente do projeto.
-   `.gitignore`: Define quais arquivos e diretórios o Git deve ignorar (ex: ambientes virtuais, modelos salvos, arquivos temporários).
-   `data/`: Diretório para armazenar o dataset original.
    -   `libras_movement.data`: O arquivo de dados brutos baixado do UCI.
-   `notebooks/`: Contém os Jupyter Notebooks para exploração de dados e análises aprofundadas.
    -   `01_data_exploration.ipynb`: Para análise exploratória de dados (EDA), visualização da estrutura e características do dataset.
    -   `02_model_analysis.ipynb`: Para comparar o desempenho dos modelos, analisar erros e gerar visualizações finais.
-   `src/`: Contém o código-fonte principal do projeto, organizado em módulos para melhor manutenibilidade e clareza.
    -   `data_loader.py`: Funções para carregar o dataset, realizar pré-processamento e dividir os dados em conjuntos de treinamento, validação e teste.
    -   `baseline_models.py`: Implementa e treina os modelos de Machine Learning tradicionais (baselines).
    -   `deep_learning_models.py`: Define e treina as arquiteturas de Redes Neurais (FCNNs, RNNs, LSTMs, GRUs).
    -   `train_evaluate.py`: O script principal que orquestra todo o pipeline: carregamento de dados, treinamento de todos os modelos, avaliação e salvamento de resultados.
-   `models/`: Diretório para salvar os modelos treinados (por exemplo, arquivos `.pkl` para Scikit-learn, `.h5` para Keras/TensorFlow).
-   `reports/`: Diretório para armazenar relatórios de desempenho, matrizes de confusão e outros gráficos de avaliação.

## Como Configurar e Rodar o Projeto

Siga os passos abaixo para configurar o ambiente e executar o projeto.

### 1. Clonar o Repositório

Primeiro, clone este repositório para sua máquina local usando o Git:

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <nome_do_seu_repositorio>
