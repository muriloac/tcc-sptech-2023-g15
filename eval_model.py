"""
eval_model
==========

Este módulo fornece funções para carregar conjuntos de dados de teste, fazer previsões e avaliar modelos em PyTorch.

Funcionalidades
---------------
- **load_test_dataset**: Carrega o conjunto de dados de teste.
- **make_predictions**: Realiza previsões usando o modelo fornecido.
- **calculate_aucs**: Calcula a AUC (Área Sob a Curva ROC) para cada rótulo.
- **make_pred_multilabel**: Realiza previsões multirrótulo e calcula a AUC para cada rótulo.

.. note::
   Este módulo faz uso extensivo do PyTorch e presume que os modelos usados são baseados em arquiteturas de redes neurais profundas.
"""

# Imports para registro de logs e manipulação de avisos
import logging  #: Fornece funções e classes para registro de mensagens em uma ampla variedade de situações.
import warnings  #: Módulo de avisos para emitir e controlar a exibição de alertas.

# Imports para manipulação de dados e operações matemáticas
import numpy as np  #: Módulo do pacote NumPy para trabalhar com arrays e matrizes.
import pandas as pd  #: Módulo do pacote Pandas para estruturas de dados e funções para dados estruturados.
import sklearn.metrics as sklm  #: Submódulo de métricas do scikit-learn para funções de pontuação e métricas.
# Imports relacionados ao PyTorch
from torch.autograd import Variable  #: Diferenciação automática de operações em tensores.
from torch.utils.data import DataLoader  #: Utilitários para carregar e transformar conjuntos de dados.

# Imports específicos do projeto
import cxr_dataset as CXR  #: Módulo personalizado para conjuntos de dados de radiografias.

# Configuração inicial
logging.basicConfig(level=logging.INFO)  #: Configuração inicial para registro de logs.
warnings.simplefilter(action='ignore', category=FutureWarning)  #: Ignora avisos sobre características descontinuadas.


def load_test_dataset(data_transforms, PATH_TO_IMAGES, fold="test"):
    """
    Carrega o conjunto de dados de teste.

    Esta função carrega o conjunto de dados de teste usando o módulo `CXR` e verifica se o conjunto de dados não está vazio.

    :param data_transforms: Um dicionário contendo as transformações a serem aplicadas aos dados.
        Espera-se que tenha uma chave 'val' para as transformações de validação/teste.
    :type data_transforms: dict

    :param PATH_TO_IMAGES: Caminho para o diretório que contém as imagens.
    :type PATH_TO_IMAGES: str

    :param fold: Indica a pasta (ou "fold") do conjunto de dados a ser carregado. Por padrão, é "test".
    :type fold: str, optional

    :returns: O conjunto de dados carregado.
    :rtype: CXR.CXRDataset

    :raises ValueError: Se o conjunto de dados estiver vazio.
    """
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold=fold,
        transform=data_transforms['val']
    )

    if len(dataset) == 0:
        logging.error("O conjunto de dados de teste está vazio!")
        raise ValueError("O conjunto de dados de teste está vazio. Por favor, verifique seus dados.")

    return dataset


def make_predictions(model, dataloader):
    """
    Realiza previsões usando um modelo fornecido e um carregador de dados.

    Esta função percorre o carregador de dados (dataloader) e usa o modelo para fazer previsões. As previsões e os
    rótulos reais são salvos em DataFrames do Pandas para posterior análise.

    :param model: O modelo a ser utilizado para fazer previsões.
    :type model: torch.nn.Module

    :param dataloader: Carregador de dados contendo as imagens e rótulos para fazer as previsões.
    :type dataloader: torch.utils.data.DataLoader

    :returns: Dois DataFrames do Pandas: o primeiro contendo as previsões e o segundo contendo os rótulos verdadeiros.
    :rtype: tuple(pd.DataFrame, pd.DataFrame)
    """
    BATCH_SIZE = 16
    model.train(False)
    size = len(dataloader.dataset)

    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    for i, data in enumerate(dataloader, start=0):
        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        for j, (actual, pred) in enumerate(zip(true_labels, probs)):
            thisrow = {"Image Index": dataloader.dataset.df.index[BATCH_SIZE * i + j]}
            truerow = {"Image Index": dataloader.dataset.df.index[BATCH_SIZE * i + j]}

            for k, label in enumerate(dataloader.dataset.PRED_LABEL):
                thisrow[f"prob_{label}"] = pred[k]
                truerow[label] = actual[k]

            pred_df = pd.concat([pred_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)
            true_df = pd.concat([true_df, pd.DataFrame(truerow, index=[0])], ignore_index=True)

        if i % 10 == 0:
            logging.info(str(i * BATCH_SIZE))

    return pred_df, true_df


def calculate_aucs(pred_df, true_df):
    """
    Calcula a área sob a curva (AUC) para previsões realizadas.

    Esta função calcula o valor AUC para cada rótulo no DataFrame fornecido.
    Se não for possível calcular o AUC devido a um erro (por exemplo, se todas as classes verdadeiras forem iguais),
    o valor será registrado como NaN.

    :param pred_df: DataFrame contendo as probabilidades previstas, com colunas no formato "prob_LABEL".
    :type pred_df: pd.DataFrame

    :param true_df: DataFrame contendo os rótulos verdadeiros.
    :type true_df: pd.DataFrame

    :returns: DataFrame contendo os rótulos e seus valores AUC correspondentes.
    :rtype: pd.DataFrame
    """
    auc_df = pd.DataFrame(columns=["label", "auc"])

    for column in true_df.columns:
        if column == "Image Index":
            continue
        actual = true_df[column]
        pred = pred_df[f"prob_{column}"]
        thisrow = {"label": column, "auc": np.nan}
        try:
            thisrow["auc"] = sklm.roc_auc_score(actual.values.astype(int), pred.values)
        except Exception as e:
            logging.error(f"Não foi possível calcular AUC para {column}. Erro: {e}")
        auc_df = pd.concat([auc_df, pd.DataFrame(thisrow, index=[0])], ignore_index=True)

    return auc_df


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES, fold="test"):
    """
    Realiza previsões multirrótulo para um conjunto de dados e calcula a AUC para cada rótulo.

    Esta função utiliza um modelo fornecido para realizar previsões multirrótulo em um conjunto de dados
    especificado e, em seguida, calcula a AUC para cada rótulo. Os resultados das previsões e AUC são salvos
    em arquivos CSV.

    :param data_transforms: Um dicionário contendo as transformações a serem aplicadas aos dados.
        Espera-se que tenha uma chave 'val' para as transformações de validação/teste.
    :type data_transforms: dict

    :param model: O modelo a ser utilizado para fazer previsões.
    :type model: torch.nn.Module

    :param PATH_TO_IMAGES: Caminho para o diretório que contém as imagens.
    :type PATH_TO_IMAGES: str

    :param fold: Indica a pasta (ou "fold") do conjunto de dados a ser carregado. Por padrão, é "test".
    :type fold: str, optional

    :returns: Dois DataFrames do Pandas: o primeiro contendo as previsões e o segundo contendo os valores AUC para cada rótulo.
    :rtype: tuple(pd.DataFrame, pd.DataFrame)
    """
    dataset = load_test_dataset(data_transforms, PATH_TO_IMAGES, fold)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=8)

    pred_df, true_df = make_predictions(model, dataloader)
    auc_df = calculate_aucs(pred_df, true_df)

    with open("results/preds.csv", "w") as f:
        pred_df.to_csv(f, index=False)
    with open("results/aucs.csv", "w") as f:
        auc_df.to_csv(f, index=False)

    return pred_df, auc_df
