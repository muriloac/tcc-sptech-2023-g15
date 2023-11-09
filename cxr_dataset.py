"""
cxr_dataset
===========

Este módulo contém uma classe para representar o conjunto de dados de radiografias (CXR).

A classe `CXRDataset` permite carregar, filtrar e transformar imagens para treinamento e avaliação de modelos de
aprendizado de máquina. Além disso, fornece métodos para filtrar os dados por descobertas específicas,
obter o tamanho do conjunto de dados e acessar itens individuais do conjunto de dados.

.. note::
   Este módulo presume que os conjuntos de dados consistem em imagens de radiografias e metadados associados.
"""

# Imports para registro de logs e manipulação de arquivos do sistema
import logging  #: Fornece funções e classes para registro de mensagens em uma ampla variedade de situações.
import os  #: Para manipulação de caminhos e arquivos no sistema de arquivos.

# Imports para manipulação de dados, operações matemáticas e processamento de imagens
import numpy as np  #: Módulo do pacote NumPy para trabalhar com arrays e matrizes.
import pandas as pd  #: Módulo do pacote Pandas para estruturas de dados e funções para dados estruturados.
from PIL import Image  #: Módulo para manipulação e tratamento de imagens.

# Imports específicos do projeto
from config_variables import SAMPLE_SIZE  #: Importa uma constante de configuração.

# Configuração inicial
logging.basicConfig(level=logging.INFO)  #: Configuração inicial para registro de logs.


class CXRDataset:
    """
    Classe para representar o conjunto de dados de radiografias (CXR).

    A classe permite carregar, filtrar e transformar imagens para treinamento e avaliação de modelos de aprendizado de máquina.

    **Atributos**:
       - transform: Transformações a serem aplicadas às imagens.
       - path_to_images: Caminho onde as imagens estão armazenadas.
       - df: DataFrame contendo metadados e rótulos das imagens.
       - PRED_LABEL: Lista de rótulos para classificação.

    **Exemplo de Uso**:

    .. code-block:: python

        dataset = CXRDataset(path_to_images="caminho/para/imagens/", fold="train", labels_file="labels.csv")

    """

    def __init__(self, path_to_images, fold, labels_file="nih_labels.csv", transform=None, sample=SAMPLE_SIZE,
                 finding="any", starter_images=False, labels=None):
        """
        Inicializa o conjunto de dados.

        :param path_to_images: Caminho para onde as imagens estão armazenadas.
        :param fold: Identificador para subconjuntos de dados.
        :param labels_file: Nome do arquivo contendo rótulos. Padrão é "nih_labels.csv".
        :param transform: Transformações a serem aplicadas nas imagens.
        :param sample: Quantidade de amostras a serem retiradas do conjunto de dados.
        :param finding: Tipo específico de descoberta para filtrar no conjunto de dados.
        :param starter_images: Se True, filtra o conjunto de dados com base em um arquivo chamado "starter_images.csv".
        :param labels: Lista de rótulos para o conjunto de dados. Se None, uma lista padrão será usada.
        """
        self.transform = transform
        self.path_to_images = path_to_images
        if not os.path.isdir(self.path_to_images):
            logging.error(f"O caminho fornecido {self.path_to_images} não existe ou não é um diretório.")
            raise ValueError("Caminho inválido para as imagens.")

        self.df = self.load_labels_file(labels_file)
        self.df = self.df[self.df['fold'] == fold]

        if starter_images:
            self.df = self.merge_with_starter_images()

        if 0 < sample < len(self.df):
            self.df = self.df.sample(sample, random_state=42)

        self.df = self.filter_by_finding(finding)

        self.df = self.df.set_index("Image Index")

        if labels:
            self.PRED_LABEL = labels
        else:
            self.PRED_LABEL = [
                'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
            ]

    def load_labels_file(self, labels_file):
        """
        Carrega o arquivo de rótulos (labels).

        :param labels_file: Caminho do arquivo contendo os rótulos.
        :return: DataFrame do Pandas com os rótulos carregados.
        :raises: Exception em caso de erro ao carregar o arquivo.

        **Exemplo de Uso**:

        .. code-block:: python

            df_labels = load_labels_file("caminho/para/arquivo_labels.csv")
        """
        try:
            return pd.read_csv(labels_file)
        except Exception as e:
            logging.error(f"Erro ao carregar o arquivo de rótulos {labels_file}. Erro: {e}")
            raise

    def merge_with_starter_images(self):
        """
        Mescla o DataFrame atual com o arquivo "starter_images.csv".

        :return: DataFrame do Pandas mesclado.
        :raises: Exception em caso de erro na mesclagem.

        **Exemplo de Uso**:

        .. code-block:: python

            df_mesclado = merge_with_starter_images()
        """
        try:
            starter_images = pd.read_csv("starter_images.csv")
            return pd.merge(left=self.df, right=starter_images, how="inner", on="Image Index")
        except Exception as e:
            logging.error(f"Erro na mesclagem com as imagens iniciais. Erro: {e}")
            raise

    def filter_by_finding(self, finding):
        """
        Filtra o DataFrame com base em uma determinada descoberta.

        :param finding: Descoberta ou rótulo específico para filtrar.
        :return: DataFrame filtrado.
        :raises: ValueError em caso de descoberta inválida.

        **Exemplo de Uso**:

        .. code-block:: python

            df_filtrado = filter_by_finding("Pneumonia")
        """
        if finding != "any":
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    return self.df[self.df[finding] == 1]
                else:
                    logging.warning(f"Sem casos positivos para {finding}. Retornando casos não filtrados.")
            else:
                logging.error(
                    f"Não é possível filtrar pela descoberta {finding} pois ela não está nos dados. Por favor, verifique a ortografia.")
                raise ValueError(f"Descoberta inválida: {finding}")
        return self.df

    def __len__(self):
        """
        Retorna o número de elementos no DataFrame.

        :return: Inteiro representando o número de elementos.

        **Exemplo de Uso**:

        .. code-block:: python

            dataset = CXRDataset(...)
            tamanho = len(dataset)
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Obtém um item do dataset dado um índice.

        :param idx: Índice do elemento desejado.
        :return: Uma tupla contendo a imagem, o rótulo e o índice da imagem.

        **Exemplo de Uso**:

        .. code-block:: python

            dataset = CXRDataset(...)
            imagem, rótulo, índice = dataset[5]
        """
        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0:
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return image, label, self.df.index[idx]
