#: Caminho para a pasta que contém as imagens.
#: Este é um caminho relativo que aponta para uma pasta chamada "images" no diretório atual.
PATH_TO_IMAGES = "./images/"

#: Constante de decaimento de peso.
#: Usado geralmente em otimizadores para regularizar e evitar o overfitting.
WEIGHT_DECAY = 1e-4

#: Taxa de aprendizado para o otimizador.
#: Determina o tamanho do passo durante a otimização.
LEARNING_RATE = 0.01

#: Número total de épocas para treinar o modelo.
#: Uma época é uma passagem completa através do conjunto de dados.
NUM_EPOCHS = 20

#: Tamanho do lote usado durante o treinamento.
#: Refere-se ao número de amostras que serão propagadas pela rede.
BATCH_SIZE = 14

#: Tamanho da amostra para alguma operação (por exemplo, visualização ou debug).
#: Refere-se ao número de amostras que serão usadas para a tarefa específica.
#: 0 significa que todas as amostras serão usadas.
SAMPLE_SIZE = 0
