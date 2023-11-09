"""
main
====

Este script é o ponto de entrada principal para treinar uma rede neural convolucional (CNN) usando um conjunto de dados de radiografias.

Ao ser executado, este script importa variáveis de configuração, como o caminho para as imagens, a taxa de aprendizado e o decaimento de peso. Em seguida, treina uma CNN usando essas configurações e recupera as previsões e as AUCs.

.. note::
   Este script presume que o conjunto de dados e as configurações relevantes estão disponíveis e corretamente configurados.

Uso
---
Para executar este script, use o seguinte comando:

.. code-block:: bash

   $ python main.py

"""

#: Importa variáveis de configuração do módulo `config_variables`.
#: As variáveis incluem o caminho para as imagens, taxa de aprendizado e decaimento de peso.
from config_variables import PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY

#: Importa a função `train_cnn` do módulo `data_utils`.
#: Esta função é responsável por treinar uma rede neural convolucional.
from data_utils import train_cnn

#: Verifica se este script é o ponto de entrada principal.
#: Isso é útil para permitir que o script seja importado como um módulo ou executado como um script autônomo.
if __name__ == '__main__':
    #: Treina a CNN e recupera as previsões e as AUCs.
    preds, aucs = train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)
