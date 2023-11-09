"""
data_utils
==========

Este módulo fornece utilitários para manipulação de dados, treinamento e avaliação de modelos em PyTorch.

Funcionalidades
---------------
- **checkpoint**: Salva o checkpoint do modelo.
- **train_model**: Treina o modelo e retorna o melhor modelo e a melhor epoch.
- **train_cnn**: Treina um modelo torchvision com os dados NIH dados hiperparâmetros de alto nível.

.. note::
   Este módulo faz uso extensivo do PyTorch e presume que os modelos usados são baseados em arquiteturas de redes neurais profundas.
"""

# Imports para manipulação de dados
import csv  #: Para leitura/gravação de arquivos CSV.
# Imports para registro de logs e acompanhamento do progresso
import logging  #: Para registrar mensagens de log.
import os  #: Para manipulação de caminhos e arquivos no sistema de arquivos.
import time  #: Para medir o tempo de execução.
from shutil import rmtree  #: Para remover diretórios e seu conteúdo.

# Imports relacionados ao PyTorch
import torch  #: O PyTorch em si.
import torch.nn as nn  #: Para construir modelos de rede neural.
import torch.optim as optim  #: Para otimizar modelos.
from torch.utils.data import Dataset  #: Para criar conjuntos de dados personalizados.
from torchvision import models  #: Para importar modelos pré-treinados.
from torchvision import transforms  #: Para aplicar transformações em imagens.
from torchvision.models import DenseNet121_Weights  #: Importa os pesos pré-treinados do modelo DenseNet121.
from tqdm import tqdm  #: Para exibir barras de progresso durante loops.

# Imports específicos do projeto
import cxr_dataset as CXR  #: Módulo personalizado para o conjunto de dados de radiografias.
import eval_model as E  #: Módulo personalizado para avaliação do modelo.
from config_variables import BATCH_SIZE, NUM_EPOCHS  #: Importa constantes de configuração.


def checkpoint(model, best_loss, epoch, LR, save_path='results-ia', filename_prefix='model'):
    """
    Salva o checkpoint do modelo. O nome do arquivo do checkpoint é composto por um prefixo, o número da Epoch e a
    perda de validação do modelo.

    :param model: O modelo a ser salvo.
    :type model: torch.nn.Module
    :param best_loss: A melhor perda alcançada até agora.
    :type best_loss: float
    :param epoch: O número da Epoch atual.
    :type epoch: int
    :param LR: A taxa de aprendizado atual.
    :type LR: float
    :param save_path: O diretório onde o checkpoint será salvo. O padrão é 'results-ia'.
    :type save_path: str, opcional
    :param filename_prefix: O prefixo para o nome do arquivo do checkpoint. O padrão é 'model'.
    :type filename_prefix: str, opcional
    :returns: None

    **Exemplo de Uso**:

    .. code-block:: python

       model = MyModel()
       best_loss = 0.05
       epoch = 10
       LR = 0.001
       checkpoint(model, best_loss, epoch, LR)

    **Logs**:

    - "Salvando checkpoint para [caminho_do_arquivo]..." ao iniciar o salvamento.
    - "Checkpoint salvo com sucesso." após salvar com sucesso.
    - "Erro ao salvar o checkpoint. [detalhes_do_erro]" em caso de erro.
    """

    try:
        # Criar o diretório se ele não existir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Criar o nome do arquivo com o prefixo, número da epoch e perda de validação
        filename = f"{filename_prefix}_epoch{epoch}_loss{best_loss:.4f}.pth"
        filepath = os.path.join(save_path, filename)

        logging.info('Salvando checkpoint para %s...', filepath)

        state = {
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
            'rng_state': torch.get_rng_state(),
            'LR': LR,
            'model': model
        }

        torch.save(state, filepath)
        logging.info('Checkpoint salvo com sucesso.')

    except Exception as e:
        logging.error('Erro ao salvar o checkpoint. %s', str(e))


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        save_path='results-ia',
        filename_prefix='model'):
    """
    Treina um modelo utilizando PyTorch.

    :param model: Modelo a ser treinado.
    :type model: torch.nn.Module
    :param criterion: Critério de perda (e.g., BCELoss).
    :type criterion: callable
    :param optimizer: Otimizador a ser utilizado no treinamento (e.g., SGD).
    :type optimizer: torch.optim.Optimizer
    :param LR: Taxa de aprendizado.
    :type LR: float
    :param num_epochs: Continue treinando até este número de epochs.
    :type num_epochs: int
    :param dataloaders: Dataloaders de treinamento e validação do PyTorch.
    :type dataloaders: dict
    :param dataset_sizes: Tamanho dos datasets de treinamento e validação.
    :type dataset_sizes: dict
    :param save_path: Diretório para salvar checkpoints e logs. Padrão é 'results-ia'.
    :type save_path: str, opcional
    :param filename_prefix: Prefixo para nomes de arquivos de checkpoint. Padrão é 'model'.
    :type filename_prefix: str, opcional
    :returns:
      - model: Modelo treinado.
      - best_epoch: Epoch na qual o modelo alcançou sua melhor perda de validação.
    :rtype: Tuple[torch.nn.Module, int]

    **Exemplo de Uso**:

    .. code-block:: python

       model = MyModel()
       criterion = torch.nn.BCELoss()
       optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
       LR = 0.001
       num_epochs = 10
       dataloaders = {"train": train_loader, "val": val_loader}
       dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
       train_model(model, criterion, optimizer, LR, num_epochs, dataloaders, dataset_sizes)

    **Logs**:

    - "Epoch [número_da_epoch]/[epochs_totais]" no início de cada epoch.
    - "-----" como separador.
    - "[fase] Loss: [valor_da_loss]" após cada fase de treinamento ou validação.
    - "Sem melhoria em 3 epochs, parando." se não houver melhoria após 3 epochs.
    - "Treinamento completo em [minutos]m [segundos]s" ao concluir o treinamento.
    - "Melhor Epoch: [melhor_epoch], Melhor perda: [melhor_loss]" ao finalizar o treinamento.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    best_loss = float('inf')
    best_epoch = -1
    last_train_loss = -1

    log_path = os.path.join(save_path, "log_train.csv")
    with open(log_path, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, num_epochs + 1):
            logging.info('Epoch {}/{}'.format(epoch, num_epochs))
            logging.info('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                for data in tqdm(dataloaders[phase]):
                    inputs, labels, _ = data
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]

                if phase == 'train':
                    last_train_loss = epoch_loss
                else:
                    scheduler.step(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_epoch = epoch
                        checkpoint(model, best_loss, epoch, LR, save_path, filename_prefix)

                logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'val':
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

            if (epoch - best_epoch) >= 3:
                logging.info("Sem melhoria em 3 epochs, parando.")
                break

    time_elapsed = time.time() - since
    logging.info('Treinamento completo em {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Melhor Epoch: %d, Melhor loss: %.4f', best_epoch, best_loss)

    os.rename(os.path.join(save_path, f"{filename_prefix}_epoch{best_epoch}_loss{best_loss:.4f}.pth"),
              os.path.join(save_path, f"{filename_prefix}_best.pth"))

    checkpoint_best = torch.load(
        os.path.join(save_path, f"{filename_prefix}_best.pth"),
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint_best['model_state_dict'])

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Treina um modelo torchvision com os dados NIH dados hiperparâmetros de alto nível.

    :param PATH_TO_IMAGES: Caminho para as imagens NIH.
    :type PATH_TO_IMAGES: str
    :param LR: Taxa de aprendizado.
    :type LR: float
    :param WEIGHT_DECAY: Parâmetro de decaimento de peso para SGD.
    :type WEIGHT_DECAY: float
    :returns:
       - preds: Previsões do modelo torchvision no conjunto de teste com ground truth para comparação.
       - aucs: AUCs para cada par de treino e teste.
    :rtype: Tuple[torch.Tensor, List[float]]

    Pré-requisitos:
    - Espera-se que as constantes `BATCH_SIZE` e `NUM_EPOCHS` estejam definidas globalmente.
    - O módulo `CXR` deve estar disponível e contém a classe `CXRDataset`.
    - O módulo `E` deve estar disponível e contém a função `make_pred_multilabel`.

    **Exemplo de Uso**:

    .. code-block:: python

       PATH_TO_IMAGES = "caminho/para/imagens/"
       LR = 0.001
       WEIGHT_DECAY = 0.0001
       preds, aucs = train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY)

    **Logs**:

    - "Número de GPUs ativas: [número_de_gpus]" no início do treinamento.
    - Exceção "Erro, requer GPU" se a GPU não estiver disponível.
    """

    if not torch.cuda.is_available():
        raise ValueError("Erro, GPU necessária")

    print(f"Número de GPUs ativas: {torch.cuda.device_count()}")

    # Remove o diretório de resultados se ele já existir
    if os.path.exists('results-ia/'):
        rmtree('results-ia/')
    os.makedirs("results-ia/")

    # Imagenet mean e std para normalização
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    N_LABELS = 14

    # Define transformações para os dados
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Cria datasets e dataloaders
    transformed_datasets = {
        'train': CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='train',
            transform=data_transforms['train']),
        'val': CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold='val',
            transform=data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            transformed_datasets['train'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8),
        'val': torch.utils.data.DataLoader(
            transformed_datasets['val'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8)
    }

    # Inicializa o modelo
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    """
    1 7x7 Convolution
    58 3x3 Convolution
    61 1x1 Convolution
    4 AvgPool
    1 Fully Connected Layer
    
    121 layers no total
    """
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    model = model.cuda()

    # Define a função de perda e o otimizador
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes)

    preds, aucs = E.make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs
