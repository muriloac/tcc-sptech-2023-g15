import base64
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from flask import Flask, request
from torchcam.methods import XGradCAM
from torchvision import transforms

app = Flask(__name__)

# Médias de normalização para as imagens
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
# Desvios padrão de normalização para as imagens
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Lista de possíveis achados em imagens médicas
FINDINGS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Suponha que model seja o seu modelo carregado
model = torch.load('results/densenet_best.pth')['model']
model.eval()
model.cpu()

label_baseline_probs = {
    'Atelectasis': 0.103,
    'Cardiomegaly': 0.025,
    'Effusion': 0.119,
    'Infiltration': 0.177,
    'Mass': 0.051,
    'Nodule': 0.056,
    'Pneumonia': 0.012,
    'Pneumothorax': 0.047,
    'Consolidation': 0.042,
    'Edema': 0.021,
    'Emphysema': 0.022,
    'Fibrosis': 0.015,
    'Pleural_Thickening': 0.03,
    'Hernia': 0.002
}


# Função para carregar a imagem e transformá-la em um tensor
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


# Função para obter as principais probabilidades e índices
def get_top_probabilities(output, topk=3):
    # Aplicar softmax para obter probabilidades
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_idxs = torch.topk(probabilities, topk)
    return top_probs.squeeze(), top_idxs.squeeze()


# Função para gerar o CAM (Class Activation Map) para uma imagem
def generate_cam(image_tensor, model, top_idx, original_image_size):
    # Inicializar o extrator de CAM com o modelo e a camada-alvo
    cam_extractor = XGradCAM(model,
                             target_layer=model.features.norm5)  # Verifique se 'norm5' é o correto para o seu modelo

    # Realizar a passagem para frente através do modelo
    out = model(image_tensor)
    # Calcular o CAM para a classe de interesse
    activation_map = cam_extractor(out.squeeze().argmax().item(), out)

    # Verificar se o mapa de ativação é uma lista e pegar o primeiro item se for
    if isinstance(activation_map, list):
        activation_map = activation_map[0]

    # Redimensionar o CAM para o tamanho da imagem de entrada
    result = transforms.Resize(original_image_size)(activation_map.unsqueeze(0))

    # Normalizar o CAM para ter valores entre 0 e 1 para melhor visualização
    result = result.squeeze().numpy()
    result = (result - result.min()) / (result.max() - result.min())

    # Inverter o mapa de ativação para garantir que valores mais altos correspondam a cores mais quentes
    result = 1 - result

    return result


# Função para processar uma imagem e gerar resultados
def process_image(image_path, model):
    image_tensor = load_image(image_path)
    original_image = Image.open(image_path).convert('RGB')
    original_image_size = original_image.size  # Obter o tamanho da imagem original
    output = model(image_tensor)
    top_probs, top_idxs = get_top_probabilities(output)

    # Gerar o CAM para cada um dos top_idxs
    cams = [generate_cam(image_tensor, model, idx, original_image_size) for idx in top_idxs]

    # Exibir a imagem original e os CAMs
    plt.figure(figsize=(20, 10))
    for i, (cam, idx) in enumerate(zip(cams, top_idxs)):
        plt.subplot(1, len(top_idxs), i + 1)
        plt.imshow(original_image, alpha=1)
        plt.imshow(cam, cmap='hot', alpha=0.5)  # Ajustar a transparência para o CAM
        plt.title(f"{FINDINGS[idx]}: {top_probs[i].item():.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

    # Apresentar os resultados em texto
    print(f"As principais probabilidades são: {top_probs.detach().numpy()}")
    print(f"Achados correspondentes: {[FINDINGS[i] for i in top_idxs]}")


# Rota para processar uma imagem enviada via POST
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    file = request.files['image']
    if file.filename == '':
        return 'Nenhum arquivo selecionado', 400

    try:
        image_tensor = load_image(file.stream)
        original_image = Image.open(file.stream).convert('RGB')
        original_image_size = original_image.size
        output = model(image_tensor)

        # Subtrair as probabilidades de referência
        baseline_probs_tensor = torch.tensor(list(label_baseline_probs.values()))
        output = torch.sub(output, baseline_probs_tensor)

        top_probs, top_idxs = get_top_probabilities(output)

        # Gerar o CAM apenas para os top_idxs com maior probabilidade
        cam = generate_cam(image_tensor, model, top_idxs[0], original_image_size)

        # Salvar as imagens CAM e a imagem original com a sobreposição de CAM em objetos BytesIO

        # Criar um novo buffer de bytes para a imagem CAM
        cam_buffer = BytesIO()

        # Converter o CAM para uma imagem PIL e salvar no buffer
        cam_image = Image.fromarray(np.uint8(plt.cm.hot(cam) * 255))
        plt.imshow(original_image, alpha=1)
        plt.imshow(cam_image, alpha=0.5)  # Ajustar a transparência para o CAM
        plt.axis('off')

        # Salvar a imagem composta em um buffer de bytes
        plt.savefig(cam_buffer, format='PNG')
        cam_buffer.seek(0)  # Voltar para o início do buffer de bytes
        plt.close()

        response = {
            'top_findings': [FINDINGS[i] for i in top_idxs],
            'cam_image': base64.b64encode(cam_buffer.getvalue()).decode('utf-8')
        }

        return response

    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
