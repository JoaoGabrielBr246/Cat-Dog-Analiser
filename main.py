# -*- coding: utf-8 -*-
"""cats-vs-dogs.ipynb

Este arquivo foi gerado automaticamente pelo Google Colab.

Localização original do arquivo:
    https://colab.research.google.com/drive/1ADJyCjz66r0U6Hw-CyZc3yIf1qFxNgfw
"""

# Monta o Google Drive para acesso aos arquivos
from google.colab import drive
drive.mount('/content/drive')

# Instala a biblioteca TensorFlow, que é usada para aprendizado de máquina
# !pip install tensorflow

# Importa as bibliotecas necessárias do TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define o diretório base onde as imagens de treinamento estão localizadas
base_dir = '/content/drive/MyDrive/Colab Notebooks/cats-dogs/PetImages'  # Atualize o caminho para a pasta correta

# Cria um gerador de dados para pré-processar as imagens de treinamento
train_datagen = ImageDataGenerator(rescale=1.0/255.0,  # Normaliza os pixels da imagem para a faixa [0, 1]
                                    rotation_range=20,  # Rotaciona a imagem aleatoriamente em até 20 graus
                                    width_shift_range=0.2,  # Desloca a imagem horizontalmente em até 20% da largura
                                    height_shift_range=0.2,  # Desloca a imagem verticalmente em até 20% da altura
                                    shear_range=0.2,  # Aplica uma transformação de cisalhamento
                                    zoom_range=0.2,  # Aplica um zoom aleatório de até 20%
                                    horizontal_flip=True,  # Inverte horizontalmente as imagens
                                    fill_mode='nearest')  # Preenche os pixels ausentes com os valores mais próximos

# Cria um gerador de dados que flui as imagens do diretório e as prepara para treinamento
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # Redimensiona as imagens para 224x224 pixels
    batch_size=32,  # Tamanho do lote (quantidade de imagens processadas ao mesmo tempo)
    class_mode='binary')  # Para classificação binária (gato ou cachorro)

# Carrega o modelo MobileNetV2 pré-treinado, excluindo a parte superior
base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congela a base do modelo para evitar o treinamento

# Cria um novo modelo adicionando camadas ao modelo base
model = models.Sequential([
    base_model,  # Adiciona o modelo base
    layers.GlobalAveragePooling2D(),  # Reduz a dimensionalidade da saída do modelo base
    layers.Dense(1, activation='sigmoid')  # Camada densa com ativação sigmoide para saída binária
])

# Compila o modelo, definindo o otimizador, a função de perda e as métricas
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treina o modelo com os dados gerados
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de passos por época
                    epochs=10)  # Número de épocas (iterações sobre todo o conjunto de dados)

# Salva o modelo treinado em um arquivo .keras no Google Drive
model.save('/content/drive/MyDrive/Colab Notebooks/cats-dogs/PetImages/my_model.keras')

# Baixa arquivos necessários para a detecção de objetos usando YOLO
# !wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O yolov3.cfg
# !wget https://pjreddie.com/media/files/yolov3.weights
# !wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O coco.names

# Importa bibliotecas para manipulação de imagens e visualização
import cv2  # OpenCV para processamento de imagens
import numpy as np  # Numpy para operações numéricas
import requests  # Para fazer requisições HTTP
import matplotlib.pyplot as plt  # Para visualização de gráficos e imagens

# Função para carregar a imagem de uma URL
def load_image_from_url(img_url):
    response = requests.get(img_url)  # Faz uma requisição GET para a URL da imagem
    # Decodifica a imagem recebida e a retorna
    img = cv2.imdecode(np.asarray(bytearray(response.content), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

# Função para processar a imagem e fazer a detecção de objetos
def detect_objects(img):
    # Carrega o modelo YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Carrega os pesos e a configuração do modelo
    layer_names = net.getLayerNames()  # Obtém os nomes das camadas
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Obtém as camadas de saída

    # Processa a imagem
    height, width, _ = img.shape  # Obtém as dimensões da imagem
    # Prepara a imagem para o modelo (redimensionamento e normalização)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)  # Define a entrada para a rede
    outputs = net.forward(output_layers)  # Realiza a detecção

    # Lista para guardar os resultados
    boxes = []  # Caixas delimitadoras
    confidences = []  # Confiança das detecções
    class_ids = []  # IDs das classes detectadas

    # Extraindo informações dos outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # O array de classes
            class_id = np.argmax(scores)  # Índice da classe com maior confiança
            confidence = scores[class_id]  # Confiança da predição

            # Filtra detecções com confiança maior que 0.5
            if confidence > 0.5:
                center_x = int(detection[0] * width)  # Coordenada X do centro
                center_y = int(detection[1] * height)  # Coordenada Y do centro
                w = int(detection[2] * width)  # Largura da caixa
                h = int(detection[3] * height)  # Altura da caixa

                # Calcula coordenadas do retângulo
                x = int(center_x - w / 2)  # Coordenada X do canto superior esquerdo
                y = int(center_y - h / 2)  # Coordenada Y do canto superior esquerdo

                boxes.append([x, y, w, h])  # Adiciona a caixa à lista
                confidences.append(float(confidence))  # Adiciona a confiança à lista
                class_ids.append(class_id)  # Adiciona o ID da classe à lista

    # Aplica Non-Maximum Suppression para eliminar caixas sobrepostas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Filtra as caixas
    return boxes, confidences, class_ids, indexes  # Retorna as informações das detecções

# Função para desenhar as detecções na imagem
def draw_labels(boxes, confidences, class_ids, indexes, img):
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]  # Carrega os nomes das classes

    for i in range(len(boxes)):
        if i in indexes:  # Verifica se a detecção está na lista de índices
            x, y, w, h = boxes[i]  # Obtém as coordenadas da caixa
            label = str(classes[class_ids[i]])  # Obtém o nome da classe
            confidence = confidences[i]  # Obtém a confiança da detecção
            color = (255, 0, 0)  # Cor azul para o retângulo

            # Desenha o retângulo na imagem
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Ajusta a posição do texto para garantir que ele esteja dentro da imagem
            text_x = max(x, 0)  # Evita que a posição fique negativa
            text_y = max(y - 10, 0)  # Evita que a posição fique acima da imagem

            # Adiciona o texto da etiqueta com a confiança
            cv2.putText(img, f"{label}: {confidence:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# URL da imagem com múltiplos animais
img_url = 'https://st2.depositphotos.com/1499498/7037/i/600/depositphotos_70370967-stock-photo-cute-kittens-and-puppy.jpg'

# Carrega a imagem da URL
img = load_image_from_url(img_url)

# Chama a função de detecção de objetos
boxes, confidences, class_ids, indexes = detect_objects(img)

# Desenha as detecções na imagem
draw_labels(boxes, confidences, class_ids, indexes, img)

# Exibe a imagem com as detecções
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Converte a imagem de BGR para RGB para exibição
plt.axis('off')  # Remove os eixos
plt.show()  # Mostra a imagem
