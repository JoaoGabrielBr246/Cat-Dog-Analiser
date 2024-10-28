# -*- coding: utf-8 -*-
"""cats-vs-dogs.ipynb

Este arquivo foi gerado automaticamente pelo Google Colab.

Localização original do arquivo:
    https://colab.research.google.com/drive/1ADJyCjz66r0U6Hw-CyZc3yIf1qFxNgfw
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

diretorio_base = '/content/drive/MyDrive/Colab Notebooks/cats-dogs/PetImages'

gerador_treinamento = ImageDataGenerator(rescale=1.0/255.0,  
                                         rotation_range=20,  
                                         width_shift_range=0.2,  
                                         height_shift_range=0.2,  
                                         shear_range=0.2,  
                                         zoom_range=0.2,  
                                         horizontal_flip=True,  
                                         fill_mode='nearest')  

gerador_imagens = gerador_treinamento.flow_from_directory(
    diretorio_base,
    target_size=(224, 224),  
    batch_size=32,  
    class_mode='binary')  

modelo_base = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
modelo_base.trainable = False  

modelo = models.Sequential([
    modelo_base,  
    layers.GlobalAveragePooling2D(),  
    layers.Dense(1, activation='sigmoid')  
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

historico = modelo.fit(gerador_imagens,
                        steps_per_epoch=gerador_imagens.samples // gerador_imagens.batch_size,  
                        epochs=10)  

modelo.save('/content/drive/MyDrive/Colab Notebooks/cats-dogs/PetImages/meu_modelo.keras')

import cv2  
import numpy as np  
import requests  
import matplotlib.pyplot as plt  

def carregar_imagem_da_url(url_imagem):
    resposta = requests.get(url_imagem)  
    imagem = cv2.imdecode(np.asarray(bytearray(resposta.content), dtype=np.uint8), cv2.IMREAD_COLOR)
    return imagem

def detectar_objetos(imagem):
    rede = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  
    nomes_camadas = rede.getLayerNames()  
    camadas_saida = [nomes_camadas[i - 1] for i in rede.getUnconnectedOutLayers()]  

    altura, largura, _ = imagem.shape  
    blob = cv2.dnn.blobFromImage(imagem, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    rede.setInput(blob)  
    saidas = rede.forward(camadas_saida)  

    caixas = []  
    confiancas = []  
    ids_classes = []  

    for saida in saidas:
        for deteccao in saida:
            pontuacoes = deteccao[5:]  
            id_classe = np.argmax(pontuacoes)  
            confianca = pontuacoes[id_classe]  

            if confianca > 0.5:
                centro_x = int(deteccao[0] * largura)  
                centro_y = int(deteccao[1] * altura)  
                w = int(deteccao[2] * largura)  
                h = int(deteccao[3] * altura)  

                x = int(centro_x - w / 2)  
                y = int(centro_y - h / 2)  

                caixas.append([x, y, w, h])  
                confiancas.append(float(confianca))  
                ids_classes.append(id_classe)  

    indexes = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.4)  
    return caixas, confiancas, ids_classes, indexes  

def desenhar_rotulos(caixas, confiancas, ids_classes, indexes, imagem):
    with open('coco.names', 'r') as f:
        classes = [linha.strip() for linha in f.readlines()]  

    for i in range(len(caixas)):
        if i in indexes:
            x, y, w, h = caixas[i]  
            rotulo = str(classes[ids_classes[i]])  
            confianca = confiancas[i]  
            cor = (255, 0, 0)  

            cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)

            texto_x = max(x, 0)  
            texto_y = max(y - 10, 0)  

            cv2.putText(imagem, f"{rotulo}: {confianca:.2f}", (texto_x, texto_y), cv2.FONT_HERSHEY_PLAIN, 1, cor, 2)

url_imagem = 'https://st2.depositphotos.com/1499498/7037/i/600/depositphotos_70370967-stock-photo-cute-kittens-and-puppy.jpg'

imagem = carregar_imagem_da_url(url_imagem)

caixas, confiancas, ids_classes, indexes = detectar_objetos(imagem)

desenhar_rotulos(caixas, confiancas, ids_classes, indexes, imagem)

plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))  
plt.axis('off')  
plt.show()  

