# Desafio — Treinamento de Redes Neurais com Transfer Learning

**Bootcamp DIO — Machine Learning Practitioner**

## Resumo
Este repositório reúne o desafio de Transfer Learning realizado durante o Bootcamp da DIO. O objetivo foi treinar um classificador de imagens utilizando um modelo pré‑treinado (Transfer Learning) no framework **TensorFlow** — todo o desenvolvimento foi feito em **Python** usando o **Google Colab**.

## Ambiente e links úteis
- **Notebook (Google Colab):** https://colab.research.google.com/drive/1_fe1D8SKQ5ePmKwrN00RpBq4s7rcpau5?usp=sharing
- **Referência principal (tutorial TensorFlow):** https://www.tensorflow.org/tutorials/images/transfer_learning?hl=pt-br
- Linguagem: **Python**
- Framework: **TensorFlow / Keras**

---

## Objetivo do projeto
Treinar um modelo de classificação de imagens aplicando Transfer Learning: aproveitar pesos pré‑treinados, adaptar a cabeça (head) do modelo para o problema específico, treinar apenas as camadas finais.

---
### Passo 1: Importar bibliotecas
> Inicialmente importei as bibliotecas:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os 
import zipfile # extração de arquivo zip
from pathlib import Path # manipulação dos caminhos de arquivos
import urllib.request # para requisição de download do arquivo
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # necessário para verificar e excluir arquivos corrompidos do dataset
```

### Passo 2: Baixar e Extrair dataset
> Criei duas funções:
> Uma para realizar o download do dataset Cats and Dogs indicado no material de apoio
```python
# Função para baixar o arquivo zip
def download_dataset(url: str, zip_name: str) -> None:
    print(f"Baixando o arquivo {zip_name}")
    try:
      urllib.request.urlretrieve(url, zip_name)
      print("Download concluído")
    except Exception as e:
      print(f"Erro ao baixar o arquivo: {e}")
```
> A Outra função para descompactar o arquivo primeiramente verifica se o arquivo já foi baixado, e se não foi, ele chama a função para realizar o download. Em seguida verifica se a pasta `PetsImages` existe na raiz e se não existir descompacta o arquivo `.zip`
```python
# Função descompactar o arquivo zip
def extract_zip_file(url: str, zip_name: str) -> None:
  if not os.path.exists(zip_name): # verifica se o arquivo não existe na pasta raiz
   download_dataset(url, zip_name) # chama a função para baixar o arquivo
  # Tentará extrair o arquivo zip
  try:
    if not os.path.exists('PetImages'): # Verifica se pasta PetImages não existe
      print(f"Descompactando o arquivo {zip_name}")
      with zipfile.ZipFile(zip_name, 'r') as zip_file:
        zip_file.extractall(".") # extrai tudo na pasta raiz
      print("Descompactação concluída")
  except Exception as e:
    print(f"Erro ao extrair o arquivo: {e}")
```
> Quando já havia terminado o código e fiz o primeiro treino do modelo, ocorreu um problema, e verificando no Gemini do Colab, ele indicou um possível problema de imagens corrompidas, então criei uma função para remover imagens corrompidas do dataset
```python
# remover imagens corrompidas do dataset
def remove_corrupted_images(folder):
    print("Verificando imagens corrompidas...")
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path)
                    img.verify() # verificando a imagem
                except (IOError, SyntaxError):
                    print(f"Removendo imagem corrompida: {img_path}")
                    os.remove(img_path)
```
> Finalmente executando o processo de donwload, descompactação e remoção de imagens corrompidas:
```python
url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
zip_name = url.split('/')[-1] # Atritui apenas o nome do arquivo zip, a partir da url

extract_zip_file(url, zip_name) # Executa a função para extrair o arquivo zip
remove_corrupted_images('PetImages') # Executa a função para remover imagens corrompidas
```
```bash
SAÍDA NO CONSOLE:
Baixando o arquivo kagglecatsanddogs_5340.zip
Download concluído
Descompactando o arquivo kagglecatsanddogs_5340.zip
Descompactação concluída
Verificando imagens corrompidas...
Removendo imagem corrompida: PetImages/Cat/666.jpg
Removendo imagem corrompida: PetImages/Cat/Thumbs.db
Removendo imagem corrompida: PetImages/Dog/11702.jpg
Removendo imagem corrompida: PetImages/Dog/Thumbs.db
```

### Passo 3: Verificar a estrutura dos dados
> Não era necessário mas achei interessante verificar a estrutura do dataset.
> Observe que inicialmente o dataset continha 12500 imagens de cada classe totalizando 25000 imagens com a exclusão das imagens corrompidas ficaram 12499 de cada classe
```python
if os.path.exists('PetImages'):
  print("Estrutura do dataset: ")
  for root, dirs, files in os.walk('PetImages'):
    print(f"{root}: {len(files)}")
```
```
Estrutura do dataset: 
PetImages: 0
PetImages/Cat: 12499
PetImages/Dog: 12499
```

### Passo 4: Configuração dos parametros e pré-processamento
> Criei as constantes necessárias para alguns parâmetros dos datasets que serão usadas ao longo do código
> Incialmente tentei setar o parâmetro `IMG_SIZE` em 224, e o `BATCH_SIZE` em 32, mas o Colab dava sobre carga no uso da RAM, então reduzi esses parâmetros para melhorar na etapa de treinamento  
```python
# CONSTANTES PARA OS PARAMETROS DAS FUNÇÕES DE TREINO E VALIDAÇÃO
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

# FUNÇÃO DO TENSORFLOW QUE ESPECIFICA OS PARAMETROS DE TREINO E SUAS CLASSES
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'PetImages', # Especifica o caminho da pasta raiz onde estão as classes Cat e Dog
    validation_split=0.2, # Determina que 20% das imagens serão usadas na validação ()
    subset='training', # Determinta que o conjunto será de treino
    seed=123, # Garante a divisão  treino/validação seja igual
    image_size=(IMG_SIZE, IMG_SIZE), # Redimenciona todas as imagens para o padrão da rede neural
    batch_size=BATCH_SIZE, # determina o tamanho do lote de imagens que serão passadas ao modelo por vez
)

# FUNÇÃO DO TENSORFLOW QUE ESPECIFICA OS PARAMETROS DE VALIDAÇÃO E SUAS CLASSES
val_dataset = tf.keras.utils.image_dataset_from_directory(
    'PetImages',
    validation_split=0.2,
    subset='validation',  # Determinta que o conjunto será de validação
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)
```
```
Found 24998 files belonging to 2 classes. 
Using 19999 files for training.
Found 24998 files belonging to 2 classes.
Using 4999 files for validation.
```
> Usei a biblioteca Matplotlib para visualizar algumas imagens
```python
# VISUALIZAR ALGUMAS IMAGENS
print("Visualizando algumas imagens do dataset:")
plt.figure(figsize=(5, 5))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
```
<img width="467" height="490" alt="image" src="https://github.com/user-attachments/assets/07714f77-b795-4347-9772-98ef50969e5e" />




