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
---
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
---
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
---
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

---
### Passo 5: Otimização de performance e normalização
> O tutorial do tensorflow recomenda essa otimização, e pesquisando mais sobre essa otimização, descobri outros métodos de otimização:
> O método `cache()` armazena os dados em memória (RAM) ou em disco (caso não caiba na RAM) e evita que as imagens sejam lidas e processadas repetidamente do disco a cada época, acelerando o processo de treinamento após a primeira época.
> O método `shuffle(1000)` mmbaralha a ordem dos exemplos a cada época. O número 1000 é o buffer size, ou seja, a quantidade de elementos carregados e embaralhados na memória de cada vez, evitando que a rede aprenda padrões artificiais da ordem dos dados, melhorando a generalização.
> O método `prefetch(buffer_size=AUTOTUNE)` permite que o carregamento dos próximos lotes de dados aconteça em paralelo ao treinamento do modelo. O `AUTOTUNE` deixa o TensorFlow escolher automaticamente o melhor número de lotes para manter em pré-carregamento, reduzindo tempo ocioso da GPU/TPU melhorando a eficiência.
```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
```
---
### Passo 6: Carregar o modelo Base MobileNetV2
> Ao carregar devemos determinar `include_top = False` não incluindo a classificação final e os pesos `weights` da ImageNet.
> Em seguida congelar as camadas do modelo base
```python
# Carregar MobileNetV2 pré-treinado na imagenet
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False, # não inclui a camada de classificação final
                                               weights='imagenet') # usar os pesos pre-treinados da ImageNet

# Congela as camadas do modelo base
base_model.trainable = False

print(f"modelo base carregado: {base_model.name}")
print(f"modelo base possui {len(base_model.layers)} camadas")
```
```
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
modelo base carregado: mobilenetv2_1.00_160
modelo base possui 154 camadas
```
---
### Passo 7: Construir o modelo completo
```python
# Cria camadas de pré-processamento
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Criar o modelo completo
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Pré-Processamento para o MobileNetV2
x = preprocess_input(inputs)

# Modelo Base
x = base_model(x, training=False)

# Camadas de classificação personalizadas
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Cria o modelo
model = tf.keras.Model(inputs, outputs)

# Compilação do Modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Gera um resumo do modelo
model.summary()
```
<img width="783" height="534" alt="image" src="https://github.com/user-attachments/assets/98082740-80ec-4906-909d-d536ec0d5800" />

---
### Passo 8: Treinar o Modelo
> Mesmo excluindo as imagens corrompidas ainda estava ocorrendo alguns problemas de imagens que a lib `PIL` não estava conseguindo excluir, então pesquisando achei essa solução experimental disponível para a versão 2.19 do Tensorflow do Colab que ignora erros `tf.data.experimental.ignore_errors()` e apliquei tanto no dataset de treino quanto no de validação e finalmente consegui realizar o treinamento.
```python
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())

val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())

# Treinar Modelo
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset,
                    verbose=1)

print("Treinamento concluído")
```
```
Epoch 1/10
   1238/Unknown 62s 20ms/step - accuracy: 0.8697 - loss: 0.3090/usr/local/lib/python3.12/dist-packages/keras/src/trainers/epoch_iterator.py:160: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 83s 37ms/step - accuracy: 0.8700 - loss: 0.3084 - val_accuracy: 0.9734 - val_loss: 0.0879
Epoch 2/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 24s 17ms/step - accuracy: 0.9719 - loss: 0.0851 - val_accuracy: 0.9772 - val_loss: 0.0651
Epoch 3/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 11ms/step - accuracy: 0.9802 - loss: 0.0624 - val_accuracy: 0.9794 - val_loss: 0.0572
Epoch 4/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 12ms/step - accuracy: 0.9825 - loss: 0.0522 - val_accuracy: 0.9810 - val_loss: 0.0533
Epoch 5/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 12ms/step - accuracy: 0.9823 - loss: 0.0536 - val_accuracy: 0.9816 - val_loss: 0.0518
Epoch 6/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 12ms/step - accuracy: 0.9828 - loss: 0.0498 - val_accuracy: 0.9826 - val_loss: 0.0498
Epoch 7/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 18s 15ms/step - accuracy: 0.9847 - loss: 0.0470 - val_accuracy: 0.9828 - val_loss: 0.0492
Epoch 8/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 12ms/step - accuracy: 0.9846 - loss: 0.0448 - val_accuracy: 0.9824 - val_loss: 0.0486
Epoch 9/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 14s 12ms/step - accuracy: 0.9846 - loss: 0.0443 - val_accuracy: 0.9822 - val_loss: 0.0482
Epoch 10/10
1243/1243 ━━━━━━━━━━━━━━━━━━━━ 15s 12ms/step - accuracy: 0.9844 - loss: 0.0422 - val_accuracy: 0.9820 - val_loss: 0.0475
Treinamento concluído
```
---
### Análise da Curva de Aprendizado
> Essa análise eu copie do tutorial do tensorflow
```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```
<img width="700" height="701" alt="image" src="https://github.com/user-attachments/assets/496a1183-5583-49b0-8722-43861162f896" />

---
### Considerações Finais
Ao estudar sobre `transfer learning`, percebi a possibilidade de realizar o fine-tuning, que consiste em descongelar todas as camadas do modelo base. No entanto, observei que trata-se de um processo mais pesado, o que provavelmente inviabiliza sua execução no Colab Free.

A realização deste desafio foi uma experiência valiosa, que me permitiu compreender melhor as etapas do `transfer learning` e os principais desafios que podem surgir durante o treinamento.

