import os
import h5py
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from google.colab import drive

drive.mount('/content/drive')

!ls /content/drive/MyDrive/Colab\ Notebooks/IA_EI2_Rede_Neural/fma_small
!ls /content/drive/MyDrive/Colab\ Notebooks/IA_EI2_Rede_Neural/fma_metadata

fma_small_dir = "/content/drive/MyDrive/Colab Notebooks/IA_EI2_Rede_Neural/fma_small"
metadata = "/content/drive/MyDrive/Colab Notebooks/IA_EI2_Rede_Neural/fma_metadata/tracks.csv"

#Lê a metadata
tracks = pd.read_csv(metadata, index_col=0, header=[0, 1])
genre_mapping = tracks['track', 'genre_top'].to_dict()

all_tracks = [
    {
        "path": os.path.join(root, file),
        "genre": genre_mapping.get(int(file.split('.')[0]), "Unknown")
    }
    for root, _, files in os.walk(fma_small_dir)
    for file in files
    if file.endswith('.mp3')
]

#Converte em datarame
labeled_df = pd.DataFrame(all_tracks)

#Função para fazer sample equilibrada entre os gêneros
def balance_genre_sample(df, total_samples=6000):
    df = df[df['genre'] != 'Unknown']

    #Pega um gênero único
    genres = df['genre'].unique()

    #Calcula as samples por gênero
    samples_per_genre = total_samples // len(genres)
    remainder = total_samples % len(genres)

    balanced_tracks = []
    for genre in genres:
        genre_tracks = df[df['genre'] == genre]

        #Determina o número de samples por gênero
        num_samples = samples_per_genre + (1 if remainder > 0 else 0)
        remainder -= 1

        #Faz samples aleatóriamente
        sampled_tracks = genre_tracks.sample(n=min(num_samples, len(genre_tracks)))
        balanced_tracks.append(sampled_tracks)

    #Combina e embaralha
    final_df = pd.concat(balanced_tracks).sample(frac=1).reset_index(drop=True)

    return final_df

balanced_labeled_df = balance_genre_sample(labeled_df, total_samples=6000)

#Salva em CSV
balanced_labeled_df.to_csv("labeled_tracks.csv", index=False)

print(balanced_labeled_df['genre'].value_counts())

#Carrega o dataset
labeled_df = pd.read_csv('labeled_tracks.csv')

#Faz encode
label_encoder = LabelEncoder()
labeled_df['label'] = label_encoder.fit_transform(labeled_df['genre'])

#Função para extrair os espectrograma
def extract_spectrogram(file_path, sr=22050, duration=30, max_pad_len=128):
    try:
        #Carrega um arquivo de áudio
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)

        #Gera o espectrograma
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        #Converte para escala logaritmica
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        #Normaliza a escala
        spectrogram_normalized = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

        #Faz pad ou truncate para uma shape consistente
        if spectrogram_normalized.shape[1] > max_pad_len:
            spectrogram_normalized = spectrogram_normalized[:, :max_pad_len]
        else:
            padding = max_pad_len - spectrogram_normalized.shape[1]
            spectrogram_normalized = np.pad(
                spectrogram_normalized,
                ((0, 0), (0, padding)),
                mode='constant'
            )

        return spectrogram_normalized.reshape((128, 128, 1))

    except Exception as e:
        print(f"erro {file_path}: {e}")
        return None

#Inicializa o arquivo HDF5 para salvar os dados
with h5py.File('spectrograms.h5', 'w') as h5f:
    spectrograms = h5f.create_dataset(
        'X',
        (0, 128, 128, 1),  #n_mels x time_steps x channels
        maxshape=(None, 128, 128, 1),
        chunks=True
    )
    labels = h5f.create_dataset('y', (0,), maxshape=(None,), chunks=True)

    for idx, row in labeled_df.iterrows():
        spectrogram = extract_spectrogram(row['path'])
        if spectrogram is not None:
            #Redimensiona e anexa espectrogramas e labels
            spectrograms.resize(spectrograms.shape[0] + 1, axis=0)
            spectrograms[-1:] = spectrogram
            labels.resize(labels.shape[0] + 1, axis=0)
            labels[-1:] = row['label']

with h5py.File('spectrograms.h5', 'r') as hf:
    X = hf['X'][:]
    y = hf['y'][:]

#Separa os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#One-hot encode das labels para o uso da função de loss
y_train = keras.utils.to_categorical(y_train, num_classes=len(np.unique(y)))
y_test = keras.utils.to_categorical(y_test, num_classes=len(np.unique(y)))

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
    input_shape=X_train[0].shape,
    kernel_regularizer=l2(0.001)),  # Regularização L2

    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu',
    kernel_regularizer=l2(0.001)),

    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu',
    kernel_regularizer=l2(0.001)),

    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),  #Aumenta o dropout

    keras.layers.Conv2D(128, (3, 3), activation='relu',
    kernel_regularizer=l2(0.001)),

    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.37),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu',
    kernel_regularizer=l2(0.001)),  #Regulariza a dense layer

    keras.layers.Dropout(0.5),  #Dropout na dense layer

    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Faz uma early stop para evitar o overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=50,
    restore_best_weights=True,
    min_delta=0.01
)

#Reduz a learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.66,
    patience=10,
    min_lr=0.000005
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

#Avalia o modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")

#Função para prever o gênero de uma nova música
def predict_genre(file_path):
    #Extrai o espectrograma
    spectrogram = extract_spectrogram(file_path)

    if spectrogram is not None:
        #Faz reshape para a previsão
        spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)

        #Prevê
        prediction = model.predict(spectrogram)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        return predicted_label[0]

    return "erro"

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

new_song = 'Death Grips - Guillotine (It goes Yah).mp3'

predicted_genre = predict_genre(new_song)
print(f"Predicted Genre: {predicted_genre}")