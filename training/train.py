import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.utils import shuffle
import os
import csv
from scipy import signal
import librosa

# データセットのファイル数
num_files = 131

# データセットの分割
div = 100

# 特徴量抽出器の選択（mode = 0...stft, 1...mfcc）
mode = 0

# 読み込むデータのディレクトリを指定
directory = 'data'

# 3次元テンソルを作成するための空の配列
data_tensor = []

# stftによる特徴量抽出
if mode == 0:
    model_name ='model_stft.h5'
    width = 513   # 行数
    height = 42 # 列数

    # 全てのCSVファイルに対して処理を行う
    for i in range(num_files):
        filename = f"signal{i}.csv"
        file_path = os.path.join(directory, filename)

        # CSVファイルからデータを読み込む
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.extend(row)

        # データをNumPy配列に変換
        data = np.array(data, dtype=np.float32)
        
        # スペクトログラムを計算
        f, t, Sxx = signal.spectrogram(data, fs=44100, nperseg=1024, noverlap=512)

        # データを3次元テンソルに追加
        data_tensor.append(Sxx)

# mfccによる特徴量抽出
else:
    model_name ='model_mfcc.h5'
    width = 128   # 行数
    height = 44 # 列数

    # 全てのCSVファイルに対して処理を行う
    for i in range(num_files):
        filename = f"signal{i}.csv"
        file_path = os.path.join(directory, filename)

        # CSVファイルからデータを読み込む
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.extend(row)

        # データをNumPy配列に変換
        data = np.array(data, dtype=np.float32)
        
        # メルスペクトログラムを計算
        Sxx = librosa.feature.melspectrogram(y=data, sr=44100, n_fft=1024, hop_length=512)
        Sxx_db = librosa.power_to_db(Sxx, ref=np.max)

        # データを3次元テンソルに追加
        data_tensor.append(Sxx_db)

# 作成された3次元テンソルの形状を確認
X = np.array(data_tensor)
print(X.shape)

# 最大値と最小値の取得
X_min_value = min(min(min(row) for row in matrix) for matrix in X)
X_max_value = max(max(max(row) for row in matrix) for matrix in X)

# 正規化
X_normalized = [[[((X_element - X_min_value) / (X_max_value - X_min_value)) for X_element in row] for row in matrix] for matrix in X]

# ラベルを格納するための空の配列
y = np.zeros(num_files)

# ラベルの準備
for i in range(num_files):
        y[i] = i % 4

# データのシャッフル
X_shuffled, y_shuffled = shuffle(X_normalized, y)

# 入力データの準備
input_data = []
for spectrogram in X_shuffled:
    input_data.append(np.array(spectrogram).reshape(width, height, 1))
input_data = np.array(input_data)

# ラベルの準備
labels = np.array(y_shuffled)

# データセットの分割（例えば、トレーニングセットとテストセットの分割）
X_train = input_data[:div]
y_train = labels[:div]
X_test = input_data[div:]
y_test = labels[div:]

# ラベルをone-hotエンコーディングする
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

# モデルの構築（spectrogram_width: スペクトログラムの幅）
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルの学習
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# モデルの保存
model.save(model_name)
