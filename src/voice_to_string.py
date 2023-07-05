import librosa
import numpy as np
from scipy import signal
import tensorflow as tf

# モデルの選択（stft...0, mel...1）
mode = 1

# ロードする学習済みモデルのパス
model_path = 'model_mfcc.h5'

# モデルのロード
model = tf.keras.models.load_model(model_path)

def voice_to_string(data):
    if mode == 0:
        # スペクトログラムの計算
        f, t, Sxx = signal.spectrogram(data, fs=44100, nperseg=1024, noverlap=512)
        Sxx_db = Sxx

    else:
        # メルスペクトログラムの計算
        Sxx = librosa.feature.melspectrogram(y=data, sr=44100, n_fft=1024, hop_length=512)
        Sxx_db = librosa.power_to_db(Sxx, ref=np.max)

    # リスト内の最小値と最大値を求める
    min_value = min(min(row) for row in Sxx_db)
    max_value = max(max(row) for row in Sxx_db)

    # 正規化を行う
    normalized_Sxx = [[(value - min_value) / (max_value - min_value) for value in row] for row in Sxx_db]

    # 2次元リストをNumPy配列に変換
    normalized_Sxx = np.array(normalized_Sxx)

    print(normalized_Sxx.shape)

    
    #予測の実行
    if mode == 0:
        predictions = model.predict(normalized_Sxx.reshape(1, 513, 42, 1))
    else:
        predictions = model.predict(normalized_Sxx.reshape(1, 128, 44, 1))
    print(predictions)
    max_index = np.argmax(predictions)
    print(max_index)
    return str(int(max_index))
    