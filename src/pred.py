import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
import librosa
import socket

# モデルの選択（stft...0, mel...1）
mode = 1

# ロードする学習済みモデルのパス
model_path = 'model_mfcc.h5'

# モデルのロード
model = tf.keras.models.load_model(model_path)

# 送信先の設定
target_ip = "127.0.0.1"  # IPアドレス
target_port = 12345      # ポート番号

def callback(indata, frames, time, status):
    def savefunc(data):
        global count

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

        #予測の実行
        if mode == 0:
            predictions = model.predict(normalized_Sxx.reshape(1, 513, 42, 1))
            print(predictions)
            max_index = np.argmax(predictions)
            print(max_index)

        else:
            predictions = model.predict(normalized_Sxx.reshape(1, 128, 44, 1))
            print(predictions)
            max_index = np.argmax(predictions)
            print(max_index)

        # ソケット通信でデータを送信
        send_data = str(max_index)  # 送信するデータを文字列に変換
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((target_ip, target_port))
            sock.sendall(send_data.encode())  # データをバイト列にエンコードして送信

        count = 0

    # indata.shape=(n_samples, n_channels)
    global plotdata
    global count
    flag = False
    data = indata[::downsample, 0]
    shift = len(data)
    if count == 0:
        for val in data:
            if val > 0.05:
                print("utter")
                count += 1
                flag = True
                break
    plotdata = np.roll(plotdata, -shift, axis=0)
    plotdata[-shift:] = data
    if not flag:
        if count != 0:
            count += 1
        if count > (35 * recording_time):
            sd_flag = 1
            savefunc(plotdata)          

def update_plot(frame):
    """This is called by matplotlib for each plot update.
    """
    global plotdata
    line.set_ydata(plotdata)
    return line,

# 録音の設定
recording_time = 0.50   #録音時間
downsample = 1          #サンプルの圧縮サイズ
sd_flag = 0
length = int(1000 * 44100 / (1000 * downsample) * recording_time) 
plotdata = np.zeros((length))
count = 0
fig, ax = plt.subplots()
line, = ax.plot(plotdata)
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([0, length])
ax.yaxis.grid(True)
fig.tight_layout()

stream = sd.InputStream(
    channels=1,
    dtype='float32',
    callback=callback
)
ani = FuncAnimation(fig, update_plot, interval=30, blit=True)
with stream:
    plt.show()