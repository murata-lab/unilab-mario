# recording.py データセット作成モジュール

'''
# 録音前の準備
68行目のfile_pathにパスを設定してください。（'\signal'より前の部分の変更をしてください）
一度以下の録音方法に従って動作確認をお願いいたします。

# 録音方法
28行目の`recording_number`の値を設定（これ忘れると前の人のが上書きされるので注意してください）
ターミナルに python recording.py と打つ（実行）
画面に波形が表示されたら、「とまれ」「すすめ」「もどれ」「ジャンプ」と言ってもらいます。
１つ１つの単語は、画面の波形がある程度１本線になってから言ってもらわないと取り直しになるので注意してください。
また、言う順番もそのままでお願いします。間違えたら取り直してください。
取り直す際の`recording_number`の値の変更は必要ありません。（自動で上書きされます）
`hoge#.png`は録音ができているかの確認に使ってください。
`hoge#.png`の波形の後ろが切れてしまっている場合は撮り直しをお願いします。
'''

import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy import signal
import csv
import os
import librosa

recording_number = 31

def callback(indata, frames, time, status):
    def savefunc(data):
        global count
        global name_count
        fig_1, (ax_1, ax_2, ax_3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 1]})
        
        # プロットデータのスペクトログラムを計算
        f, t, Sxx_1 = signal.spectrogram(data, fs=44100, nperseg=1024, noverlap=512)
        Sxx_db_1 = 10 * np.log10(Sxx_1)
        arr_1 = np.array(Sxx_db_1)
        print(arr_1.shape)

        # メルスペクトログラムを計算
        Sxx_2 = librosa.feature.melspectrogram(y=data, sr=44100, n_fft=1024, hop_length=512)
        Sxx_db_2 = librosa.power_to_db(Sxx_2, ref=np.max)
        arr_2 = np.array(Sxx_db_2)
        print(arr_2.shape)
        
        # 波形プロット
        ax_1.plot(data)
        ax_1.set_ylim([-1.0, 1.0])
        ax_1.set_xlim([0, length])
        ax_1.yaxis.grid(True)
        
        # スペクトログラムプロット
        ax_2.pcolormesh(t, f, Sxx_db_1, shading='auto')
        ax_2.set_ylim([0, 10000])
        ax_2.set_ylabel('Frequency [Hz]')

        # メルスペクトログラムプロット
        librosa.display.specshow(Sxx_db_2, sr=44100, hop_length=512, x_axis='time', y_axis='mel', ax=ax_3)
        ax_3.set_ylim([0, 10000])
        ax_3.set_ylabel('Frequency [Hz]')

        # ファイル番号の準備
        file_num = 4 * recording_number + name_count

        # CSVファイルに音源データの値を保存
        file_path = r'C:\Python\Mario\data\signal' + str(file_num) + '.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        fig_1.savefig("hoge" + str(name_count))
        name_count += 1
        count = 0
        if name_count == 1:
            print("「すすめ」と言ってください")
        if name_count == 2:
            print("「もどれ」と言ってください")
        if name_count == 3:
            print("「ジャンプ」と言ってください")
        if name_count == 4:
            os._exit(0)
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

print("「とまれ」と言ってください")
recording_time = 0.50   #録音時間
downsample = 1          #サンプルの圧縮サイズ
sd_flag = 0
length = int(1000 * 44100 / (1000 * downsample) * recording_time) 
plotdata = np.zeros((length))
count = 0
name_count = 0
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