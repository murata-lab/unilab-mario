# rec.py データセット作成モジュール

'''
# 録音前の準備
44行目のfile_pathにパスを設定してください。（'\signal'より前の部分の変更をしてください）
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
import csv
import os

recording_number = 38

def callback(indata, frames, time, status):
    def savefunc(data):
        global count
        global name_count
        fig_1, (ax_1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]})
        
        # 波形プロット
        ax_1.plot(data)
        ax_1.set_ylim([-1.0, 1.0])
        ax_1.set_xlim([0, length])
        ax_1.yaxis.grid(True)
        
        # ファイル番号の準備
        file_num = 4 * recording_number + name_count

        # CSVファイルに音源データの値を保存
        file_path = r'C:\Python\unilab-mario\data\signal' + str(file_num) + '.csv'
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
