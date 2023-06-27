import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy import signal

def callback(indata, frames, time, status):
    def savefunc(data):
        global count
        global name_count
        fig_1, (ax_1, ax_2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
        
        # プロットデータのスペクトログラムを計算
        f, t, Sxx = signal.spectrogram(data, fs=44100, nperseg=1024, noverlap=512)

        # スペクトログラムをログスケールに変換
        Sxx_db = 10 * np.log10(Sxx)
        arr = np.array(Sxx_db)
        print(arr.shape)
        
        # 波形プロット
        ax_1.plot(data)
        ax_1.set_ylim([-1.0, 1.0])
        ax_1.set_xlim([0, length])
        ax_1.yaxis.grid(True)
        
        # スペクトログラムプロット
        ax_2.pcolormesh(t, f, Sxx_db, shading='auto')
        ax_2.set_ylim([0, 10000])
        ax_2.set_ylabel('Frequency [Hz]')
        
        fig_1.savefig("hoge" + str(name_count))
        name_count += 1
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