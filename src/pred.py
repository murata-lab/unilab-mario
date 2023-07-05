import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


import voice_to_string
import string_to_command


# config
voice_config = {"threshold": 0.08, "skip": 35, "y_size": 0.5}
command_config = {"step": 20}
def callback(indata, frames, time, status):
    # indata.shape=(n_samples, n_channels)
    global plotdata
    global count
    global command
    flag = False
    data = indata[::downsample, 0]
    shift = len(data)
    if count == 0:
        for val in data:
            if val > voice_config["threshold"]:
                print("utter")
                count += 1
                flag = True
                break
    plotdata = np.roll(plotdata, -shift, axis=0)
    plotdata[-shift:] = data
    command = string_to_command.command_admin(step, command, command_config["step"])
    if not flag:
        if count != 0:
            count += 1
        if count > (35 * recording_time):
            command = voice_to_string.voice_to_string(plotdata)
            count = 0

def update_plot(frame):
    """This is called by matplotlib for each plot update.
    """
    global plotdata
    line.set_ydata(plotdata)
    return line,

# 録音の設定
command = "0"
step = [0,0]
recording_time = 0.50   #録音時間
downsample = 1          #サンプルの圧縮サイズ
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
