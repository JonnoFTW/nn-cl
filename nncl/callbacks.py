import matplotlib.pyplot as plt

plt.ion()
import numpy as np
from typing import List


class PlotCallback:
    def __init__(self, fields: List[str], title: str, batch_end):
        self.fig = plt.figure(title)
        self.ax = self.fig.add_subplot(111, label=title)
        self.lines = {f: self.ax.plot([], [], label=f.title())[0] for f in fields}
        self.fig.show()
        self.batch_end = batch_end

    def __call__(self, losses):
        for f, line in self.lines.items():
            ydata = losses[f]
            xdata = np.arange(0, len(ydata))
            self.ax.plot(xdata, ydata, 'r')
            # line.set_xdata(xdata)
            # line.set_ydata(ydata)
        self.fig.canvas.draw()
        plt.pause(1e-17)
