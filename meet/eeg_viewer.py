'''
Simple interactive EEG viewer

Submodule of the Modular EEg Toolkit - MEET for Python.

Author:
-------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de
'''

from . import _np
from . import _signal

import scipy.constants as _const
from matplotlib import pyplot as _plt
from matplotlib.widgets import RectangleSelector
_plt.ioff()
from matplotlib.ticker import FuncFormatter as _FuncFormatter
from matplotlib.widgets import Cursor as _Cursor

def _seconds_formatter(t, pos):
    '''Assume that t is in seconds'''
    return '%02d:%02d' % (t // 60, t % 60)

_x_axis_formatter = _FuncFormatter(_seconds_formatter)
_line_props = [{'color': 'k'}, {'color': 'r'}]

#detect screen resolution
import tkinter as _Tk
_root = _Tk.Tk()
_res =  (_root.winfo_screenwidth() / float(_root.winfo_screenmmwidth())
        * 1000 * _const.inch)
_root.destroy()

class plotEEG:
    def __init__(self, signals, ylabels, t, t_res=30, title = False):
        """Function to Plot EEG-Signals in of several channels
        Def: PlotEEG(signals, ylabels, t, t_res, title = False)

        Interaction:
        ------------
        PageUp -> Go backward a big step
        PageDown -> Go forward a big step
        Up -> Go backward a small step
        Down -> go forward a small step
        i -> Zoom in (show smaller temporal window)
        o -> Zoom out (show larger temporal window)
        + -> increase gain
        - -> decrease gain
        Pos1 -> Go to start
        End -> Go to end
        LeftMouseClick -> saves the x coordinate in self.clicks
        Selection -> Select an area
        s -> store the x-coordinates of the selection in self.select

        Input:
        ------
            -signals: 2d array of input data
            -ylabels: list of channel names in same order as in
                      \'signals\' (unit is presumed to be mikro V)
            -t: array of time values in s
            -t_res: mm per second (standard: 30 mm / s)
            -title: title string
        Output:
        -------
            -EEG_viewer class: plot with EEG.show()
             clicks are saved in self.clicks
        """
        signals = _np.array(signals)
        if len(t) == signals.shape[1]:
            self.signals = signals
            self.p, self.n = signals.shape
        elif len(t) == signals.shape[0]:
            self.signals = signals.T
            self.n, self.p = signals.shape
        self.ylabels = ylabels
        self.t_diff = (t[-1] - t[0]) / float( len(t) -1)
        if not _np.all(_np.abs(_np.diff(_np.diff(t))) /
                self.t_diff < 1E5):
            raise ValueError('t must be equally spaced')
        else:
            self.t = t
        self.t_res = t_res
        self.title = title
        self._PlotInitSignals()
        self.fig.canvas.draw()
        self.clicks = []
        self.select = []
        # initialize the selector
        self.RS = RectangleSelector(ax=self.ax, onselect=self._on_select,
                drawtype='box', useblit=True, interactive=True, button=1)
        return

    def _PlotInitSignals(self):
        self.fig = _plt.figure(dpi=_res)
        self.ax = self.fig.add_axes([0.07, 0.1, 0.93, 0.85])
        w, h = self.fig.get_size_inches() 
        self._ax_width_mm = w * 0.93 * _const.inch * 1000
        #time that fits into ax_width:
        self.t_show = _np.max([_np.min([self._ax_width_mm / self.t_res,
            self.t.ptp()]), 2*self.t_diff])
        self.t0 = 0
        self.t1 = int(self.t0 + self.t_show / self.t_diff)
        #get data to plot
        d_plot =  _signal.detrend(self.signals[:,self.t0:self.t1] -
                _np.median(self.signals[:,self.t0:self.t1],
                    -1)[:,_np.newaxis], axis=-1, type='constant')
        #initially seperate the data by the MAD
        self._offset = _np.median(_np.abs(d_plot), None)
        #get the ylims
        self._ymin = 0
        self._ymax = self.p * 4 * self._offset
        self._ylocs = _np.arange(2, 4*self.p + 2, 4) * self._offset
        self.lines = [self.ax.plot(self.t[self.t0:self.t1], d_plot[i] +
            self._ylocs[i], lw=1.0, ls='-', **_line_props[i%2])[0]
            for i in range(self.p)]
        self.ax.set_ylim([self._ymin, self._ymax])
        self.ax.set_xlim([self.t[self.t0], self.t[self.t1]])
        self.ax.tick_params(axis='both', bottom=True, top=True,
                left=False, right=False, labelleft=False,
                labelright=False, labeltop=True, labelbottom=True)
        self.ax.grid(axis='x', which='major', linestyle='-', color='b',
                lw=1)
        self.ax.set_xlabel('time in s')
        self.ax.xaxis.set_major_formatter(_x_axis_formatter)
        if self.title: self.ax.set_title(self.title)
        #add labels
        [self.fig.text(x = 0.01, y = 0.1 + (i+0.5)* 0.85/(self.p), s=s,
            ha='left', va='center', **_line_props[i%2])
            for i, s in enumerate(self.ylabels)]
        ##add cursor:
        #self._cursor = _Cursor(self.ax, useblit=False, color='r',
        #        linewidth=2, linestyle='-', horizOn=False)
        #add voltage reference bar
        self._bar_ax = self.fig.add_axes([0.01, 0.02, 0.05, 0.1],
                frame_on=False)
        self._bar_ax.plot([0,0], [0,1], 'b-', lw=3) 
        self._bar_ax.tick_params(axis='both', bottom=False, top=False,
                left=False, right=False, labelleft=False,
                labelright=False, labeltop=False, labelbottom=False)
        self._bar_ax.set_xlim([0,1])
        self._bar_ax.set_ylim([0,1])
        self._bar_scale = self._bar_ax.text(s=r'$%.1f \mu V$' % ((
            self._ymax - self._ymin) / 8.5), x=0.1, y=0.5, va='center',
            ha='left')
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        return

    def _on_resize(self, event):
        '''
        On resize change the x-scale to keep same temporal resolution,
        keep the y-scale unchanged
        '''
        w = event.width / float(self.fig.dpi)
        self._ax_width_mm = w * 0.93 * _const.inch * 1000
        self.t_show = _np.max([_np.min([self._ax_width_mm / self.t_res,
            self.t.ptp()]), 2*self.t_diff])
        self.t0 = _np.min([self.t0, int(self.n -
            self.t_show/self.t_diff)])
        self.t1 = int(self.t0 + self.t_show / self.t_diff)
        #get data to plot
        d_plot =  _signal.detrend(self.signals[:,self.t0:self.t1] -
                _np.median(self.signals[:,self.t0:self.t1],
                    -1)[:,_np.newaxis], axis=-1, type='constant')
        [l.set_data(_np.vstack([self.t[self.t0:self.t1], d_plot[i] + 
            self._ylocs[i]])) for i, l in enumerate(self.lines)]
        self.ax.set_xlim([self.t[self.t0], self.t[self.t1]])
        self.fig.canvas.draw()
        return

    def _on_key(self, event):
        '''
        If buttons are clicked, to the corresponding action
        '''
        if event.key == '+':
            #increase gain - make offset smaller by 10%
            self.change_gain(new_offset = self._offset*0.9)
        if event.key == '-':
            #decrease gain - make offset larger by 10%
            self.change_gain(new_offset = self._offset*1.1)
        if event.key == 'pageup':
            #go backward
            self.change_t(new_t0 = _np.max([0, int(self.t0 -
                0.9*self.t_show/self.t_diff)]),
                new_t_show = self.t_show)
        if event.key == 'pagedown':
            #go forward
            self.change_t(new_t0 = _np.min([int(self.n -
                self.t_show/self.t_diff), int(self.t0 +
                    0.9*self.t_show/self.t_diff)]),
                new_t_show = self.t_show)
        if event.key == 'left':
            #go backward small
            self.change_t(new_t0 = _np.max([0, int(self.t0 -
                0.1*self.t_show/self.t_diff)]),
                new_t_show = self.t_show)
        if event.key == 'right':
            #go forward small
            self.change_t(new_t0 = _np.min([int(self.n -
                self.t_show/self.t_diff), int(self.t0 +
                    0.1*self.t_show/self.t_diff)]),
                new_t_show = self.t_show)
        if event.key == 'i':
            #zoom in
            new_t_show = _np.max([_np.min([0.8*self.t_show,
                self.t.ptp()]), 2*self.t_diff])
            new_t0 =  _np.max([_np.min([int(self.n -
                new_t_show/self.t_diff), self.t0]), 0])
            self.change_t(new_t0 = new_t0, new_t_show = new_t_show)
        if event.key == 'o':
            #zoom out
            new_t_show = _np.max([_np.min([1.2*self.t_show,
                self.t.ptp()]), 2*self.t_diff])
            new_t0 =  _np.max([_np.min([int(self.n -
                new_t_show/self.t_diff), self.t0]), 0])
            self.change_t(new_t0 = new_t0, new_t_show = new_t_show)
        if event.key == 'end':
            #go to end
            self.change_t(new_t0 = int(self.n -
                self.t_show/self.t_diff), new_t_show = self.t_show)
        if event.key == 'home':
            #go to start
            self.change_t(new_t0 = 0, new_t_show = self.t_show)
        if event.key == 'r':
            print('Saving rectangle')
            self.select.append(self.RS.corners[0][:2])
        return

    def change_gain(self, new_offset):
        self._offset = new_offset
        #get the ylims
        self._ymin = 0
        self._ymax = self.p * 4 * self._offset
        self._ylocs = _np.arange(2, 4*self.p + 2, 4) * self._offset
        d_plot =  _signal.detrend(self.signals[:,self.t0:self.t1] -
                _np.median(self.signals[:,self.t0:self.t1],
                    -1)[:,_np.newaxis], axis=-1, type='constant')
        [l.set_data(_np.vstack([self.t[self.t0 : self.t1], d_plot[i] +
            self._ylocs[i]])) for i, l in enumerate(self.lines)]
        self.ax.set_ylim([self._ymin, self._ymax])
        self._bar_scale.set_text(r'$%.1f \mu V$' % ((self._ymax -
            self._ymin) / 8.5))
        self.fig.canvas.draw()
        return

    def change_t(self, new_t0, new_t_show):
        self.t0 = new_t0
        self.t_show = new_t_show
        w, h = self.fig.get_size_inches() 
        self._ax_width_mm = w * 0.93 * _const.inch * 1000
        self.t_res =  self._ax_width_mm / self.t_show
        #change t_res accordingly
        self.t1 = int(self.t0 + self.t_show / self.t_diff)
        #get data to plot
        d_plot =  _signal.detrend(self.signals[:,self.t0:self.t1] -
                _np.median(self.signals[:,self.t0:self.t1],
                    -1)[:,_np.newaxis], axis=-1, type='constant')
        [l.set_data(_np.vstack([self.t[self.t0:self.t1], d_plot[i] +
            self._ylocs[i]])) for i, l in enumerate(self.lines)]
        #self.fig.canvas.draw()
        self.ax.set_xlim([self.t[self.t0], self.t[self.t1]])
        self.fig.canvas.draw()
        return

    def _on_select(self, eclick, erelease):
        print('Click at x: %f, y: %f' % (eclick.xdata, eclick.ydata))
        self.clicks.append(_np.argmin(_np.abs(self.t-eclick.xdata)))
        pass

    def show(self):
        _plt.show()
        return
