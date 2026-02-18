import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MapLayout:
    def __init__(self, figsize=(12, 10), use_inset=False, use_section=False):
        self.fig = plt.figure(figsize=figsize)
        self.use_inset = use_inset
        self.use_section = use_section

        self.ax_main = None
        self.ax_inset = None
        self.ax_legend = None
        self.ax_section = None

        self._init_axes()

    def _init_axes(self):
        if self.use_inset and self.use_section:
            gs = gridspec.GridSpec(
                3, 2,
                width_ratios=[3, 1],
                height_ratios=[3, 1.2, 1.2],
                figure=self.fig
            )
            self.ax_main = self.fig.add_subplot(gs[0:2, 0])
            self.ax_inset = self.fig.add_subplot(gs[0, 1])
            self.ax_legend = self.fig.add_subplot(gs[1, 1])
            self.ax_section = self.fig.add_subplot(gs[2, :])
            self.ax_legend.axis('off')
        elif self.use_inset:
            # 2 колонки: 3/4 карты, 1/4 панель справа
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], figure=self.fig)
            self.ax_main = self.fig.add_subplot(gs[:, 0])
            self.ax_inset = self.fig.add_subplot(gs[0, 1])
            self.ax_legend = self.fig.add_subplot(gs[1, 1])
            self.ax_legend.axis('off')
        elif self.use_section:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], figure=self.fig)
            self.ax_main = self.fig.add_subplot(gs[0, 0])
            self.ax_section = self.fig.add_subplot(gs[1, 0])
        else:
            self.ax_main = self.fig.add_subplot(111)

    def set_main_extent(self, xlim, ylim):
        if xlim: self.ax_main.set_xlim(xlim)
        if ylim: self.ax_main.set_ylim(ylim)

    def set_inset_extent(self, xlim, ylim):
        if self.ax_inset and xlim: self.ax_inset.set_xlim(xlim)
        if self.ax_inset and ylim: self.ax_inset.set_ylim(ylim)
        if self.ax_inset: self.ax_inset.set_title("Врезка")

    def save(self, filename, dpi=200):
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
        print(f"Saved to {filename}")
