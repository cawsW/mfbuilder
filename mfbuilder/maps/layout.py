import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MapLayout:
    def __init__(self, figsize=(12, 10), use_inset=False, use_section=False):
        self.fig = plt.figure(figsize=figsize, layout='constrained')
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
                width_ratios=[3, 1.2],
                height_ratios=[3, 1.2, 1.2],
                figure=self.fig
            )
            self.ax_main = self.fig.add_subplot(gs[0:2, 0])
            self.ax_inset = self.fig.add_subplot(gs[0, 1])
            self.ax_legend = self.fig.add_subplot(gs[1, 1])
            self.ax_section = self.fig.add_subplot(gs[2, :])

            self.ax_legend.axis('off')
        elif self.use_inset:
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
        # if self.ax_inset: self.ax_inset.set_title("Врезка")

    def align_axes(self, event=None):
        self.ax_main.set_anchor('C')

        pos_main = self.ax_main.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        pos_sec = self.ax_section.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        fig_width, fig_height = self.fig.get_size_inches()

        x0 = pos_main.x0 / fig_width
        width = pos_main.width / fig_width
        y0_sec = pos_sec.y0 / fig_height
        height_sec = pos_sec.height / fig_height

        self.ax_section.set_position([x0, y0_sec, width, height_sec])

    def save(self, filename, dpi=200):
        self.ax_main.set_aspect('equal', adjustable='datalim')
        # self.ax_main.set_aspect('auto')
        if self.use_section and self.use_inset:
            self.fig.set_layout_engine(None)
            self.fig.canvas.mpl_connect('draw_event', self.align_axes)
            self.ax_inset.set_aspect('equal', adjustable='datalim')
        # if self.use_inset:
            # self.ax_inset.set_aspect('auto')
            # self.ax_legend.set_aspect(1, anchor="N")
            # self.ax_legend.set_aspect('auto')
        # plt.tight_layout()
        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi) #dpi=dpi)
        print(f"Saved to {filename}")
