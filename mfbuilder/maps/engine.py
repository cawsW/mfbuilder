import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import Collection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS
import textwrap

from mfbuilder.maps.layout import MapLayout
from mfbuilder.maps.loader import LayerFactory
from mfbuilder.dto.maps import RootConfig


class MapBuilder:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
        self.config = RootConfig(**raw_data)
        try:
            self.project_crs = CRS.from_user_input(self.config.settings.crs)
        except Exception:
            self.project_crs = self.config.settings.crs

        self.layout = MapLayout(
            figsize=self.config.settings.figsize,
            use_inset=self.config.inset_map.enabled
        )

        self.main_layers = []
        for layer_conf in self.config.main_map.layers:
            self.main_layers.append(LayerFactory.create_layer(layer_conf, self.project_crs))

        self.inset_layers = []
        if self.config.inset_map.enabled:
            source_confs = self.config.inset_map.layers or self.config.main_map.layers
            for layer_conf in source_confs:
                self.inset_layers.append(LayerFactory.create_layer(layer_conf, self.project_crs))

        self._apply_global_styles()

    def _apply_global_styles(self):
        base_size = self.config.settings.base_fontsize

        # Настраиваем основные элементы Matplotlib на лету
        plt.rcParams.update({
            'font.size': base_size,  # Базовый размер
            'axes.titlesize': base_size + 2,  # Заголовки осей чуть больше
            'axes.labelsize': base_size,  # Подписи осей
            'xtick.labelsize': base_size - 1,  # Координаты по X
            'ytick.labelsize': base_size - 1,  # Координаты по Y
            'legend.fontsize': base_size,  # Шрифт в самой легенде
            'legend.title_fontsize': base_size + 1  # Заголовок легенды
        })
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': base_size,
        })

    def build(self):
        if self.config.main_map.xlim and self.config.main_map.ylim:
            self.layout.set_main_extent(
                self.config.main_map.xlim,
                self.config.main_map.ylim
            )

        basemaps = [l for l in self.main_layers if getattr(l.config, 'type', '') == 'basemap']
        data_layers = [l for l in self.main_layers if getattr(l.config, 'type', '') != 'basemap']

        for layer in data_layers:
            layer.draw(self.layout.ax_main)

        for layer in basemaps:
            layer.draw(self.layout.ax_main)

        if self.layout.ax_inset:
            if self.config.inset_map.xlim:
                self.layout.set_inset_extent(self.config.inset_map.xlim, self.config.inset_map.ylim)

            for layer in self.inset_layers:
                layer.draw(self.layout.ax_inset)

        self._compile_legend()

        self._add_colorbars()

        print(f"Saving map to {self.config.settings.output}...")
        self.layout.save(self.config.settings.output)

    def _add_colorbars(self):
        target_layer = None
        for layer in self.main_layers:
            if hasattr(layer, 'mappable') and layer.mappable is not None:
                if hasattr(layer.config, 'colorbar') and layer.config.colorbar.enabled:
                    target_layer = layer
                    break

        if target_layer:
            cb_conf = target_layer.config.colorbar
            divider = make_axes_locatable(self.layout.ax_main)

            if cb_conf.orientation == 'horizontal':
                cax = divider.append_axes("bottom", size="2%", pad=0.4)
            else:
                cax = divider.append_axes("right", size="2%", pad=0.2)

            cbar = self.layout.fig.colorbar(
                target_layer.mappable, cax=cax, orientation=cb_conf.orientation
            )
            cbar.set_label(cb_conf.label) #fontsize=10
            # cbar.ax.tick_params(labelsize=9)

    def _compile_legend(self):
        # 1. Получаем автоматические хендлы от matplotlib (то, что нарисовалось через plot)
        handles, labels = self.layout.ax_main.get_legend_handles_labels()

        # 2. Фильтрация: убираем сложные коллекции (PatchCollection), с которыми плохо работает легенда
        # и оставляем только то, что явно является линией или патчем
        valid_handles = []
        valid_labels = []
        for h, l in zip(handles, labels):
            # Фильтруем системные слои и коллекции без явной поддержки легенды
            if l != '_nolegend_' and not isinstance(h, Collection):
                valid_handles.append(h)
                valid_labels.append(l)

        handles, labels = valid_handles, valid_labels

        # 3. Добавляем хендлы от слоев (через метод get_legend_handles)
        for layer in self.main_layers:
            extra_handles = layer.get_legend_handles()
            for h in extra_handles:
                # Дедупликация: добавляем только если такого лейбла еще нет
                if getattr(h, '_mfbuilder_force_legend', False):
                    handles.append(h)
                    labels.append(h.get_label())
                elif h.get_label() not in labels:
                    handles.append(h)
                    labels.append(h.get_label())

        # 4. Добавляем ручную легенду из конфига (ИСПРАВЛЕНИЕ ЗДЕСЬ)
        for item in self.config.legend:
            # Проверяем на дубликаты, если нужно (опционально)
            if item.label in labels:
                continue

            if item.type == 'patch':
                h = Patch(facecolor=item.color, label=item.label)
                handles.append(h)
                labels.append(item.label)  # <--- ЭТОЙ СТРОКИ НЕ ХВАТАЛО
            elif item.type == 'line':
                h = Line2D([0], [0], color=item.color, label=item.label)
                handles.append(h)
                labels.append(item.label)  # <--- ЭТОЙ СТРОКИ НЕ ХВАТАЛО

        # 5. Отрисовка
        max_width = self.config.settings.legend_wrap_width if hasattr(self.config.settings, 'legend_wrap_width') else 30

        wrapped_labels = [
            "\n".join(textwrap.wrap(label, width=max_width))
            for label in labels
        ]

        # 6. Отрисовка
        target_ax = self.layout.ax_legend if self.layout.ax_legend else self.layout.ax_main
        loc = 'center' if self.layout.ax_legend else self.config.settings.legend_loc

        if handles:
            leg = target_ax.legend(
                handles=handles,
                labels=wrapped_labels,  # Используем обработанные метки
                loc=loc,
                title="Условные обозначения"
            )

            # Опционально: выравнивание текста по левому краю, если Matplotlib центрирует многострочный текст
            for text in leg.get_texts():
                text.set_ha('left')
