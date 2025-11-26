import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, mapping
import fiona
import pathlib


class RasterMf:
    """
    Класс для работы с растровыми данными, включая операции вычитания
    и обрезки по полигону.
    """

    def __init__(self, filepath: str):
        """
        Инициализация объекта растра.

        :param filepath: Путь к растровому файлу.
        """
        self.filepath = filepath
        self.dataset = rasterio.open(filepath)

    def __del__(self):
        """Закрывает файл при удалении объекта."""
        self.dataset.close()

    def _resample_raster(self, source_raster, target_transform, target_shape, target_crs):
        """
        Ресемплирует растр под заданные параметры.

        :param source_raster: Открытый объект растра для ресемплинга.
        :param target_transform: Трансформация целевого растра.
        :param target_shape: Размеры (высота, ширина) целевого растра.
        :param target_crs: Система координат целевого растра.
        :return: Ресемплированный массив данных.
        """
        # Читаем данные исходного растра
        source_data = source_raster.read(1)

        # Подготовка массива для результата ресемплинга
        resampled_data = np.empty(target_shape, dtype=source_raster.dtypes[0])

        # Выполняем ресемплинг
        reproject(
            source=source_data,
            destination=resampled_data,
            src_transform=source_raster.transform,
            src_crs=source_raster.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

        return resampled_data

    def subtract(self, other_raster_path: str, output_path: str):
        """
        Вычитает значения из другого растрового файла.

        :param other_raster_path: Путь к растровому файлу, который нужно вычесть.
        :param output_path: Путь для сохранения нового растрового файла.
        """
        with rasterio.open(other_raster_path) as other_raster:
            # Проверяем размеры растров
            if self.dataset.shape != other_raster.shape or self.dataset.transform != other_raster.transform:
                # Ресемплирование второго растра под размер первого
                target_shape = self.dataset.shape
                target_transform = self.dataset.transform
                target_crs = self.dataset.crs

                resampled_data = self._resample_raster(
                    source_raster=other_raster,
                    target_transform=target_transform,
                    target_shape=target_shape,
                    target_crs=target_crs,
                )
            else:
                # Если размеры совпадают, читаем данные напрямую
                resampled_data = other_raster.read(1)

            # Вычитание растров
            result_data = self.dataset.read(1) - resampled_data

            # Параметры для нового файла
            profile = self.dataset.profile
            profile.update(dtype=rasterio.float32)

            # Сохранение результата
            with rasterio.open(output_path, 'w', **profile) as dest:
                dest.write(result_data, 1)

    def clip_by_polygon(self, shapefile_path: str, output_path: str):
        """
        Обрезает растр по заданному полигону.

        :param shapefile_path: Путь к shapefile с полигоном.
        :param output_path: Путь для сохранения обрезанного растрового файла.
        """
        with fiona.open(shapefile_path, "r") as shapefile:
            # Предполагается, что в shapefile один полигон
            geoms = [shape(feature["geometry"]) for feature in shapefile]

        # Преобразование геометрии в формат GeoJSON
        geoms_json = [mapping(geom) for geom in geoms]

        # Маскирование растров
        out_image, out_transform = mask(self.dataset, geoms_json, crop=True)

        # Параметры для сохранения
        profile = self.dataset.profile
        profile.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Сохранение результата
        with rasterio.open(output_path, 'w', **profile) as dest:
            dest.write(out_image)

    def delete_raster(self):
        """
        Удаляет файл растрового изображения.

        :raises FileNotFoundError: Если файл не существует.
        :raises PermissionError: Если недостаточно прав для удаления.
        """
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
                print(f"Файл {self.filepath} успешно удалён.")
            except PermissionError as e:
                raise PermissionError(f"Нет прав для удаления файла: {self.filepath}.") from e
            except Exception as e:
                raise RuntimeError(f"Ошибка при удалении файла {self.filepath}: {e}") from e
        else:
            raise FileNotFoundError(f"Файл {self.filepath} не найден.")

    def get_sample(self, xy, filename):
        points = gpd.read_file(xy)
        coord_list = [(x, y,) for x, y in zip(points["geometry"].x, points["geometry"].y)]
        points["S"] = [round(x[0], 2) for x in self.dataset.sample(coord_list)]
        path_dir = pathlib.Path(filename)
        ws = path_dir.parent
        if not os.path.exists(ws):
            os.mkdir(ws)
        points.to_csv(filename)