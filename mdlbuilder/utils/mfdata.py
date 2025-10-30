from pathlib import Path

import rasterio
import numpy as np
from flopy.utils.rasters import Raster


class RasterHandler:
    """Обработчик растров (GeoTIFF, ASC, IMG и т.д.)"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Файл растра не найден: {self.path}")

    def read_array(self) -> np.ndarray:
        """Считать растр в numpy-массив."""
        with rasterio.open(self.path) as src:
            return src.read(1)  # первый канал

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Получить границы растра."""
        with rasterio.open(self.path) as src:
            return src.bounds

    def get_crs(self) -> str | None:
        """Получить CRS растра (в формате WKT или EPSG)."""
        with rasterio.open(self.path) as src:
            if src.crs:
                return src.crs.to_string()
            return None

    def resample_to_grid(self, grid, method="nearest") -> np.ndarray:
        """(Пример) Пересэмплировать растр под сетку модели."""
        rio = Raster.load(self.path)
        return rio.resample_to_grid(grid, band=rio.bands[0], method=method)

    @staticmethod
    def reduce_arrays(arrays) -> np.ndarray:
        rst = np.stack(arrays)
        ln = rst.shape[0] - 1

        rst_flipped = rst[::-1]
        return np.array([np.minimum.reduce(rst_flipped[i:]) for i in range(ln + 1)])[::-1]

    @staticmethod
    def expand_arrays(adjusted: np.ndarray, expand_val: float) -> np.ndarray:
        tops = adjusted[:-1]
        bottoms = adjusted[1:]
        diff = tops - bottoms
        too_thin = diff < expand_val
        bottoms[too_thin] = tops[too_thin] - expand_val
        adjusted[1:] = bottoms
        return adjusted


class VectorHandler:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Файл геометрии не найден: {self.path}")


class FieldResolver:
    """
    Универсальный резолвер для любого поля SourceSinksFeature.
    Умеет работать с числами, полями GeoDataFrame, растровыми файлами, выражениями.
    """

    def __init__(self, value, grid, geom_gdf):
        self.value = value
        self.grid = grid
        self.geom_gdf = geom_gdf
        self._cached_array = None

    def get_value(self, icell, context=None):
        """Получает значение для одной ячейки сетки."""
        # 🔹 Просто число
        if isinstance(self.value, (int, float)):
            return self.value

        # 🔹 Поле в GeoDataFrame
        if isinstance(self.value, str) and self.value in self.geom_gdf.columns:
            return float(self.geom_gdf[self.value].iloc[0])

        # 🔹 Растр (tif, asc)
        from pathlib import Path
        if isinstance(self.value, (str, Path)) and Path(self.value).suffix.lower() in {'.tif', '.asc'}:
            if self._cached_array is None:
                raster = RasterHandler(self.value)
                self._cached_array = raster.resample_to_grid(self.grid)
            return float(self._cached_array[icell])

        # 🔹 Выражение (например, 'stage - 3')
        if isinstance(self.value, str) and any(op in self.value for op in ('-', '+', '*', '/')):
            try:
                return float(eval(self.value, {}, context or {}))
            except Exception as e:
                raise ValueError(f"Ошибка в выражении {self.value}: {e}")

        raise TypeError(f"Некорректный тип значения: {type(self.value)}")


class FieldResolverCache:
    """
    Класс, создающий и кэширующий FieldResolver-ы для одной Feature.
    """

    def __init__(self, feature, grid, geom_gdf):
        self.feature = feature
        self.grid = grid
        self.geom_gdf = geom_gdf
        self._cache = self._build_cache()

    def _build_cache(self) -> dict[str, FieldResolver]:
        """Создаёт FieldResolver для всех параметров фичи."""
        from mdlbuilder.dto.packages import SourceSinksFeature

        base_fields = set(SourceSinksFeature.model_fields.keys())
        cache = {}
        for name in type(self.feature).model_fields.keys():
            if name in base_fields:
                continue
            val = getattr(self.feature, name, None)
            cache[name] = FieldResolver(val, self.grid, self.geom_gdf)
        return cache

    def resolve_all(self, icell) -> dict[str, float]:
        """Возвращает вычисленные значения всех полей для одной ячейки."""
        result = {}
        for name, resolver in self._cache.items():
            result[name] = resolver.get_value(icell, context=result)
        # вызов postprocess(), если определён в модели
        if hasattr(self.feature, "postprocess"):
            result = self.feature.postprocess(result)
        return result
