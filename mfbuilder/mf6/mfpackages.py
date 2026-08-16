import geopandas as gpd
from shapely.geometry import Point
from flopy.mf6 import (ModflowGwf, ModflowGwfriv, ModflowGwfdrn,
                       ModflowGwfghb, ModflowGwfwel, ModflowGwflak, ModflowGwfchd)
from mfbuilder.mdlbase.mixins import VertexGridMixin, StructuredGridMixin
from mfbuilder.utils.mfdata import FieldResolverCache

class MF6StructuredRivHandler(StructuredGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        icell = tuple(icell)
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, *icell), vals["stage"], vals["cond"], vals["elev"], bname]


class MF6VertexRivHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfriv(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        # Scale cond proportionally to the length of the river within this cell.
        # Without this, each cell that intersects the segment gets the full cond,
        # so total conductance grows with refinement level → excessive drainage → dry cells.
        # river_geom = cache.geom_gdf.geometry.iloc[geom_index]
        # total_len = river_geom.length
        # if total_len > 0:
        #     clip_len = river_geom.intersection(self.grid_gdf.geometry.loc[icell]).length
        #     cond = vals["cond"] * clip_len / total_len
        # else:
        #     cond = vals["cond"]
        return [(layer - 1, icell), vals["stage"], vals["cond"], vals["elev"], bname]
        # return [(layer - 1, icell), vals["stage"], cond, vals["elev"], bname]


class MF6VertexGhbHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfghb(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["bhead"], vals["cond"], bname]


class MF6VertexDrnHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfdrn(model, stress_period_data=records, boundnames=True, mover=self.mover)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["head"], vals["cond"], bname]


class MF6VertexWelHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self._iterate_wel_summed()
        return ModflowGwfwel(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["rate"], bname]

    def _iterate_wel_summed(self):
        """Like iterate_features but accumulates rate when multiple wells share a cell."""
        records = {}
        for spd, feature in self.data.items():
            cellmap = {}  # (layer, icell) → [cellid_tuple, rate_sum, bname]

            for f in feature.data:
                geom_gdf = f.get_filtered_geometry()
                resolver_cache = FieldResolverCache(f, self.grid, geom_gdf)

                for geom_index, geom in enumerate(geom_gdf.geometry):
                    cells = self.map_to_grid(geom)
                    layers = f.resolve_layers(geom_gdf, geom_index)
                    bname = resolver_cache.resolve_boundname(geom_gdf, geom_index)

                    for layer in layers:
                        for icell in cells:
                            vals = resolver_cache.resolve_all(icell, geom_index)
                            rate = vals["rate"]
                            key = (layer, icell)
                            if key in cellmap:
                                cellmap[key][1] += rate
                            else:
                                cellmap[key] = [(layer - 1, icell), rate, bname]

            records[spd] = list(cellmap.values())
        return records


class MF6VertexChdHandler(VertexGridMixin):
    def build_package(self, model: ModflowGwf):
        records = self.iterate_features(self.build_record)
        return ModflowGwfchd(model, stress_period_data=records, boundnames=True)

    def build_record(self, layer, icell, cache, bname=None, geom_index=0):
        vals = cache.resolve_all(icell, geom_index)
        return [(layer - 1, icell), vals["head"], bname]


class MF6VertexLakHandler(VertexGridMixin):
    """Озёра/пруды (полигоны) как пакет LAK. Каждый полигон -> одно озеро,
    подключенное вертикально ко всем ячейкам сетки, которые он покрывает."""

    claktype = "VERTICAL"

    def build_package(self, model: ModflowGwf):
        periods = sorted(self.data.keys())
        packagedata = []
        connectiondata = []
        perioddata = {}
        lake_boundnames = []
        lake_geoms = []

        for period in periods:
            zone = self.data[period]
            period_records = []
            lakeno = 0

            for feature in zone.data:
                geom_gdf = feature.get_filtered_geometry()
                resolver_cache = FieldResolverCache(feature, self.grid, geom_gdf)

                for geom_index, geom in enumerate(geom_gdf.geometry):
                    cells = self.map_to_grid(geom)
                    if not cells:
                        continue
                    layers = feature.resolve_layers(geom_gdf, geom_index)
                    vals = resolver_cache.resolve_all(cells[0], geom_index)

                    if "precip" in vals and "runoff_coeff" in vals:
                        # runoff = площадь_пруда * catchment_multiplier * precip * runoff_coeff -
                        # автоматический расчёт вместо ручного числа (см. LakFeature.precip/runoff_coeff).
                        catchment_mult = vals.get("catchment_multiplier", 4.0)
                        catchment_area = geom.area * catchment_mult
                        vals["runoff"] = catchment_area * vals["precip"] * vals["runoff_coeff"]

                    if "precip" in vals and vals.get("auto_rainfall", 1.0):
                        # rainfall (LT-1) - прямые осадки на зеркало озера; MF6 сам умножает
                        # на площадь озера, площадь пруда тут учитывать не нужно (в отличие
                        # от runoff, который - объёмный расход).
                        vals["rainfall"] = vals["precip"]

                    if period == periods[0]:
                        bname = resolver_cache.resolve_boundname(geom_gdf, geom_index)
                        iconn = 0
                        for layer in layers:
                            for icell in cells:
                                connectiondata.append([
                                    lakeno, iconn, (layer - 1, icell),
                                    self.claktype, vals["cond"], 0.0, 0.0, 0.0, 0.0,
                                ])
                                iconn += 1
                        packagedata.append([lakeno, vals["head"], iconn, bname])
                        lake_boundnames.append(bname)
                        lake_geoms.append(geom)

                    period_records.append([lakeno, "evaporation", vals["evaporation"]])
                    period_records.append([lakeno, "runoff", vals["runoff"]])
                    if "rainfall" in vals:
                        period_records.append([lakeno, "rainfall", vals["rainfall"]])

                    status = getattr(feature, "status", None)
                    if status:
                        period_records.append([lakeno, "status", status.upper()])
                        if status.upper() == "CONSTANT":
                            # Уровень, на котором принудительно держим озеро (например, НПУ) -
                            # используется, чтобы через невязку бюджета озера ("CONSTANT")
                            # узнать требуемый расход дополнительной подпитки.
                            period_records.append([lakeno, "stage", vals["head"]])

                    lakeno += 1

            perioddata[int(period)] = period_records

        outlets = self._build_outlets(self.data[periods[0]], lake_boundnames, lake_geoms)
        observations = self._build_observations(model, lake_boundnames, outlets)

        return ModflowGwflak(
            model,
            boundnames=True,
            mover=self.mover,
            print_input=True,
            print_stage=True,
            print_flows=True,
            save_flows=True,
            budget_filerecord=f"{model.name}.lak.bud",
            budgetcsv_filerecord=f"{model.name}.lak.bud.csv",
            stage_filerecord=f"{model.name}.lak.stage.bin",
            observations=observations,
            nlakes=len(packagedata),
            noutlets=len(outlets),
            packagedata=packagedata,
            connectiondata=connectiondata,
            outlets=outlets or None,
            time_conversion=86400,
            perioddata=perioddata,
        )

    # Виды наблюдений LAK OBS, применимые к любому озеру независимо от его связей
    # (переливов/mover) - проверено запуском MF6 6.5.0.
    _UNIVERSAL_OBS_TYPES = (
        "stage", "ext-inflow", "rainfall", "runoff", "withdrawal",
        "evaporation", "storage", "constant", "volume", "surface-area",
    )
    # Виды с адресацией "по связи" (ID2=iconn) - MF6 не умеет резолвить их по
    # boundname, если boundname сам выглядит как целое число (например, пруд
    # с именем "2"/"3"), поэтому для таких боунднеймов их приходится пропускать.
    _PER_CONNECTION_OBS_TYPES = ("lak", "conductance")

    def _build_observations(self, model, lake_boundnames, outlets):
        """Настраивает вывод бюджета по каждому пруду отдельно (LAK OBS) -
        нужен, чтобы читать текущий бюджет каждого пруда (осадки/испарение/
        сток/обмен с ПВ/переток по трубам) и невязку при status=CONSTANT."""
        outlet_source = {rec[1] for rec in outlets}  # lakeno, владеющие своим outlet'ом
        outlet_external = {rec[1] for rec in outlets if rec[2] == -1}  # ... с внешним сбросом

        records = []
        for lakeno, bname in enumerate(lake_boundnames):
            types = list(self._UNIVERSAL_OBS_TYPES)
            if not bname.lstrip("-").isdigit():
                types.extend(self._PER_CONNECTION_OBS_TYPES)
            if lakeno in outlet_source:
                types.append("outlet-inflow")
            if lakeno in outlet_external:
                types.append("ext-outflow")
            for otype in types:
                records.append((f"{bname}_{otype.replace('-', '_')}", otype, bname))

        if not records:
            return None
        return {f"{model.name}.lak.obs.csv": records}

    def _build_outlets(self, zone, lake_boundnames, lake_geoms):
        """Строит блок OUTLETS пакета LAK по линиям перетоков между озёрами
        (например, TrubiSvzpi.shp - переливные трубы между прудами,
        DrenaGlavnaa.shp - выпуск из пруда за пределы модели)."""
        if not getattr(zone, "outlets", None):
            return []

        lake_gdf = gpd.GeoDataFrame(
            {"lakeno": range(len(lake_boundnames)), "boundname": lake_boundnames},
            geometry=lake_geoms,
        )

        outlets = []
        outletno = 0
        for outlet_cfg in zone.outlets:
            lines_gdf = outlet_cfg.get_filtered_geometry()
            tol = outlet_cfg.match_tolerance
            for _, row in lines_gdf.iterrows():
                coords = list(row.geometry.coords)
                p_start, p_end = Point(coords[0]), Point(coords[-1])

                lakein = self._nearest_lake(lake_gdf, p_start, tol)
                if lakein is None:
                    continue  # линия не начинается в озере - не переток LAK

                lakeout_match = self._nearest_lake(lake_gdf, p_end, tol)
                lakeout = lakeout_match if lakeout_match is not None else -1

                outlets.append([
                    outletno, lakein, lakeout, outlet_cfg.couttype,
                    self._resolve_row_value(outlet_cfg.invert, row),
                    self._resolve_row_value(outlet_cfg.width, row),
                    self._resolve_row_value(outlet_cfg.rough, row),
                    self._resolve_row_value(outlet_cfg.slope, row),
                ])
                outletno += 1
        return outlets

    @staticmethod
    def _nearest_lake(lake_gdf, point, tol):
        distances = lake_gdf.geometry.distance(point)
        idx = distances.idxmin()
        if distances.loc[idx] <= tol:
            return int(lake_gdf.loc[idx, "lakeno"])
        return None

    @staticmethod
    def _resolve_row_value(value, row):
        if isinstance(value, str) and value in row.index:
            return float(row[value])
        return float(value)
