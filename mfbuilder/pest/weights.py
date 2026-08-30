import numpy as np
import pyemu

from mfbuilder.dto.pest import WeightRule, TiedParameterConfig


class WeightRuleEngine:
    """Применяет список WeightRule по порядку к pst.observation_data — каждое
    следующее правило может перекрыть эффект предыдущего на пересекающихся
    строках (ровно так это делалось руками, построчными масками, во всех
    текущих pest_*.py скриптах — тут декларативно и в одном месте).

    PstFrom всегда выставляет наблюдениям ненулевой вес по умолчанию, поэтому
    типичный первый элемент списка — общее "weight: 0.0" без фильтра по
    имени/группе, дальше — точечные правила поверх него.

    Фильтры сравниваются не с pst.observation_data['obsnme'/'obgnme'] — при
    longnames=False (см. PestDirector) PstFrom заменяет их короткими
    автосгенерированными кодами ('ob0', 'obg0', ...), не связанными с тем, что
    объявлено в конфиге. Настоящие, предсказуемые имена/группы (obsgp из
    ObservationGroupConfig, index_cols-значения в имени) сохраняются в
    служебных колонках 'oglong'/'longname' — по ним и фильтруем.
    """

    def apply(self, pst: pyemu.Pst, rules: list[WeightRule]) -> None:
        obs = pst.observation_data
        for rule in rules:
            mask = self._mask(obs, rule)
            if rule.weight is not None:
                obs.loc[mask, "weight"] = rule.weight
            if rule.obsval is not None:
                obs.loc[mask, "obsval"] = rule.obsval

    @staticmethod
    def _mask(obs, rule: WeightRule) -> np.ndarray:
        if rule.obsgp is None and rule.name_contains is None and rule.names is None:
            raise ValueError("WeightRule должен задавать хотя бы один фильтр (obsgp/name_contains/names).")

        mask = np.ones(len(obs), dtype=bool)
        if rule.obsgp is not None:
            mask &= (obs["oglong"] == rule.obsgp).to_numpy()
        if rule.name_contains is not None:
            mask &= obs["longname"].str.contains(rule.name_contains, regex=False).to_numpy()
        if rule.names is not None:
            mask &= obs["longname"].isin(rule.names).to_numpy()
        return mask


class TiedParameterApplier:
    """pst.parameter_data.partrans = 'tied' постфактум — тем же способом,
    что в aibat's pest_v2.py (прямая правка parameter_data, не аргумент
    add_parameters).

    par_names/tied_to сравниваются и с parnme (короткое сгенерированное имя),
    и с longname (полное, читаемое, но заранее непредсказуемое имя) — обычно
    tied заполняется вторым проходом, после того как собранный pst уже
    посмотрели глазами (pst.parameter_data[['parnme','longname']])."""

    def apply(self, pst: pyemu.Pst, tied: list[TiedParameterConfig]) -> None:
        if not tied:
            return
        pdata = pst.parameter_data
        if "partied" not in pdata.columns:
            pdata["partied"] = np.nan

        for rule in tied:
            mask = pdata["parnme"].isin(rule.par_names) | pdata["longname"].isin(rule.par_names)
            if not mask.any():
                raise ValueError(f"Параметры для tied не найдены: {rule.par_names}")

            tied_to_row = pdata.loc[(pdata["parnme"] == rule.tied_to) | (pdata["longname"] == rule.tied_to)]
            if tied_to_row.empty:
                raise ValueError(f"tied_to='{rule.tied_to}' — такого параметра нет.")
            tied_to_parnme = tied_to_row["parnme"].iloc[0]

            pdata.loc[mask, "partrans"] = "tied"
            pdata.loc[mask, "partied"] = tied_to_parnme
