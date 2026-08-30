import pyemu

from mfbuilder.dto.pest import RegularizationConfig


class RegularizationBuilder:
    """Tikhonov-регуляризация (используется только с GLM).

    zero_order_tikhonov — предпочтение к стартовому значению для всех
    параметров. first_order_pearson_tikhonov — предпочтение к похожести
    соседних пилотных точек внутри одной группы, по ковариации того же
    geostruct, что использовался при её параметризации (иначе регуляризация
    не будет согласована с тем, как параметры коррелированы пространственно).

    control_data.pestmode намеренно не переключается в 'regularization' —
    во всех текущих проектах это делалось расчётно ('estimation' с заданными
    reg_data.phimlim/phimaccept), т.к. автопоиск phimlim в pestpp-glm
    оказывался нестабилен при большом числе параметров (пилотные точки).
    """

    def __init__(self, geostructs: dict[str, pyemu.geostats.GeoStruct]):
        self._geostructs = geostructs

    def apply(self, pst: pyemu.Pst, cfg: RegularizationConfig) -> None:
        if cfg.zero_order:
            pyemu.helpers.zero_order_tikhonov(pst, reset=True)

        for pargp in cfg.pilot_point_groups:
            self._apply_first_order(pst, pargp, cfg.abs_drop_tol)

        pst.reg_data.phimlim = cfg.phimlim
        pst.reg_data.phimaccept = cfg.phimaccept if cfg.phimaccept is not None else cfg.phimlim * 1.1
        # pyemu.helpers.*_tikhonov переключают pestmode в 'regularization' сами —
        # возвращаем 'estimation': во всех текущих проектах это осознанный обход
        # нестабильного автопоиска phimlim в pestpp-glm при большом числе
        # параметров (см. docstring класса).
        pst.control_data.pestmode = "estimation"

    def _apply_first_order(self, pst: pyemu.Pst, pargp: str, abs_drop_tol: float) -> None:
        geostruct = self._geostructs.get(pargp)
        if geostruct is None:
            raise ValueError(
                f"Нет geostruct для группы пилотных точек '{pargp}' — first-order регуляризация "
                "требует ту же geostruct, что использовалась при параметризации этой группы."
            )
        # pargp в parameter_data — короткий сгенерированный код (longnames=False
        # в PstFrom), объявленное в конфиге имя группы сохраняется в 'pglong'.
        pdata = pst.parameter_data
        group_df = pdata.loc[pdata["pglong"] == pargp]
        if group_df.empty:
            raise ValueError(f"Группа параметров '{pargp}' не найдена в parameter_data.")
        if not {"x", "y"}.issubset(group_df.columns):
            raise ValueError(f"У группы '{pargp}' нет координат x/y — это не пилотно-точечная параметризация?")

        # parameter_data - строковый DataFrame (пишется в текстовый .pst), x/y нужно привести к float
        cov = geostruct.covariance_matrix(
            x=group_df["x"].astype(float), y=group_df["y"].astype(float), names=group_df["parnme"]
        )
        pyemu.helpers.first_order_pearson_tikhonov(pst, cov, reset=False, abs_drop_tol=abs_drop_tol)
