import pyemu

from mfbuilder.dto.pest import ObservationGroupConfig


class ObservationBuilder:
    """Добавляет один источник наблюдений через PstFrom.add_observations.

    Мягкое ограничение-неравенство — тот же класс: PstFrom сам распознаёт
    obsgp с префиксом 'greater_than_'/'less_than_' и превращает его в
    одностороннее неравенство, отдельный тип конфига не нужен.
    """

    def add(self, pf: pyemu.utils.PstFrom, group: ObservationGroupConfig) -> None:
        pf.add_observations(
            group.file,
            index_cols=group.index_cols,
            use_cols=group.use_cols,
            use_rows=group.use_rows,
            prefix=group.prefix,
            obsgp=group.obsgp,
            **group.extra,
        )
