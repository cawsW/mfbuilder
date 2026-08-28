from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class ZoneBudgetConfig(BaseModel):
    """Настройки постпроцессинга ZoneBudget — считается после запуска модели
    (нужны готовые .grb/.cbc файлы), выводом являются CSV-таблицы баланса.

    Всегда (при enabled: true) строятся две таблицы:
      - по зонам из `zones` (если задан) либо по слоям (если не задан);
      - итоговая — суммарный баланс по всей активной области модели.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(default=True, description="Считать баланс через ZoneBudget после расчёта модели")
    zones: Path | None = Field(
        default=None,
        description="Векторный файл с полем-номером зоны на ячейку. Не задан — зоны = номера слоёв."
    )
    zone_field: str = Field(default="zone", description="Имя поля с номером зоны в файле zones")
    exe_path: Path | None = Field(
        default=None, description="Путь к zbud6 (по умолчанию — рядом с exe_path модели, .../bin/zbud6)"
    )
    output_dir: Path = Field(default=Path("../output/tables"), description="Куда сохранять CSV с балансом")
