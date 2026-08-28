"""
Создаёт стандартную структуру папок для нового проекта в mfbuilder/projects
и раскладывает по ней шаблоны конфигов.

Запускается прямо из IDE (без аргументов командной строки) — название
проекта задаётся переменной PROJECT_NAME ниже.

Безопасно перезапускать на уже существующем проекте: mkdir(exist_ok=True)
не трогает то, что уже есть в папках, а copy2() ниже пропускает файл, если
в целевом месте уже лежит файл с таким именем — ничего не перезаписывается
и не удаляется.
"""
import shutil
from pathlib import Path

PROJECT_NAME = "test_v1"

TEMPLATES_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = TEMPLATES_DIR.parent / "projects"

SUBDIRS = [
    "gis",
    "notebooks",
    "scripts",
    "report",
    "source",
    "models",
    "input/vectors",
    "input/rasters",
    "input/tables",
    "output/vectors",
    "output/rasters",
    "output/tables",
    "output/pictures",
    "configs/pics",
]

MAIN_CONFIG_TEMPLATE = TEMPLATES_DIR / "model_config.yml"
MAIN_CONFIG_DEST_NAME = "config_v1.yml"
PICS_TEMPLATES_DIR = TEMPLATES_DIR / "config_pics"
START_SCRIPT_TEMPLATE = TEMPLATES_DIR / "start_model.py"


def _copy_if_missing(src: Path, dst: Path) -> None:
    """Копирует src -> dst, только если dst ещё не существует."""
    if dst.exists():
        print(f"  пропущено (уже существует): {dst}")
        return
    shutil.copy2(src, dst)
    print(f"  создано: {dst}")


def create_project(name: str) -> Path:
    project_dir = PROJECTS_DIR / name
    for sub in SUBDIRS:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    _copy_if_missing(MAIN_CONFIG_TEMPLATE, project_dir / "configs" / MAIN_CONFIG_DEST_NAME)

    for pic_template in sorted(PICS_TEMPLATES_DIR.glob("*.yml")):
        _copy_if_missing(pic_template, project_dir / "configs" / "pics" / pic_template.name)

    _copy_if_missing(START_SCRIPT_TEMPLATE, project_dir / "scripts" / START_SCRIPT_TEMPLATE.name)

    return project_dir


if __name__ == "__main__":
    path = create_project(PROJECT_NAME)
    print(f"Проект создан: {path}")
