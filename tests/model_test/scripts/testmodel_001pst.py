# Создаем шаблонный файл для гидравлической проводимости
model_ws = "model_ws"
tpl_file = f"{model_ws}/hk.tpl"
with open(tpl_file, "w") as f:
    f.write("ptf ~\n")  # Указывает символ шаблона
    f.write("hk  ~   hk_value   ~\n")  # Переменная для калибровки

# Создаем файл инструкций для извлечения уровней воды
ins_file = f"{model_ws}/head.ins"
with open(ins_file, "w") as f:
    f.write("pif ~\n")  # Указывает символ инструкции
    for obs in ["obs1", "obs2"]:  # Наблюдения
        f.write(f"l1 !{obs}!\n")  # Извлекаем уровни воды для каждого наблюдения


# Создаем управляющий файл
pst_file = f"{model_ws}/model.pst"

# Параметры PEST
parameters = {
    "hk_value": [1.0, 0.1, 10.0]  # [Начальное значение, минимальное, максимальное]
}

# Наблюдения (известные уровни воды)
observations = {
    "obs1": 10.8,
    "obs2": 11.3,
}

# Вес наблюдений
weights = {
    "obs1": 1.0,
    "obs2": 1.0
}

# Генерация управляющего файла
with open(pst_file, "w") as f:
    f.write("pcf\n")
    f.write("* control data\n")
    f.write("RSTFLE F\n")
    f.write("PESTMODE ESTIMATION\n")
    f.write("* parameter groups\n")
    f.write("hk_value relative 0.01 0.0 switch 2.0 parabolic\n")
    f.write("* parameter data\n")
    for param, vals in parameters.items():
        f.write(f"{param} log {vals[1]} {vals[2]} {vals[0]} adjustable\n")
    f.write("* observation groups\n")
    f.write("heads\n")
    f.write("* observation data\n")
    for obs, value in observations.items():
        weight = weights.get(obs, 1.0)
        f.write(f"{obs} {value} {weight}\n")
    f.write("* model command line\n")
    f.write("mf2005.exe\n")
    f.write("* model input/output\n")
    f.write("hk.tpl hk\n")
    f.write("head.ins model.hds\n")