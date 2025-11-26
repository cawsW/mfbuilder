import flopy

# Создаем папку для модели
model_name = "simple_model"
model_ws = "./model_ws"

# Создаем объект модели
mf = flopy.modflow.Modflow(model_name, exe_name="../../bin/mf2005", model_ws=model_ws)

# Размеры сетки
nlay = 1  # Количество слоев
nrow = 10  # Количество строк
ncol = 10  # Количество столбцов
delr = delc = 100.0  # Размер ячейки (м)

ibound = [[[1
           for col in range(ncol)] for row in range(nrow)]]
print(ibound)
strt = 10.0  # Начальный уровень воды (м)

# Создаем пакеты модели
dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=10, botm=0)

bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
chd = flopy.modflow.ModflowChd(mf, stress_period_data=[[0, row, col, 10 if row==0 else 11, 10 if row ==0 else 11 ] for col in range(ncol) for row in range(nrow) if row == 0 or row == nrow - 1])

# Гидравлическая проводимость
hk = 1.0  # Начальное значение K (м/сут)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk)

# Настраиваем пакет вывода
pcg = flopy.modflow.ModflowPcg(mf)

# Указываем, куда сохранять результаты
oc = flopy.modflow.ModflowOc(mf)
ts_obs1 = [[1, 10.8]]  # Временные значения для первой точки
ts_obs2 = [[1, 11.3]]  # Для второй точки

# Определение наблюдений
obs1 = flopy.modflow.HeadObservation(
    mf, layer=0, row=2, column=2, time_series_data=ts_obs1, obsname="obs1"
)
obs2 = flopy.modflow.HeadObservation(
    mf, layer=0, row=7, column=7, time_series_data=ts_obs2, obsname="obs2"
)

# Добавляем наблюдения в пакет HOB
hob = flopy.modflow.ModflowHob(
    mf, iuhobsv=1, hobdry=-9999.0, obs_data=[obs1, obs2],
)
# Сохраняем модель
mf.write_input()
mf.run_model()
import matplotlib.pyplot as plt
import flopy.plot as fplot

# Загружаем выходные файлы модели
headfile = f"{model_ws}/{model_name}.hds"
hds = flopy.utils.HeadFile(headfile)

# Считываем уровни воды
head = hds.get_data(kstpkper=(0, 0))  # Первый временной шаг и период

# Визуализация уровней воды
fig, ax = plt.subplots(figsize=(10, 8))
modelmap = fplot.PlotMapView(model=mf, ax=ax)
contour = modelmap.plot_array(head[0], cmap="viridis")  # Уровни воды для первого слоя
modelmap.plot_grid()  # Добавляем сетку
cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label("Уровень воды (м)")

ax.set_title("Распределение уровней воды")
plt.show()