import os
import platform
import stat


class ConfigValidator:
    def __init__(self, config: dict, editing):
        self.config = config
        self.editing = editing

    def validate_config(self):
        self._validate_base()
        self._validate_grid()
        self._validate_parameters()

    def _validate_base(self):
        base = self.config.get("base")
        if not base:
            raise ValueError("No base options specified")
        self._validate_name(base)
        self._validate_workspace(base)
        self._validate_mf_exe(base)
        self._validate_perioddata(base)

    def _validate_grid(self):
        grid = self.config.get("grid")
        if not grid and not self.editing:
            raise ValueError("No grid options specified")
        self._validate_boundary(grid)
        self._validate_grid_type(grid)
        self._validate_gridgen_exe(grid)
        self._validate_cell_size(grid)
        self._validate_nlay(grid)
        self._validate_top(grid)
        self._validate_botm(grid)

    def _validate_parameters(self):
        if not self.config.get("parameters") and not self.editing:
            raise ValueError("No parameters options specified")

    @staticmethod
    def _validate_name(base: dict):
        if not base.get("name"):
            raise ValueError("No name specified")

    @staticmethod
    def _validate_workspace(base: dict):
        workspace = base.get("workspace")
        if not workspace:
            raise ValueError("No workspace specified")
        if not os.path.exists(workspace):
            os.mkdir(workspace)

    def _validate_mf_exe(self, base: dict):
        exe = base.get("exe")
        if not exe:
            raise ValueError("No executable specified")
        self._check_exe_exists(exe)
        if platform.system() != "Windows":
            st = os.stat(exe)
            os.chmod(exe, st.st_mode | stat.S_IEXEC)

    @staticmethod
    def _validate_perioddata(base: dict):
        if not base.get("perioddata") and not base.get("steady"):
            raise ValueError("No time steps specified for transient model")

    @staticmethod
    def _validate_boundary(grid: dict):
        if not grid.get("boundary"):
            raise ValueError("No boundary specified")

    @staticmethod
    def _validate_cell_size(grid:dict):
        if not grid.get("cell_size"):
            raise ValueError("No cell size specified")

    @staticmethod
    def _validate_nlay(grid: dict):
        if not grid.get("nlay"):
            raise ValueError("No number of layers specified")

    @staticmethod
    def _validate_top(grid: dict):
        if not grid.get("top"):
            raise ValueError("No top specified")

    def _validate_botm(self, grid: dict):
        botm = grid.get("botm")
        nlay = grid.get("nlay")
        if not botm:
            raise ValueError("No bottom specified")
        if len(botm) != nlay and not self.editing:
            raise ValueError("Number of botm layers does not match number of layers")

    @staticmethod
    def _validate_grid_type(grid: dict):
        if not grid.get("type"):
            raise ValueError("No grid type specified")

    def _validate_gridgen_exe(self, grid: dict):
        typegrd = grid.get("type")
        if typegrd != "structured":
            if grid.get("gridgen_exe"):
                gridgen_exe = grid.get("gridgen_exe")
                self._check_exe_exists(gridgen_exe)
                if platform.system() != "Windows":
                    st = os.stat(gridgen_exe)
                    os.chmod(gridgen_exe, st.st_mode | stat.S_IEXEC)
            else:
                raise ValueError("No gridgen executable specified")

    @staticmethod
    def _check_exe_exists(exe):
        if not os.path.isfile(exe):
            raise ValueError(f"{exe}: exe file doesn't exist")