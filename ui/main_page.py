from pathlib import Path
from typing import Any
import time

from python_ex.project import Config

import gui_toolbox
from gui_toolbox.viser import Interface_Config

from ui.plotter import Custom_plotter
# from ui.data_table import Data_Editor


class Test_App(gui_toolbox.App):
    def _Set_data(self, **kwarg: Any):
        # set 3d plotter
        _viser = kwarg["viser_plotter"]
        _viser = Custom_plotter(
            Config.Read_from_file(Interface_Config, Path(_viser["cfg_file"]))[1],
            _viser["host"],
            _viser["port"]
        )
        setattr(self, "3d_plotter_thread", _viser)
        getattr(self, "3d_plotter_webview").setUrl(_viser.Get_http_address())

        for _name, _save_dir in kwarg["visualize_data_profiles"].items():
            _viser.load_list.append((_name, _save_dir))

    def _Start(self):
        # run 3d_plotter
        _plotter: Custom_plotter = getattr(self, "3d_plotter_thread")
        _plotter.start()

    def _Stop(self):
        _plotter: Custom_plotter = getattr(self, "3d_plotter_thread")
        _plotter.is_active = False

        while _plotter.isRunning():
            _plotter.is_active = False
            time.sleep(0.1)

    def Set_camera(self):
        ...
        # _popup_page = Data_Editor(_cfg, self)
        # _popup_page.show()

    def Set_pose(self):
        ...
        # _cfg = self.cfg.sub_page_cfg["data_table"]
        # _popup_page = Data_Editor(_cfg, self)
        # _popup_page.show()
