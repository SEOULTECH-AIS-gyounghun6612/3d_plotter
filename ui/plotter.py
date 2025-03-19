from __future__ import annotations
from typing import Any

import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from PySide6.QtCore import QObject

from viser import ViserServer, SceneNodeHandle

from python_ex.system import Path_utils
from python_ex.file import Json
from python_ex.project import Config
from gui_toolbox.viser import Server, Interface_Config, Draw

from .utils.camera_and_scene import Scene_Sequence, Camera, Point_Cloud


_cam_ref = R.from_euler(
    "xyz", (-90, 0, -90), degrees=True).as_quat(scalar_first=True)


class Draw_Cfg():
    @dataclass
    class Basement(Config.Basement):
        display_term: int = 30

    @dataclass
    class Cam_Path(Config.Basement):
        color: tuple[int, int, int] = (255, 0, 255)
        visible: bool = True
        line_width: int = 5

    @dataclass
    class Point_Cloud(Config.Basement):
        visible: bool = True
        point_size: float = 0.001
        point_shape: str = "rounded"


class Renderer():
    @staticmethod
    def Cam_path(
        viser_server: ViserServer,
        names: tuple[str, str],  # profile, data
        data: dict[int, Camera],
        display_term: int, display_cfg: Draw_Cfg.Cam_Path
    ):
        _c = display_cfg.color
        _visible = display_cfg.visible
        _term = display_term

        _tr: list[list[float]] = []

        _cam_handler: list[Any] = []
        _frame_handler: dict[int, list] = {}
        for _id, _camera in data.items():
            _quat, _tp = _camera.Get_pose_to_vector("quat")
            _tr.append(_tp)

            if _id % _term:
                continue

            # set cam frame
            _name = f"{names[0]}/frame/{names[1]}/{_id}"
            _arg = {
                "name": _name, "color": _c,
                "position": tuple(_tp), "wxyz": tuple(_quat),
                "visible": _visible
            }

            _frame_handler[_id] = [Draw.Frame(viser_server, **_arg)]

            _arg = {
                "name": f"{_name}/cam", "color": _c,
                "img": _camera.img,
                "focal_length": sum(_camera.k_values[:2]) / 2
            }
            _cam_handler.append(Draw.Camera(viser_server, **_arg))  # just draw

        _w = display_cfg.line_width

        if _w:
            # draw trajectory
            _cam_handler.append(Draw.Line(
                viser_server,
                f"{names[0]}/global/trj_{names[1]}", _tr, _c, _w
            ))
        return _frame_handler, _cam_handler

    @staticmethod
    def Point_cloud(
        viser_server: ViserServer,
        names: tuple[str, str, str],  # profile, ord_data, data
        data: dict[int, Point_Cloud],
        display_term: int, display_cfg: Draw_Cfg.Point_Cloud
    ):
        _term = display_term

        _pts_handler: list[Any] = []
        for _id, _pts in data.items():
            if _id < 0:  # background
                _name = f"{names[0]}/global/bg_{names[1]}"
            elif _id % _term:
                continue
            else:
                _name = f"{names[0]}/frame/{names[1]}/{_id}/{names[2]}_pts"

            _pts_handler.append(Draw.Point_cloud(
                viser_server,
                _name, _pts.points, _pts.colors,
                **display_cfg.Config_to_dict()
            ))  # just draw

        return _pts_handler


@dataclass
class Display_Config(Config.Basement):
    cfg_path: InitVar[Path]

    basement: Draw_Cfg.Basement = field(
        default_factory=Draw_Cfg.Basement)
    cam_path: Draw_Cfg.Cam_Path = field(
        default_factory=Draw_Cfg.Cam_Path)
    point_cloud: Draw_Cfg.Point_Cloud = field(
        default_factory=Draw_Cfg.Point_Cloud)

    scene_data: dict[str, tuple[str, Scene_Sequence]] = field(
        default_factory=dict
    )  # target ord, Scene_Sequence

    def __post_init__(self, cfg_path: Path):
        _is_done, _data = Json.Read_from(cfg_path)

        if not _is_done:
            return

        if "visualize" in _data:
            _visualize_cfg: dict[str, dict[str, Any]] = _data["visualize"]

            for _type, _arg in _visualize_cfg.items():
                _cfg_name = "_".join(
                    [_s.capitalize() for _s in _type.split("_")])
                _cfg = getattr(Draw_Cfg, _cfg_name)(**_arg)
                setattr(self, _type, _cfg)

        if "scene_data" in _data:
            _scene_data_cfg: dict[str, dict[str, Any]] = _data["scene_data"]
            for _name, _cfg in _scene_data_cfg.items():
                _format: str = _cfg["data_format"]
                _ord: str = _cfg.pop("ord")
                _type = _format
                if _format == "camera":
                    _cfg["data_format"] = Camera
                    _cfg["load_source"] = False
                    _ord = _name
                elif _format == "point_cloud":
                    _cfg["data_format"] = Point_Cloud
                    _cfg["load_source"] = True
                elif _format == "3d_gaussian":
                    ...
                else:
                    raise ValueError
                self.scene_data[_name] = _ord, Scene_Sequence(**_cfg)

    def Config_to_dict(self) -> dict[str, Any]:
        return {
            "visualize": {
                "cam_path": self.cam_path.Config_to_dict(),
            },
            "scene_data": dict((
                _k, {"ord": _s_sq[0], **_s_sq[1].Get_config()}
            ) for _k, _s_sq in self.scene_data.items())
        }


class Custom_plotter(Server):
    def __init__(
        self,
        interface_cfg: Interface_Config,
        host: str = "127.0.0.1", port: int = 8080,
        parent: QObject | None = None
    ):
        super().__init__(interface_cfg, host, port, parent)

        # {profile_name: (save_dir, scene)}
        self.profiles: dict[str, Display_Config] = {}

        self.del_list: list[str] = []
        self.draw_list: list[str] = []

        self.load_list: list[tuple[str, str]] = []  # name, save_dir
        self.save_list: list[tuple[str, str]] = []

    def Set_event(self):
        # _upload_btn: viser.GuiUploadButtonHandle = self.holder["Upload"]

        # @_upload_btn.on_upload
        # def _(_):
        #     file = _upload_btn.value
        #     print(file.name, len(file.content), "bytes")
        ...

    def Save_profile(self, name: str, save_path: str):
        _draw_cfg = self.profiles[name]
        _file = Path_utils.Make_directory(save_path) / f"{name}.json"

        if _file.suffix != ".json":
            ...

        # Save draw config
        Json.Write_to(_file, _draw_cfg.Config_to_dict())

    def Load_profile(self, name: str, save_path: str):
        _file =  Path(save_path) / f"{name}.json"

        if _file.exists() and _file.is_file():
            self.profiles[name] = (Display_Config(_file))
            self.draw_list.append(name)

    def Draw_profile(self, name: str):
        # get cfg, opt and holder
        _profile = self.profiles[name]
        _frame_term = _profile.basement.display_term

        _display_holder = {}

        # get server
        _s = self.server

        # set origin
        _profile_origin = Draw().Frame(
            _s, name, (0, 0, 0),
            wxyz=_cam_ref, visible=True
        )
        _display_holder["origin"] = _profile_origin

        # draw pose
        for _sq_name, _s_sq in _profile.scene_data.items():
            _ord = _s_sq[0]
            _format, _sq_data = _s_sq[1].data_format, _s_sq[1].sequence_data

            if _format == Camera:
                # set profile origin
                _cfg = _profile.cam_path
                _frame_handler, _cam_handler = Renderer.Cam_path(
                    _s, (name, _sq_name), _sq_data, _frame_term, _cfg)

                _display_holder["frame"] = _frame_handler
                _display_holder[_sq_name] = _cam_handler
                continue

            if _format == Point_Cloud:
                # set profile origin
                _cfg = _profile.point_cloud
                _pts_handler = Renderer.Point_cloud(
                    _s, (name, _ord, _sq_name), _sq_data, _frame_term, _cfg)

                _display_holder[_sq_name] = _pts_handler

        self.display_data[name] = _display_holder

    def Del_profile(self, name: str):
        _display: SceneNodeHandle = self.display_data.pop(name)["origin"]
        _display.remove()

        self.profiles.pop(name)

    def run(self) -> None:
        while self.is_active:
            while self.load_list:
                self.Load_profile(*self.load_list.pop(0))
                time.sleep(0.01)

            while self.save_list:
                self.Save_profile(*self.save_list.pop(0))
                time.sleep(0.01)

            while self.draw_list:
                self.Draw_profile(self.draw_list.pop(0))
                time.sleep(0.01)

            while self.del_list:
                self.Del_profile(self.del_list.pop(0))
                time.sleep(0.01)

            time.sleep(0.05)
