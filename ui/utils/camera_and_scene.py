""" ### Camera and Scene Module
This module provides classes for handling camera parameters, image data,
and scene management.

This implementation follows a `Right-Handed Coordinate System`

------------------------------------------------------------------------
### Requirements
    - NumPy
    - SciPy

### Structure
    `Camera`: Class for handling intrinsic and extrinsic parameters.
    `Scene`: Class for managing image frames and spatial data.

"""

from __future__ import annotations

from typing import Literal, Generic, TypeVar
from dataclasses import dataclass, field, fields, InitVar, asdict
from pathlib import Path

from numpy import (
    ndarray, eye, zeros, uint8, save, load, empty, savez, savez_compressed)
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

import cv2


ROTATION = tuple[Literal["euler", "rovec", "quat"], list[float]]


@dataclass
class Pose():
    """
    Represents a camera with intrinsic and extrinsic parameters

    This implementation follows a `Right-Handed Coordinate System`

    ---------------------------------------------------------------------
    ### Args
    - `rotation`: Quaternion rotation (qw, qx, qy, qz)
    - `transfer`: Translation vector (dx, dy, dz)

    ### Structure
    - `Get_w2c`: Computes the world-to-camera transformation matrix
    - `Get_c2w`: Computes the camera-to-world transformation matrix
    - `Get_intrinsic`: Computes the intrinsic matrix
    - `convert_`: Converts rotation and translation representation

    """
    rotation: R = field(default_factory=lambda: R.from_quat(
        quat=(1, 0, 0, 0),
        scalar_first=True
    ))
    # dx, dy, dz
    transfer: list[float] = field(default_factory=lambda: [0, 0, 0])

    # file_name: InitVar[str]
    # image: ndarray | None = field(init=False)

    def Set_pose_from_vector(
        self,
        rx_type: Literal["euler", "rovec", "quat"], rx: list[float],
        tx: list[float]
    ):
        if rx_type == "quat":
            self.rotation = R.from_quat(tuple(rx[:4]), scalar_first=True)
        elif rx_type == "rovec":
            self.rotation = R.from_rovec(tuple(rx[:3]), degrees=True)
        else:
            self.rotation = R.from_euler("xyz", tuple(rx[:3]), degrees=True)

        self.transfer = tx

    def Get_pose_to_vector(
        self, rx_type: Literal["euler", "rovec", "quat"]
    ) -> tuple[list[float], list[float]]:
        if rx_type == "quat":
            _r_v: list[float] = self.rotation.as_quat(scalar_first=True)
        elif rx_type == "rovec":
            _r_v: list[float] = self.rotation.as_rotvec(degrees=True)
        elif rx_type == "euler":
            _r_v: list[float] = self.rotation.as_euler("xyz", degrees=True)
        else:
            raise ValueError("알 수 없는 회전 형식")

        return _r_v, self.transfer

    def Get_w2c(self):
        """
        Converts the camera's quaternion and translation into a 4x4
        extrinsic transformation matrix

        ------------------------------------------------------------------
        ### Returns
        - `EXTRINSIC`: 4x4 world-to-camera transformation matrix

        """
        _w2c: ndarray = eye(4)
        _w2c[:3, :3] = self.rotation.as_matrix()
        _w2c[:3, 3] = self.transfer
        return _w2c

    def Get_c2w(self):
        """
        Computes the inverse of the world-to-camera transformation matrix

        ------------------------------------------------------------------
        ### Returns
        - `EXTRINSIC`: 4x4 camera-to-world transformation matrix

        """
        return inv(self.Get_w2c())

    # def To_list(self, rx_type: Literal["euler", "rovec", "quat"] = "quat"):
    #     _r_v, _t_v = self.Get_pose_to_vector(rx_type)

    #     return ",".join([f"{_v:>.5f}" for _v in _r_v + _t_v])


@dataclass
class Sensor():
    file_name: str = ""

    def Load(self, path: str | Path) -> bool:
        raise NotImplementedError

    def Save(self, path: str | Path, force: bool = True) -> bool:
        raise NotImplementedError

    def To_line(self) -> str:
        raise NotImplementedError

    def From_line(self, *line: str):
        raise NotImplementedError

    def Get_annotation(self) -> list[str]:
        raise NotImplementedError


@dataclass
class Camera(Sensor, Pose):
    img: ndarray | None = None
    k_values: list[float] = field(
        default_factory=lambda: [1., 1., 180., 120.]
    )  # fx, fy, cx, cy

    def Get_intrinsic(self):
        """
        Constructs the intrinsic 3x3 camera matrix from stored parameters

        ------------------------------------------------------------------
        ### Returns
        - `INTRINSIC`: 3x3 intrinsic matrix

        """
        _in = eye(9)
        _in[[0, 4, 2, 5]] = self.k_values
        return _in.reshape(3, 3)

    def Load(self, path: str | Path):
        _path = path if isinstance(path, Path) else Path(path)
        _file = _path / self.file_name

        if _file.exists():
            match _file.suffix:
                case ".jpg" | ".png" | ".jpeg":
                    self.img = cv2.imread(str(_file))
                    return True
                case ".npy":
                    self.img = load(_file)
                    return True

        _cx, _cy = self.k_values[2:]
        self.img = zeros(
            shape=(round(_cy * 2), round(_cx * 2)), dtype=uint8
        )
        return False

    def Save(self, path: str | Path, force: bool = True):
        _path = path if isinstance(path, Path) else Path(path)
        _name = self.file_name
        _img = self.img

        if _name and (_img is not None):
            _file = _path / _name

            _save_dir = _file.parent
            if _save_dir.exists() or force:
                _save_dir.mkdir(exist_ok=True)

                _ext = _file.suffix if _file.suffix else ""
                match _ext, force:
                    case (".jpg" | ".jpeg" | ".png", _):
                        cv2.imwrite(str(_file), _img)
                        return True
                    case (".npy", _):
                        save(str(_file), _img)
                        return True
                    case(_, True):
                        cv2.imwrite(str(_file.with_suffix(".png")), _img)
                        return True
        return False

    def To_line(self):
        _rx, _tx = self.Get_pose_to_vector("quat")
        _string = ",".join(map(str, self.k_values + _rx + _tx))
        return f"{_string},{self.file_name}"

    def From_line(self, *line: str):
        self.file_name = line[-1]
        _data = [float(_v) for _v in line[:-1]]

        self.k_values = _data[:4]
        self.Set_pose_from_vector("quat", _data[4:8], _data[8:11])

    def Get_annotation(self) -> list[str]:
        return ["# [id, fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz, source]"]


@dataclass
class Point_Cloud(Sensor):
    points: ndarray = field(
        default_factory=lambda: empty((0, 3)))  # x, y, z
    colors: ndarray = field(
        default_factory=lambda: empty((0, 3), dtype=uint8))  # b, g, r

    def __Get_field_list__(self):
        return [_c.name for _c in fields(self) if _c.name != "file_name"]

    def __Decoding_Point_data__(self, path: Path):
        _field_list = self.__Get_field_list__()
        with load(path) as _f:
            for _k, _d in _f.items():
                if _k in _field_list:
                    setattr(self, _k, _d)

    def Load(self, path: str | Path) -> bool:
        _path = path if isinstance(path, Path) else Path(path)
        _file = _path / self.file_name

        if _file.exists():
            match _file.suffix:
                case ".npz":
                    self.__Decoding_Point_data__(_file)
                    return True

        return False

    def __Encoding_Point_data__(
        self, path: str | Path, is_compressed: bool = False
    ):
        _data = asdict(self)
        _data.pop("file_name")
        (savez_compressed if is_compressed else savez)(path, **_data)

    def Save(self, path: str | Path, force: bool = True) -> bool:
        _path = path if isinstance(path, Path) else Path(path)
        _name = self.file_name
        _pts = self.points

        if _name and _pts.shape[0]:
            _file = _path / _name

            _save_dir = _file.parent
            if _save_dir.exists() or force:
                _save_dir.mkdir(exist_ok=True)

                match _file.suffix, force:
                    case (".npz", _):
                        self.__Encoding_Point_data__(_file)
                        return True
        return False

    def To_line(self) -> str:
        # return f"{','.join(self.__Get_field_list__())},{self.file_name}"
        return f"{self.file_name}"

    def From_line(self, *line: str):
        self.file_name = line[-1]

    def Get_annotation(self) -> list[str]:
        return ["# points (tx, ty, tz), colors (b, g, r), file_name"]


SENSOR = TypeVar("SENSOR", bound=Sensor)


@dataclass
class Scene_Sequence(Generic[SENSOR]):
    load_source: InitVar[bool]
    data_format: type[SENSOR]

    meta_file: str = "test.json"
    file_annotation: list[str] = field(default_factory=list)
    source_path: str = str(Path.cwd() / "data")

    sequence_data: dict[int, SENSOR] = field(init=False)

    def __post_init__(self, load_source: bool):
        self.Load_data(load_source)

    def Save_data(self, save_source: bool = True, **kwarg) -> bool:
        _file = Path(self.meta_file)  # meta file
        # check the save directory for meta file
        _file.parent.mkdir(exist_ok=True)

        # check the save source option
        _path = Path(self.source_path)  # source_path
        _save_source = save_source and _path.exists()

        _sq_data = self.sequence_data
        # check tha annotation that set in the meta file
        _annotation = _sq_data[0].Get_annotation()
        self.file_annotation = _annotation

        _meta_data = []

        for _id, _data in _sq_data.items():
            if _save_source:
                _data.Save(_path)
            _meta_data.append(f"{_id},{_data.To_line(**kwarg)}")

        _path.write_text("\n".join(_meta_data), encoding="UTF-8")
        return True

    def Load_data(self, load_source: bool = True) -> bool:
        _file = Path(self.meta_file)  # meta file
        _sq_data = {}

        if not all([_file.exists(), _file.is_file(), _file.suffix == ".txt"]):
            self.sequence_data = _sq_data
            return False

        _path = Path(self.source_path)  # source_path
        _load_source = load_source and _path.exists()

        _st = len(self.file_annotation)
        _data_format = self.data_format

        for _line in _file.read_text(encoding="UTF-8").split("\n")[_st:]:
            _data = _line.split(",")
            _sq = _data_format()
            _sq.From_line(*_data[1:])

            if _load_source:
                _sq.Load(_path)
            _sq_data[int(_data[0])] = _sq

        self.sequence_data = _sq_data
        return True

    def Get_config(self):
        return {
            "data_format": self.data_format.__name__.lower(),
            "meta_file": self.meta_file,
            "file_annotation": self.file_annotation,
            "path": self.source_path,
        }


# @dataclass
# class World_Map():
#     cfg_files: InitVar[dict[str, tuple[str, int, str]]]
#     # {name: (cfg_file, start_line, save_path)}

#     name_to_format: dict[str, type] = field(
#         default_factory=lambda: {"rgb": Camera})

#     id_to_file: dict[int, str] = field(default_factory=dict)
#     file_to_id: dict[str, int] = field(default_factory=dict)

#     save_info: list[str] = field(default_factory=list)

#     # scene: dict[str, dict[int, Sensor]] = field(default_factory=dict)

#     def __post_init__(
#         self,
#         cfg_files: dict[str, tuple[str, int, str]]
#     ):
#         self.save_info = [
#             _n for _n, (
#                 cfg, _st, _path
#             ) in cfg_files.items() if self.Load_data(_n, cfg, _st, _path)
#         ]

#     def Load_data(self, name: str, cfg_file: str, start: int, path: str):
#         _i2f: dict[int, str] = self.id_to_file
#         _f2i: dict[str, int] = self.file_to_id
#         _format: type[Sensor] = self.name_to_format[name]

#         _file = Path(cfg_file)

#         if (not _file.exists()) or _file.suffix != ".txt":
#             return False

#         data: dict[int, Sensor] = {}

#         # read the data list
#         for _line in _file.read_text(encoding="UTF-8").split("\n")[start:]:
#             _comp = _line.split(",")
#             _id, _file = int(_comp[0]), _comp[-1]
#             _sensor: Sensor = _format(_file)
#             _sensor.From_line(_comp[1:-1])

#             data[_id] = _sensor

#             if _id not in _i2f:
#                 _i2f[_id] = _file

#             if _file not in _f2i:
#                 _f2i[_file] = _id

#         setattr(self, name, {"path": path, "frame": data})
#         return True

#     def Save_data(self, save_root: str, is_save_source: bool = False):
#         _cam_txt = [(
#             "# Camera intrinsic & pose"
#             "[id, fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz, source]")]

#         _i_to_n = self.id_to_name
#         _source_dir = self.name_to_format

#         # get data to local property
#         _rgb = self.pose

#         # save data if data exist
#         for _id, _file_name in _i_to_n.items():
#             # if _rgb and :
#             ...


#         _rgb_dir = Path_utils.Make_directory("rgb", save_root)

#         # save data
#         _rgb_sace_dir = Path_utils.Make_directory(
#             _source_dir
#         ) if is_save_source else None
#         _spatial_dir

#         for _id, _cam in self.pose.items():
#             _cam_txt.append(f"{_id}, {_cam.To_list()}, {_i_to_n[_id]}")

#             if _rgb_sace_dir and _cam.image:
#                 cv2.imwrite(str(_rgb_sace_dir / _i_to_n[_id]), _cam.image)

#         _rgb_cam_file = _rgb_dir / "camera.txt"
#         _rgb_cam_file.write_text("\n".join(_cam_txt), encoding="UTF-8")
