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

# from typing import Any
from dataclasses import dataclass, field

from python_ex.project import Config

from gui_toolbox.window import Popup_Page


@dataclass
class Data_Editor_Config(Config.Basement):
    root_path: str
    annotation: str = ""


class Data_Editor(Popup_Page):
    def _Set_data(self, **kwarg):
        ...