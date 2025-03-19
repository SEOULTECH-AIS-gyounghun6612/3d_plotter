from pathlib import Path
import gui_toolbox

from ui.main_page import Test_App

main_app = Test_App(
    gui_toolbox.App_Config,
    Path(".\\config\\app.json")
)

main_app.Watcher()
