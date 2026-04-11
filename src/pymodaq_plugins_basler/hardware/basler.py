import logging
from typing import Any, Callable, List, Optional, Tuple, Union
import platform
import traceback
import threading

from numpy.typing import NDArray
from pypylon import pylon
from qtpy import QtCore, QtWidgets
import json
import os
import sys

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if sys.platform.startswith("win"):
    import ctypes
    MB_YESNO = 0x04
    MB_ICONQUESTION = 0x20
    MB_OK = 0x00
    MB_ICONINFORMATION = 0x40
    MB_ICONERROR = 0x10
    IDYES = 6
    IDNO = 7
    IDOK = 1

    def _win_message_box(title, text, buttons="yesno", icon="info"):
        icon_map = {"info": MB_ICONINFORMATION, "question": MB_ICONQUESTION, "error": MB_ICONERROR}
        flags = icon_map.get(icon, MB_ICONINFORMATION)
        if buttons == "yesno":
            flags |= MB_YESNO
        else:
            flags |= MB_OK
        return ctypes.windll.user32.MessageBoxW(0, text, title, flags)


# Number of pylon internal buffers allocated during normal operation.
_NORMAL_BUFFER_COUNT = 10

# Number of pylon internal buffers allocated during burst mode.
# Increase this if you see "Buffer underrun" errors from pylon at 1 kHz.
_BURST_BUFFER_COUNT = 200


class BaslerCamera:
    """Control a Basler camera in the style of pylablib.

    It wraps an :class:`pylon.InstantCamera` instance.

    :param info: Device info object from :meth:`list_cameras`.
    :param callback: Callback method for each grabbed image.
    """

    tlFactory: pylon.TlFactory
    camera: pylon.InstantCamera

    def __init__(self, info: str, callback: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.camera = pylon.InstantCamera()
        self.model_name = info.GetModelName()
        self.device_info = info
        self._msg_opener = None

        if platform.system() == 'Windows':
            self.base_dir = os.path.join(os.environ.get('PROGRAMDATA'), '.pymodaq')
        else:
            self.base_dir = '/etc/.pymodaq'

        self.default_device_state_path = os.path.join(
            self.base_dir, f'{self.model_name}_config.pfs'
        )

        self.configurationEventHandler = ConfigurationHandler()
        self.camera.RegisterConfiguration(
            self.configurationEventHandler,
            pylon.RegistrationMode_ReplaceAll,
            pylon.Cleanup_None,
        )
        self.imageEventHandler = ImageEventHandler()
        self.camera.RegisterImageEventHandler(
            self.imageEventHandler, pylon.RegistrationMode_Append, pylon.Cleanup_None
        )

        self.imageEventHandler.signals.imageGrabbed.connect(
            lambda x: print("Image grabbed")
        )

        self.attributes = {}
        self.open()
        if callback is not None:
            self.set_callback(callback=callback)

    def open(self) -> None:
        device = self.tlFactory.CreateDevice(self.device_info.GetFullName())
        self.camera.Attach(device)
        try:
            self.camera.Open()
        except Exception as e:
            traceback.print_exc()
            print(f"[BaslerCamera] Failed to open camera: {e}")
            raise
        self.create_default_config_if_not_exists()
        self.get_attributes()
        self.attribute_names = (
            [attr['name'] for attr in self.attributes]
            + [
                child['name']
                for attr in self.attributes
                if attr.get('type') == 'group'
                for child in attr.get('children', [])
            ]
        )

    def set_callback(
        self, callback: Callable[[NDArray], None], replace_all: bool = True
    ) -> None:
        """Setup a callback method for continuous acquisition."""
        if replace_all:
            try:
                self.imageEventHandler.signals.imageGrabbed.disconnect()
            except TypeError:
                pass
        self.imageEventHandler.signals.imageGrabbed.connect(callback)

    @staticmethod
    def list_cameras() -> List[pylon.InstantCamera]:
        """List all available cameras as camera info objects."""
        tlFactory = pylon.TlFactory.GetInstance()
        return tlFactory.EnumerateDevices()

    def get_attributes(self):
        """Get the attributes of the camera and store them in a dictionary."""
        name = self.model_name.replace(" ", "-")
        file_path = os.path.join(self.base_dir, f'config_{name}.json')
        try:
            with open(file_path, 'r') as file:
                attributes = json.load(file)
                self.attributes = self.clean_device_attributes(attributes)
        except Exception as e:
            logger.error(f"The config file was not found at {file_path}: {e}")

    def get_roi(self) -> Tuple[float, float, float, float, int, int]:
        """Return x0, x0+width, y0, y0+height, xbin, ybin."""
        x0 = self.camera.OffsetX.GetValue()
        width = self.camera.Width.GetValue()
        y0 = self.camera.OffsetY.GetValue()
        height = self.camera.Height.GetValue()
        xbin = self.camera.BinningHorizontal.GetValue()
        ybin = self.camera.BinningVertical.GetValue()
        return x0, x0 + width, y0, y0 + height, xbin, ybin

    def set_roi(
        self, hstart: int, hend: int, vstart: int, vend: int, hbin: int, vbin: int
    ) -> None:
        camera = self.camera
        m_width, m_height = self.get_detector_size()
        inc = camera.Width.Inc
        hstart = detector_clamp(hstart, m_width) // inc * inc
        vstart = detector_clamp(vstart, m_height) // inc * inc
        camera.OffsetX.SetValue(0)
        camera.Width.SetValue((detector_clamp(hend, m_width) - hstart) // inc * inc)
        camera.OffsetX.SetValue(hstart)
        camera.OffsetY.SetValue(0)
        camera.Height.SetValue((detector_clamp(vend, m_height) - vstart) // inc * inc)
        camera.OffsetY.SetValue(vstart)
        camera.BinningHorizontal.SetValue(int(hbin))
        camera.BinningVertical.SetValue(int(vbin))

    def get_detector_size(self) -> Tuple[int, int]:
        """Return width and height of detector in pixels."""
        return self.camera.SensorWidth.GetValue(), self.camera.SensorHeight.GetValue()

    def get_attribute_value(self, name, error_on_missing=True):
        """Get the camera attribute with the given name."""
        return self.attributes[name]

    def setup_acquisition(self):
        try:
            self.camera.TriggerSelector.SetValue("AcquisitionStart")
            self.camera.TriggerMode.SetValue("Off")
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("Off")
            self.camera.AcquisitionFrameRateEnable.SetValue(False)
            self.camera.AcquisitionMode.SetValue("Continuous")
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
        except Exception as e:
            logger.error(f"Could not properly setup acquisition for live grabbing: {e}")

    def close(self) -> None:
        self.camera.Close()
        self.camera.DetachDevice()

    def start_grabbing(self, frame_rate: Optional[int], burst_mode: bool = False) -> None:
        """Start continuously grabbing frames.

        Parameters
        ----------
        frame_rate:
            Target frame rate.  Passed to ``AcquisitionFrameRate``.
        burst_mode:
            When True, switches to ``GrabStrategy_OneByOne`` and raises the
            pylon internal buffer count so that no frames are dropped at the
            pylon layer.  Use this for high-speed (≥ 1 kHz) recording.
            When False (default), uses ``GrabStrategy_LatestImageOnly`` which
            is fine for live display but silently discards frames under load.
        """
        if frame_rate is not None:
            try:
                self.camera.AcquisitionFrameRate.SetValue(frame_rate)
            except pylon.LogicalErrorException:
                pass

        if burst_mode:
            # Raise pylon's internal ring buffer so the camera keeps
            # delivering even if the writer thread stalls briefly.
            try:
                self.camera.MaxNumBuffer.SetValue(_BURST_BUFFER_COUNT)
            except Exception:
                pass  # not all models expose this node
            self.camera.StartGrabbing(
                pylon.GrabStrategy_OneByOne,
                pylon.GrabLoop_ProvidedByInstantCamera,
            )
        else:
            try:
                self.camera.MaxNumBuffer.SetValue(_NORMAL_BUFFER_COUNT)
            except Exception:
                pass
            self.camera.StartGrabbing(
                pylon.GrabStrategy_LatestImageOnly,
                pylon.GrabLoop_ProvidedByInstantCamera,
            )

    def stop_grabbing(self):
        self.camera.StopGrabbing()
        return ''

    def save_device_state(self):
        save_path = self.default_device_state_path
        node_map = self.camera.GetNodeMap()
        try:
            pylon.FeaturePersistence.Save(save_path, node_map)
            print(f"Device state saved to {save_path}")
        except Exception as e:
            print(f"Failed to save device state: {e}")

    def load_device_state(self):
        load_path = self.default_device_state_path
        node_map = self.camera.GetNodeMap()
        if os.path.isfile(load_path):
            try:
                pylon.FeaturePersistence.Load(load_path, node_map)
                print(f"Device state loaded")
            except Exception as e:
                print(f"Failed to load device state: {e}")
        else:
            print("No saved settings file found to load.")

    def clean_device_attributes(self, attributes):
        clean_params = []
        if isinstance(attributes, dict):
            items = attributes.items()
        elif isinstance(attributes, list):
            items = enumerate(attributes)
        else:
            raise ValueError(f"Unsupported type for attributes: {type(attributes)}")

        for idx, attr in items:
            param = {}
            param['title'] = attr.get('title', '')
            param['name'] = attr.get('name', str(idx))
            param['type'] = attr.get('type', 'str')
            param['value'] = attr.get('value', '')
            param['default'] = attr.get('default', None)
            param['limits'] = attr.get('limits', None)
            param['readonly'] = attr.get('readonly', False)

            if param['type'] == 'group' and 'children' in attr:
                children = attr['children']
                if isinstance(children, dict):
                    children = list(children.values())
                param['children'] = self.clean_device_attributes(children)

            clean_params.append(param)
        return clean_params

    def check_attribute_names(self):
        found_exposure = None
        found_gain = None

        for exp in ["ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"]:
            try:
                if hasattr(self.camera, exp):
                    found_exposure = exp
                    break
            except pylon.LogicalErrorException:
                pass

        raw_gain = False
        for gain in ["Gain", "GainRaw", "GainAll"]:
            try:
                if hasattr(self.camera, gain):
                    found_gain = gain
                    if gain == "GainRaw":
                        raw_gain = True
                    break
            except pylon.LogicalErrorException:
                pass

        return found_exposure or "ExposureTime", found_gain or "Gain", raw_gain

    def create_default_config_if_not_exists(self):
        model_name = self.model_name.replace(" ", "-")
        config_dir = self.base_dir
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f'config_{model_name}.json')
        if os.path.exists(config_path):
            return

        self._msg_opener = DefaultConfigMsg()
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setWindowTitle("Missing Config File")
        msg.setText(f"No config file found for camera model '{model_name}'.")
        msg.setInformativeText("Would you like to auto-create a default configuration file?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
        QtCore.QTimer.singleShot(0, QtWidgets.QApplication.processEvents)
        user_choice = self.safe_exec_messagebox(msg)
        self.handle_user_choice(user_choice, config_path, model_name)

    def handle_user_choice(self, user_choice, config_path, model_name):
        if user_choice == QtWidgets.QMessageBox.Yes:
            found_exposure, found_gain, raw_gain = self.check_attribute_names()
            config_data = {
                "exposure": {
                    "title": "Exposure Settings", "name": "exposure", "type": "group",
                    "children": {
                        "Exposure Auto": {"title": "Exposure Auto", "name": "ExposureAuto",
                                          "type": "led_push", "value": False, "default": False},
                        "Exposure Time": {"title": "Exposure Time (ms)", "name": found_exposure,
                                          "type": "slide", "value": 100.0, "default": 100.0,
                                          "limits": [0.001, 10000.0]},
                    },
                },
                "gain": {
                    "title": "Gain Settings", "name": "gain", "type": "group",
                    "children": {
                        "Gain Auto": {"title": "Gain Auto", "name": "GainAuto",
                                      "type": "led_push", "value": False, "default": False},
                        "Gain": {"title": "Gain Value", "name": found_gain, "type": "slide",
                                 "value": 1.0, "default": 1.0, "limits": [0.0, 2.0],
                                 "int": raw_gain},
                    },
                },
            }
            try:
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=4)
                msg_info = QtWidgets.QMessageBox()
                msg_info.setIcon(QtWidgets.QMessageBox.Information)
                msg_info.setWindowTitle("Config Created")
                msg_info.setText(f"Default config file created for '{model_name}'.")
                msg_info.setInformativeText(
                    f"Path:\n{config_path}\n\nYou can edit this file to add/remove parameters."
                )
                self.safe_exec_messagebox(msg_info, buttons="ok")
            except Exception as e:
                msg_err = QtWidgets.QMessageBox()
                msg_err.setIcon(QtWidgets.QMessageBox.Critical)
                msg_err.setWindowTitle("Error Creating Config")
                msg_err.setText(f"Failed to write default config file:\n{e}")
                self.safe_exec_messagebox(msg_err, buttons="ok")
        else:
            msg_info = QtWidgets.QMessageBox()
            msg_info.setIcon(QtWidgets.QMessageBox.Information)
            msg_info.setWindowTitle("Config Not Created")
            msg_info.setText(
                f"You have chosen not to create a default config file for Basler '{model_name}'."
            )
            msg_info.setInformativeText(
                "You will not have access to camera parameters until you have a valid config file."
            )
            self.safe_exec_messagebox(msg_info, buttons="ok")

    def safe_exec_messagebox(self, msgbox: QtWidgets.QMessageBox, buttons: str = "yesno") -> int:
        result_container = {}
        finished_event = threading.Event()

        def show_dialog():
            try:
                result_container["choice"] = int(msgbox.exec_())
            except Exception:
                result_container["choice"] = int(QtWidgets.QMessageBox.No)
            finally:
                finished_event.set()

        if self._msg_opener is None:
            self._msg_opener = DefaultConfigMsg()

        if sys.platform.startswith("win"):
            title = str(msgbox.windowTitle() or "PyMoDAQ")
            text = str(msgbox.text() or "")
            informative = msgbox.informativeText()
            if informative:
                text += "\n\n" + str(informative)
            try:
                icon_type = "info"
                if msgbox.icon() == QtWidgets.QMessageBox.Question:
                    icon_type = "question"
                elif msgbox.icon() == QtWidgets.QMessageBox.Critical:
                    icon_type = "error"
                res = _win_message_box(title, text, buttons=buttons, icon=icon_type)
                if buttons == "yesno":
                    return int(QtWidgets.QMessageBox.Yes) if res == IDYES else int(QtWidgets.QMessageBox.No)
                return int(QtWidgets.QMessageBox.Ok)
            except Exception:
                return int(QtWidgets.QMessageBox.No)
        else:
            QtCore.QMetaObject.invokeMethod(
                self._msg_opener, "run_box",
                QtCore.Qt.ConnectionType.AutoConnection,
                QtCore.Q_ARG(object, show_dialog),
            )
            if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
                finished_event.wait()
                QtCore.QTimer.singleShot(0, QtWidgets.QApplication.processEvents)
            else:
                while not finished_event.is_set():
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            return result_container.get("choice", int(QtWidgets.QMessageBox.No))


class DefaultConfigMsg(QtCore.QObject):
    def __init__(self):
        super().__init__()

    @QtCore.Slot(object)
    def run_box(self, fn):
        fn()


class ConfigurationHandler(pylon.ConfigurationEventHandler):
    """Handle the configuration events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ConfigurationHandlerSignals()

    class ConfigurationHandlerSignals(QtCore.QObject):
        cameraRemoved = QtCore.pyqtSignal(object)

    def OnOpened(self, camera: pylon.InstantCamera) -> None:
        try:
            camera.PixelFormat.SetValue("Mono12")
        except Exception:
            pass
        camera.GainAuto.SetValue("Off")
        camera.ExposureAuto.SetValue("Off")

    def OnCameraDeviceRemoved(self, camera: pylon.InstantCamera) -> None:
        self.signals.cameraRemoved.emit(camera)


class ImageEventHandler(pylon.ImageEventHandler):
    """Handle grab events and translate them to Qt signals."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals = self.ImageEventHandlerSignals()
        self.frame_ready = False

    class ImageEventHandlerSignals(QtCore.QObject):
        imageGrabbed = QtCore.pyqtSignal(object)

    def OnImageSkipped(self, camera: pylon.InstantCamera, countOfSkippedImages: int) -> None:
        logger.warning(f"{countOfSkippedImages} images have been skipped.")

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grabResult: pylon.GrabResult) -> None:
        if grabResult.GrabSucceeded():
            self.frame_ready = True
            frame_data = {
                "frame": grabResult.GetArray(),
                "timestamp": grabResult.GetTimeStamp(),
            }
            self.signals.imageGrabbed.emit(frame_data)
        else:
            logger.warning(
                f"Grab failed with code {grabResult.GetErrorCode()}, "
                f"{grabResult.GetErrorDescription()}."
            )


class TemperatureMonitor(QtCore.QObject):
    temperature_updated = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self, camera_handle, check_interval=5000):
        super().__init__()
        self._running = True
        self.camera = camera_handle
        self.interval = check_interval

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                temp = self.camera.TemperatureAbs.Value
                self.temperature_updated.emit(temp)
            except Exception:
                pass
            QtCore.QThread.msleep(self.interval)
        self.finished.emit()


def detector_clamp(value: Union[float, int], max_value: int) -> int:
    """Clamp a value to possible detector position."""
    return max(0, min(int(value), max_value))