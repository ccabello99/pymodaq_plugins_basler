import numpy as np
import os
import imageio as iio
import h5py
import json
from datetime import datetime
from uuid6 import uuid7
from typing import Optional

from pymodaq.utils.parameter import Parameter
from pymodaq.utils.data import Axis, DataFromPlugins, DataToExport
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.control_modules.viewer_utility_classes import main, DAQ_Viewer_base, comon_parameters, params

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from pymodaq_plugins_basler.hardware.basler import BaslerCamera, TemperatureMonitor
from pymodaq_plugins_basler.hardware.burst_writer import BurstWriter
from pymodaq_plugins_basler.resources.extended_publisher import ExtendedPublisher
from qtpy import QtWidgets, QtCore

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore


class DAQ_2DViewer_BaslerWithLECO(DAQ_Viewer_base):
    """Viewer for Basler cameras with LECO integration and burst-mode HDF5 recording."""

    controller: BaslerCamera
    live_mode_available = True

    camera_list = [cam.GetFriendlyName() for cam in BaslerCamera.list_cameras()]
    settings_basler = QtCore.QSettings("PyMoDAQ", "Basler")

    params = comon_parameters + [
        {'title': 'Camera List:', 'name': 'camera_list', 'type': 'list',
         'value': '', 'limits': camera_list},

        {"title": "Device Info", "name": "device_info", "type": "group", "children": [
            {"title": "Device Model Name", "name": "DeviceModelName",
             "type": "str", "value": "", "readonly": True},
            {"title": "Device Serial Number", "name": "DeviceSerialNumber",
             "type": "str", "value": "", "readonly": True},
            {"title": "Device Version", "name": "DeviceVersion",
             "type": "str", "value": "", "readonly": True},
            {"title": "Device User ID", "name": "DeviceUserID",
             "type": "str", "value": ""},
        ]},

        {'title': 'ROI', 'name': 'roi', 'type': 'group', 'children': [
            {'title': 'Update ROI', 'name': 'update_roi',
             'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Clear ROI+Bin', 'name': 'clear_roi',
             'type': 'bool_push', 'value': False, 'default': False},
            {'title': 'Binning', 'name': 'binning', 'type': 'list',
             'limits': [1, 2], 'default': 1},
            {'title': 'Image Width', 'name': 'width',
             'type': 'int', 'value': 1280, 'readonly': True},
            {'title': 'Image Height', 'name': 'height',
             'type': 'int', 'value': 960, 'readonly': True},
        ]},

        {'title': 'Burst Recording', 'name': 'burst', 'type': 'group', 'children': [
            {'title': 'Enable Burst Mode', 'name': 'burst_enable',
             'type': 'led_push', 'value': False, 'default': False,
             'tip': 'Arms burst mode. Triggers camera and waits for Line1 '
                    '(FrameStart) then Line4 (FrameStart) hardware triggers.'},
            {'title': 'Save Path', 'name': 'burst_path',
             'type': 'browsepath', 'value': '', 'filetype': False},
            {'title': 'Filename Prefix', 'name': 'burst_prefix',
             'type': 'str', 'value': 'burst',
             'tip': 'File will be named <prefix>_YYYYMMDD_HHMMSS.h5'},
            {'title': 'Stop Condition', 'name': 'burst_stop_group',
             'type': 'group', 'children': [
                 {'title': 'Max Frames (0 = time-bounded)', 'name': 'burst_nframes',
                  'type': 'int', 'value': 1000, 'min': 0},
                 {'title': 'Max Seconds (0 = frame-bounded)', 'name': 'burst_nseconds',
                  'type': 'float', 'value': 0.0, 'min': 0.0},
             ]},
            {'title': 'Performance', 'name': 'burst_perf_group',
             'type': 'group', 'children': [
                 {'title': 'Display Every Nth Frame', 'name': 'burst_display_nth',
                  'type': 'int', 'value': 200, 'min': 1},
                 {'title': 'Write Chunk Size', 'name': 'burst_chunk',
                  'type': 'int', 'value': 20, 'min': 1},
                 {'title': 'Queue Max Size', 'name': 'burst_queue_size',
                  'type': 'int', 'value': 500, 'min': 10},
                 {'title': 'Overflow Policy', 'name': 'burst_overflow',
                  'type': 'list',
                  'limits': ['drop_newest', 'drop_oldest'],
                  'value': 'drop_newest'},
                 {'title': 'Trigger Timeout (s)', 'name': 'burst_trigger_timeout',
                  'type': 'float', 'value': 30.0, 'min': 1.0,
                  'tip': 'How long to wait for the first hardware trigger '
                         'before aborting the burst.'},
             ]},
            # Live readouts
            {'title': 'Status', 'name': 'burst_status',
             'type': 'str', 'value': 'Idle', 'readonly': True},
            {'title': 'Frames Written', 'name': 'burst_written',
             'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Frames Dropped', 'name': 'burst_dropped',
             'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Elapsed (s)', 'name': 'burst_elapsed',
             'type': 'float', 'value': 0.0, 'readonly': True},
        ]},

        {'title': 'LECO Logging', 'name': 'leco_log', 'type': 'group', 'children': [
            {'title': 'Send Frame Data?', 'name': 'leco_send',
             'type': 'led_push', 'value': False, 'default': False,
             'tip': 'Huge performance drop. Only use for single grabs, not continuous.'},
            {'title': 'Publisher Name', 'name': 'publisher_name', 'type': 'str', 'value': ''},
            {'title': 'Proxy Server Address', 'name': 'proxy_address',
             'type': 'str', 'value': 'localhost', 'default': 'localhost'},
            {'title': 'Proxy Server Port', 'name': 'proxy_port',
             'type': 'int', 'value': 11100, 'default': 11100},
            {'title': 'Metadata', 'name': 'leco_metadata',
             'type': 'str', 'value': '', 'readonly': True},
            {'title': 'Saving Base Path:', 'name': 'leco_basepath',
             'type': 'browsepath', 'value': '', 'filetype': False},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Optional[BaslerCamera] = None
        self.user_id = None
        self.data_shape = None
        self.save_frame = False
        self.metadata = None
        self.data_publisher = None
        self.send_frame_leco = False
        self._burst_active = False
        self._burst_writer: Optional[BurstWriter] = None
        self._burst_thread: Optional[QtCore.QThread] = None
        self._burst_h5_path: Optional[str] = None

    def init_controller(self) -> BaslerCamera:
        self.user_id = self.settings.param('camera_list').value()
        self.emit_status(ThreadCommand('Update_Status',
                                       [f"Trying to connect to {self.user_id}", 'log']))
        for devInfo in BaslerCamera.list_cameras():
            if devInfo.GetFriendlyName() == self.user_id:
                return BaslerCamera(info=devInfo, callback=self.emit_data_callback)
        self.emit_status(ThreadCommand('Update_Status', ["Camera not found", 'log']))
        raise ValueError(f"Camera with name {self.user_id} not found anymore.")

    def ini_detector(self, controller=None):
        self.ini_detector_init(old_controller=controller,
                               new_controller=self.init_controller())
        self.controller.setup_acquisition()
        self.controller.configurationEventHandler.signals.cameraRemoved.connect(
            self.camera_lost
        )

        self.add_attributes_to_settings()
        self.update_params_ui()
        for param in self.settings.children():
            if param.name() == 'device_info':
                continue
            param.sigValueChanged.emit(param, param.value())
            if param.hasChildren():
                for child in param.children():
                    child.sigValueChanged.emit(child, child.value())

        (x0, xend, y0, yend, xbin, ybin) = self.controller.get_roi()
        self.settings.child('roi', 'binning').setValue(xbin)
        self.settings.child('roi', 'width').setValue(yend - y0)
        self.settings.child('roi', 'height').setValue(xend - x0)

        publisher_name = self.settings.child('leco_log', 'publisher_name').value()
        proxy_address = self.settings.child('leco_log', 'proxy_address').value()
        proxy_port = self.settings.child('leco_log', 'proxy_port').value()
        if publisher_name:
            self.data_publisher = ExtendedPublisher(
                full_name=publisher_name, host=proxy_address, port=proxy_port
            )
            self.emit_status(ThreadCommand('Update_Status',
                                           [f"Data publisher {publisher_name} initialised"]))
        else:
            self.emit_status(ThreadCommand('Update_Status',
                                           ["Publisher name not set – LECO disabled"]))

        try:
            base_path = self.settings_basler.value(
                'leco_log/basepath', os.path.join(os.path.expanduser('~'), 'Downloads')
            )
        except Exception:
            base_path = ''
        self.settings.child('leco_log', 'leco_basepath').setValue(base_path)

        if not self.settings.child('burst', 'burst_path').value():
            self.settings.child('burst', 'burst_path').setValue(
                os.path.join(os.path.expanduser('~'), 'Downloads')
            )

        self._prepare_view()
        self.emit_status(ThreadCommand('Update_Status',
                                       [f"{self.user_id} initialised successfully"]))
        return "Initialized camera", True

    def commit_settings(self, param: Parameter):
        name = param.name()
        value = param.value()

        if name == "camera_list":
            if self.controller is not None:
                self.close()
            self.ini_detector()
            return

        if name == "device_state_save":
            self.controller.save_device_state()
            param.setValue(False)
            param.sigValueChanged.emit(param, False)
            return

        if name == "device_state_load":
            self.stop()
            self.controller.load_device_state()
            self.controller.setup_acquisition()
            self.add_attributes_to_settings()
            self.update_params_ui()
            for p in self.settings.children():
                p.sigValueChanged.emit(p, p.value())
                if p.hasChildren():
                    for child in p.children():
                        child.sigValueChanged.emit(child, child.value())
            self._prepare_view()
            self.grab_data()
            self.emit_status(ThreadCommand('Update_Status', ["Device state loaded"]))
            return

        if name == 'PixelFormat':
            self.stop()
            self.controller.camera.PixelFormat.SetValue(value)
            self._prepare_view()
            self.grab_data()
            return

        if name == 'TriggerSave':
            if not self.settings.child('trigger', 'TriggerMode').value():
                self.emit_status(ThreadCommand('Update_Status',
                                               ["Trigger mode is not active!"]))
                p = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                p.setValue(False)
                p.sigValueChanged.emit(p, False)
                return
            self.save_frame = bool(value)
            return

        if name == 'leco_send':
            self.send_frame_leco = bool(value)
            return
        if name == 'leco_basepath':
            if os.path.exists(value):
                self.settings_basler.setValue('leco_log/basepath', value)
            return
        if name == 'leco_metadata':
            try:
                self.metadata = json.loads(value)
            except Exception:
                self.metadata = None
            return

        if name == 'burst_enable':
            if value:
                try:
                    frame_rate = self.settings.param('AcquisitionFrameRateAbs').value()
                except Exception:
                    try:
                        frame_rate = self.settings.param('AcquisitionFrameRate').value()
                    except Exception:
                        frame_rate = None
                self._start_burst(frame_rate)
            else:
                if not self._burst_active:
                    return
                self._stop_burst()
            return

        if name in self.controller.attribute_names:
            if 'ExposureTime' in name:
                value = int(value * 1e3)
            if 'Gain' in name and 'Auto' not in name:
                value = int(value)
            if name == "DeviceUserID":
                self.user_id = value
                self.controller.camera.DeviceUserID.SetValue(value)
                camera_list = [cam.GetFriendlyName() for cam in BaslerCamera.list_cameras()]
                p = self.settings.param('camera_list')
                p.setLimits(camera_list)
                p.sigLimitsChanged.emit(p, camera_list)
                return
            if name == 'TriggerMode':
                camera_attr = getattr(self.controller.camera, name)
                camera_attr.SetIntValue(1 if value else 0)
                if not value:
                    self.save_frame = False
                    p = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
                    p.setValue(False)
                    p.sigValueChanged.emit(p, False)
                return
            for auto_name in ('GainAuto', 'ExposureAuto'):
                if name == auto_name:
                    getattr(self.controller.camera, name).SetIntValue(1 if value else 0)
                    return
            for skip_name in ('TriggerSaveLocation', 'TriggerSaveIndex',
                              'Filetype', 'Prefix', 'TemperatureMonitor'):
                if name == skip_name:
                    if name == 'TemperatureMonitor':
                        if value:
                            self.start_temperature_monitoring()
                        else:
                            self.stop_temp_monitoring()
                    return
            camera_attr = getattr(self.controller.camera, name)
            camera_attr.SetValue(value)
            return

        if name == "update_roi" and value:
            self.stop()
            (old_x, _, old_y, _, xbin, ybin) = self.controller.get_roi()
            y0, x0 = self.roi_info.origin.coordinates
            height, width = self.roi_info.size.coordinates
            new_roi = (
                (old_x + x0) * xbin, width * ybin, xbin,
                (old_y + y0) * xbin, height * ybin, ybin,
            )
            self.update_rois(new_roi)
            param.setValue(False)
            param.sigValueChanged.emit(param, False)
            self.grab_data()
        elif name == 'binning':
            (x0, w, y0, h, *_) = self.controller.get_roi()
            b = value
            self.update_rois((x0, w, b, y0, h, b))
        elif name == "clear_roi" and value:
            self.stop()
            wdet, hdet = self.controller.get_detector_size()
            self.settings.child('roi', 'binning').setValue(1)
            self.update_rois((0, wdet, 1, 0, hdet, 1))
            param.setValue(False)
            param.sigValueChanged.emit(param, False)
            self.grab_data()

    def grab_data(self, Naverage: int = 1, live: bool = False, **kwargs) -> None:
        try:
            self._prepare_view()
            try:
                frame_rate = self.settings.param('AcquisitionFrameRateAbs').value()
            except Exception:
                try:
                    frame_rate = self.settings.param('AcquisitionFrameRate').value()
                except Exception:
                    frame_rate = None

            if live:
                self.controller.start_grabbing(frame_rate, burst_mode=False)
            else:
                self.controller.start_grabbing(frame_rate, burst_mode=False)
                while not self.controller.imageEventHandler.frame_ready:
                    pass
                self.controller.stop_grabbing()

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))

    def emit_data_callback(self, frame_data: dict) -> None:
        """Called from pylon's grab thread for every frame."""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']

        if self._burst_active:
            self._burst_writer.enqueue(frame.copy(), timestamp)
            return

        dte = DataToExport(
            f'{self.user_id}',
            data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                axes=self.axes,
            )],
        )
        self.dte_signal.emit(dte)

        if self.save_frame:
            self.handle_metadata_and_saving(frame, timestamp, frame.shape)
            self.metadata = None

        self.controller.imageEventHandler.frame_ready = False

    def stop(self):
        if self._burst_active:
            self._stop_burst()
        else:
            self.controller.camera.StopGrabbing()
        return ''

    def close(self):
        if self._burst_active:
            self._stop_burst(wait=True)

        self.controller.attributes = None
        self.controller.close()

        try:
            self.stop_temp_monitoring()
        except Exception:
            pass

        try:
            p = self.settings.child('trigger', 'TriggerMode')
            p.setValue(False)
            p.sigValueChanged.emit(p, False)
            p = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
            p.setValue(False)
            p.sigValueChanged.emit(p, False)
        except Exception:
            pass

        self.status.initialized = False
        self.status.controller = None
        self.status.info = ""
        self.emit_status(ThreadCommand('Update_Status',
                                       [f"{self.user_id} communication terminated"]))

    def _start_burst(self, frame_rate):
        """Construct the BurstWriter, wire up signals, and start grabbing."""
        if self._burst_active:
            self.emit_status(ThreadCommand('Update_Status',
                                           ["Burst already in progress – ignoring."]))
            return

        self.stop()

        # Make sure trigger save is off 
        self.save_frame = False
        p = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSave')
        p.setValue(False)
        p.sigValueChanged.emit(p, False)

        max_frames = self.settings.child('burst', 'burst_stop_group', 'burst_nframes').value()
        max_seconds = self.settings.child('burst', 'burst_stop_group', 'burst_nseconds').value()
        if max_frames == 0 and max_seconds == 0.0:
            self.emit_status(ThreadCommand('Update_Status',
                                           ["Burst: set Max Frames or Max Seconds first."]))
            return
        max_frames = max_frames if max_frames > 0 else None
        max_seconds = max_seconds if max_seconds > 0.0 else None

        if self.metadata is not None:
            filepath = self.metadata['file_metadata']['filepath']
            filename = self.metadata['file_metadata']['filename']
            self.metadata['burst_metadata']['user_id'] = self.user_id
            basepath = self.settings.child('leco_log', 'leco_basepath').value()
            prefix = self.settings.child('burst', 'burst_prefix').value() or 'burst'
            filepath = os.path.normpath(
                os.path.join(basepath, f"{prefix}_{filepath.lstrip(os.path.sep)}")
            )
            fname = filename if filename.endswith('.h5') else filename + '.h5'
            full_path = os.path.join(filepath, fname)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)            
            self._burst_h5_path = full_path
        else:
            save_dir = self.settings.child('burst', 'burst_path').value()
            if not save_dir:
                save_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
            prefix = self.settings.child('burst', 'burst_prefix').value() or 'burst'
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._burst_h5_path = os.path.join(save_dir, f"{prefix}_{timestamp_str}.h5")

        actual_width = self.controller.camera.Width.GetValue()
        actual_height = self.controller.camera.Height.GetValue()
        (hstart, hend, vstart, vend, xbin, ybin) = self.controller.get_roi()

        exposure_ms = 0.0
        gain_val = 0.0
        for attr_name in self.controller.attribute_names:
            if 'Exposure' in attr_name and 'Auto' not in attr_name:
                try:
                    exposure_ms = self.settings.child('exposure', attr_name).value()
                except Exception:
                    pass
            if 'Gain' in attr_name and 'Auto' not in attr_name:
                try:
                    gain_val = self.settings.child('gain', attr_name).value()
                except Exception:
                    pass

        camera_meta = {
            "camera_model": self.controller.model_name,
            "serial_number": self.controller.device_info.GetSerialNumber(),
            "exposure_time_ms": exposure_ms,
            "gain": gain_val,
            "roi": [hstart, vstart, actual_width, actual_height],
            "fps_target": frame_rate or 1000,
        }

        if self.metadata is not None:
            burst_meta = self.metadata.get('burst_metadata', {})
            detector_meta = self.metadata.get('detector_metadata', {})
            camera_meta['sequence_uuid'] = burst_meta.get('sequence_uuid', str(uuid7()))
            camera_meta['fuzziness'] = detector_meta.get('fuzziness', 0.1)
            camera_meta['conduktor_metadata'] = self.metadata
        else:
            camera_meta['sequence_uuid'] = str(uuid7())
            camera_meta['fuzziness'] = 0.1
            camera_meta['conduktor_metadata'] = {}

        display_nth = self.settings.child('burst', 'burst_perf_group', 'burst_display_nth').value()
        chunk_size = self.settings.child('burst', 'burst_perf_group', 'burst_chunk').value()
        queue_size = self.settings.child('burst', 'burst_perf_group', 'burst_queue_size').value()
        overflow = self.settings.child('burst', 'burst_perf_group', 'burst_overflow').value()
        trigger_timeout = self.settings.child(
            'burst', 'burst_perf_group', 'burst_trigger_timeout'
        ).value()
        drop_oldest = overflow == 'drop_oldest'

        try:
            pf = self.controller.camera.PixelFormat.GetValue()
            dtype = np.uint16 if '12' in pf or '16' in pf else np.uint8
        except Exception:
            dtype = np.uint16

        self._burst_writer = BurstWriter(
            h5_path=self._burst_h5_path,
            frame_shape=(actual_height, actual_width),  # (H, W) numpy convention
            dtype=dtype,
            max_frames=max_frames,
            max_seconds=max_seconds,
            display_every_nth=display_nth,
            chunk_size=chunk_size,
            queue_maxsize=queue_size,
            drop_oldest=drop_oldest,
            camera_meta=camera_meta,
            first_frame_timeout=trigger_timeout,
        )

        self._burst_writer.signals.first_frame_received.connect(self._on_burst_first_frame)
        self._burst_writer.signals.display_frame.connect(self._on_burst_display_frame)
        self._burst_writer.signals.progress.connect(self._on_burst_progress)
        self._burst_writer.signals.burst_finished.connect(self._on_burst_finished)
        self._burst_writer.signals.error.connect(self._on_burst_error)

        self._burst_thread = QtCore.QThread()
        self._burst_writer.moveToThread(self._burst_thread)
        self._burst_thread.started.connect(self._burst_writer.run)
        self._burst_thread.finished.connect(self._burst_thread.deleteLater)

        self._set_burst_status(f"Armed – waiting for trigger on Line1… (timeout {trigger_timeout:.0f}s)")
        self._set_burst_readouts(0, 0, 0.0)

        self._burst_active = True
        self._burst_thread.start()

        self.controller.start_grabbing(frame_rate, burst_mode=True)

        self.emit_status(ThreadCommand('Update_Status',
                                       [f"Burst armed : {self._burst_h5_path}"]))

    def _stop_burst(self, wait: bool = False):
        """Request the writer to stop and clean up."""
        if not self._burst_active:
            return

        try:
            self.controller.stop_grabbing()
        except Exception:
            pass

        if self._burst_writer is not None:
            self._burst_writer.request_stop()

        if wait and self._burst_thread is not None:
            self._burst_thread.quit()
            self._burst_thread.wait(5000)

        self._burst_active = False

    @QtCore.Slot()
    def _on_burst_first_frame(self):
        """First hardware trigger received — update status to show recording is live."""
        self._set_burst_status("Recording…")
        self.emit_status(ThreadCommand('Update_Status', ["Burst: first frame received, recording."]))

    @QtCore.Slot(object)
    def _on_burst_display_frame(self, frame: np.ndarray):
        dte = DataToExport(
            f'{self.user_id}',
            data=[DataFromPlugins(
                name=f'{self.user_id}',
                data=[np.squeeze(frame)],
                dim=self.data_shape,
                labels=[f'{self.user_id}_{self.data_shape}'],
                axes=self.axes,
            )],
        )
        self.dte_signal.emit(dte)

    @QtCore.Slot(int, int, float)
    def _on_burst_progress(self, written: int, dropped: int, elapsed: float):
        self._set_burst_readouts(written, dropped, elapsed)

    @QtCore.Slot(dict)
    def _on_burst_finished(self, summary: dict):
        self._burst_active = False

        if self._burst_thread is not None:
            self._burst_thread.quit()
            self._burst_thread.wait()
            self._burst_thread = None
        self._burst_writer = None

        self._set_burst_readouts(
            summary['metadata']['burst_metadata']['frames_written'],
            summary['metadata']['burst_metadata']['frames_dropped'],
            summary['metadata']['burst_metadata']['elapsed_seconds'],
        )
        drop_pct = summary['metadata']['drop_rate_pct']
        status_msg = (
            f"Done – {summary['metadata']['burst_metadata']['frames_written']} frames, "
            f"{drop_pct:.1f}% dropped, "
            f"{summary['metadata']['actual_fps']:.1f} fps"
        )
        self._set_burst_status(status_msg)

        p = self.settings.child('burst', 'burst_enable')
        p.setValue(False)
        p.sigValueChanged.emit(p, False)

        self.emit_status(ThreadCommand('Update_Status', [status_msg]))
        self._publish_burst_summary(summary)
        self.metadata = None
        try:
            self.controller.camera.TriggerSelector.SetValue("FrameStart")
            self.controller.camera.TriggerMode.SetValue("Off")
        except Exception:
            pass        

    @QtCore.Slot(str)
    def _on_burst_error(self, msg: str):
        self._burst_active = False

        # Clean up thread
        if self._burst_thread is not None:
            self._burst_thread.quit()
            self._burst_thread.wait()
            self._burst_thread = None
        self._burst_writer = None

        # Reset arm button
        p = self.settings.child('burst', 'burst_enable')
        p.setValue(False)
        p.sigValueChanged.emit(p, False)

        try:
            self.controller.camera.TriggerSelector.SetValue("FrameStart")
            self.controller.camera.TriggerMode.SetValue("Off")
        except Exception:
            pass        

        self._set_burst_status(f"ERROR: {msg}")
        self.emit_status(ThreadCommand('Update_Status', [f"Burst error: {msg}", "log"]))

    def _set_burst_status(self, text: str):
        p = self.settings.child('burst', 'burst_status')
        p.setValue(text)
        p.sigValueChanged.emit(p, text)

    def _set_burst_readouts(self, written: int, dropped: int, elapsed: float):
        for name, val in (
            ('burst_written', written),
            ('burst_dropped', dropped),
            ('burst_elapsed', round(elapsed, 2)),
        ):
            p = self.settings.child('burst', name)
            p.setValue(val)
            p.sigValueChanged.emit(p, val)

    def _publish_burst_summary(self, summary: dict):
        if self.data_publisher is None:
            return
        try:
            publisher_name = self.settings.child('leco_log', 'publisher_name').value()
            self.data_publisher.send_data2({
                publisher_name: {
                    **summary,
                    "user_id": self.user_id,
                    "h5_path": self._burst_h5_path,
                }
            })
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status',
                                           [f"LECO burst publish failed: {e}"]))

    def handle_metadata_and_saving(self, frame, timestamp, shape):
        if not self.settings.child('trigger', 'TriggerMode').value():
            return
        metadata = self.get_metadata_and_save(frame, timestamp, shape)
        if self.send_frame_leco:
            self.publish_metadata(metadata, frame)
        else:
            self.publish_metadata(metadata)

    def get_metadata_and_save(self, frame, timestamp, shape):
        index = self.settings.child('trigger', 'TriggerSaveOptions', 'TriggerSaveIndex')
        filetype = self.settings.child('trigger', 'TriggerSaveOptions', 'Filetype').value()
        if self.metadata is not None:
            metadata = self.metadata
            filepath = self.metadata['file_metadata']['filepath']
            filename = self.metadata['file_metadata']['filename']
            self.metadata['burst_metadata']['user_id'] = self.user_id
            basepath = self.settings.child('leco_log', 'leco_basepath').value()
            filepath = os.path.normpath(
                os.path.join(basepath, filepath.lstrip(os.path.sep))
            )
        else:
            filepath = self.settings.child(
                'trigger', 'TriggerSaveOptions', 'TriggerSaveLocation'
            ).value()
            prefix = self.settings.child('trigger', 'TriggerSaveOptions', 'Prefix').value()
            if not filepath:
                filepath = os.path.join(os.path.expanduser('~'), 'Downloads')
            filename = f"{prefix}{index.value()}.{filetype}"
            metadata = {'burst_metadata': {}, 'file_metadata': {}, 'detector_metadata': {}}
            metadata['burst_metadata']['uuid'] = str(uuid7())
            metadata['burst_metadata']['user_id'] = self.user_id
            metadata['burst_metadata']['timestamp'] = timestamp
            metadata['file_metadata']['filepath'] = filepath
            metadata['file_metadata']['filename'] = filename
            index.setValue(index.value() + 1)
            index.sigValueChanged.emit(index, index.value())

        metadata['detector_metadata']['fuzziness'] = 0.1
        count = 0
        for name in self.controller.attribute_names:
            if 'Gain' in name and 'Auto' not in name:
                metadata['detector_metadata']['gain'] = self.settings.child('gain', name).value()
                count += 1
            if 'Exposure' in name and 'Auto' not in name:
                metadata['detector_metadata']['exposure_time'] = self.settings.child(
                    'exposure', name
                ).value()
                count += 1
            if count == 2:
                break
        metadata['detector_metadata']['shape'] = shape

        if filetype == 'h5':
            fname = filename if filename.endswith('.h5') else filename + '.h5'
            full_path = os.path.join(filepath, fname)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with h5py.File(full_path, 'w') as f:
                f.create_dataset(f"frame_{timestamp}", data=frame)
                for k, v in {
                    'uuid': metadata['burst_metadata']['uuid'],
                    'user_id': metadata['burst_metadata']['user_id'],
                    'timestamp': timestamp,
                    'exposure_time': metadata['detector_metadata']['exposure_time'],
                    'gain': metadata['detector_metadata']['gain'],
                    'shape': metadata['detector_metadata']['shape'],
                    'fuzziness': metadata['detector_metadata']['fuzziness'],
                    'format_version': 'hdf5-v0.1',
                }.items():
                    f.attrs[k] = v
        else:
            if filetype not in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
                self.emit_status(ThreadCommand('Update_Status',
                                               [f"Unsupported file type: {filetype}"]))
                return metadata
            fname = filename if filename.endswith(f'.{filetype}') else filename + f'.{filetype}'
            full_path = os.path.join(filepath, fname)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            iio.imwrite(full_path, frame)

        return metadata

    def publish_metadata(self, metadata, frame: Optional[np.ndarray] = None):
        if self.data_publisher is not None and self.save_frame:
            publisher_name = self.settings.child('leco_log', 'publisher_name').value()
            if self.send_frame_leco and frame is not None:
                self.data_publisher.send_data2({publisher_name: {
                    'frame': frame, 'metadata': metadata,
                    'message_type': 'detector',
                    'serial_number': self.controller.device_info.GetSerialNumber(),
                    'format_version': 'hdf5-v0.1',
                }})
            else:
                self.data_publisher.send_data2({publisher_name: {
                    'metadata': metadata,
                    'message_type': 'detector',
                    'serial_number': self.controller.device_info.GetSerialNumber(),
                    'format_version': 'hdf5-v0.1',
                }})

    def _prepare_view(self):
        (hstart, hend, vstart, vend, *binning) = self.controller.get_roi()
        try:
            xbin, ybin = binning
        except ValueError:
            xbin = ybin = 1
        height = hend - hstart
        width = vend - vstart

        self.settings.child('roi', 'width').setValue(width)
        self.settings.child('roi', 'height').setValue(height)

        mock_data = np.zeros((height, width))
        self.x_axis = Axis(label='Pixels', data=np.linspace(1, width, width), index=1)

        if width != 1 and height != 1:
            data_shape = 'Data2D'
            self.y_axis = Axis(label='Pixels', data=np.linspace(1, height, height), index=0)
            self.axes = [self.y_axis, self.x_axis]
        else:
            data_shape = 'Data1D'
            self.axes = [self.x_axis]

        if data_shape != self.data_shape:
            self.data_shape = data_shape
            self.dte_signal_temp.emit(
                DataToExport(
                    f'{self.user_id}',
                    data=[DataFromPlugins(
                        name=f'{self.user_id}',
                        data=[np.squeeze(mock_data)],
                        dim=self.data_shape,
                        labels=[f'{self.user_id}_{self.data_shape}'],
                        axes=self.axes,
                    )],
                )
            )
            QtWidgets.QApplication.processEvents()

    def update_rois(self, new_roi):
        (new_x, new_width, new_xbinning, new_y, new_height, new_ybinning) = new_roi
        if new_roi != self.controller.get_roi():
            self.controller.set_roi(
                hstart=new_x, hend=new_x + new_width,
                vstart=new_y, vend=new_y + new_height,
                hbin=new_xbinning, vbin=new_ybinning,
            )
            self.emit_status(ThreadCommand('Update_Status', [f'Changed ROI: {new_roi}']))
            self.controller.setup_acquisition()
            self._prepare_view()

    def roi_select(self, roi_info, ind_viewer):
        self.roi_info = roi_info

    def crosshair(self, crosshair_info, ind_viewer=0):
        self.crosshair_info = crosshair_info
        QtCore.QTimer.singleShot(200, QtWidgets.QApplication.processEvents)

    def camera_lost(self):
        self.close()
        self.emit_status(ThreadCommand('Update_Status',
                                       [f"Lost connection to {self.user_id}"]))

    def start_temperature_monitoring(self):
        self.temp_thread = QtCore.QThread()
        self.temp_worker = TemperatureMonitor(self.controller.camera)
        self.temp_worker.moveToThread(self.temp_thread)
        self.temp_thread.started.connect(self.temp_worker.run)
        self.temp_worker.temperature_updated.connect(self.on_temperature_update)
        self.temp_worker.finished.connect(self.temp_thread.quit)
        self.temp_worker.finished.connect(self.temp_worker.deleteLater)
        self.temp_thread.finished.connect(self.temp_thread.deleteLater)
        self.temp_thread.start()

    def stop_temp_monitoring(self):
        if hasattr(self, 'temp_worker') and self.temp_worker is not None:
            self.temp_worker.stop()
            self.temp_worker = None
        if hasattr(self, 'temp_thread') and self.temp_thread is not None:
            try:
                self.temp_thread.quit()
                self.temp_thread.wait()
            except RuntimeError:
                pass
            self.temp_thread = None
        p = self.settings.child('temperature', 'TemperatureMonitor')
        p.setValue(False)
        p.sigValueChanged.emit(p, p.value())

    def on_temperature_update(self, temp: float):
        p = self.settings.child('temperature', 'TemperatureAbs')
        p.setValue(temp)
        p.sigValueChanged.emit(p, temp)
        if temp > 60:
            self.emit_status(ThreadCommand('Update_Status',
                                           [f"WARNING: {self.user_id} camera is hot!!"]))

    def add_attributes_to_settings(self):
        existing_group_names = {child.name() for child in self.settings.children()}
        for attr in self.controller.attributes:
            attr_name = attr['name']
            if attr.get('type') == 'group':
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)
                else:
                    group_param = self.settings.child(attr_name)
                    existing_children = {c.name(): c for c in group_param.children()}
                    for expected in attr.get('children', []):
                        expected_name = expected['name']
                        if expected_name not in existing_children:
                            for old_name, old_child in existing_children.items():
                                if (old_child.opts.get('title') == expected.get('title')
                                        and old_name != expected_name):
                                    self.settings.child(attr_name, old_name).show(False)
                                    break
                            group_param.addChild(expected)
            else:
                if attr_name not in existing_group_names:
                    self.settings.addChild(attr)

    def update_params_ui(self):
        self.settings.child('device_info', 'DeviceModelName').setValue(self.controller.model_name)
        self.settings.child('device_info', 'DeviceSerialNumber').setValue(
            self.controller.device_info.GetSerialNumber())
        self.settings.child('device_info', 'DeviceVersion').setValue(
            self.controller.device_info.GetDeviceVersion())
        self.settings.child('device_info', 'DeviceUserID').setValue(
            self.controller.device_info.GetFriendlyName())

        for param in self.controller.attributes:
            param_type = param['type']
            param_name = param['name']
            if param_name in ("device_info", "device_state", "temperature"):
                continue
            if param_type == 'group':
                for child in param['children']:
                    child_name = child['name']
                    child_type = child['type']
                    if child_name == 'TriggerSaveOptions':
                        continue
                    camera_attr = getattr(self.controller.camera, child_name, None)
                    if camera_attr is None:
                        continue
                    try:
                        if child_type in ['float', 'slide', 'int', 'str']:
                            value = camera_attr.GetValue()
                        elif child_type == 'led_push':
                            value = (camera_attr.GetValue() if child_name == 'GammaEnable'
                                     else bool(camera_attr.GetIntValue()))
                        else:
                            continue
                        if 'Exposure' in child_name and 'Auto' not in child_name:
                            value *= 1e-3
                        self.settings.child(param_name, child_name).setValue(value)
                        if ('limits' in child and child_type in ['float', 'slide', 'int']
                                and not child.get('readonly', False)):
                            mn, mx = camera_attr.GetMin(), camera_attr.GetMax()
                            if 'Exposure' in child_name and 'Auto' not in child_name:
                                mn *= 1e-3
                                mx *= 1e-3
                            self.settings.child(param_name, child_name).setLimits([mn, mx])
                    except Exception:
                        pass
            else:
                camera_attr = getattr(self.controller.camera, param_name, None)
                if camera_attr is None:
                    continue
                try:
                    if param_type in ['float', 'slide', 'int', 'str']:
                        value = camera_attr.GetValue()
                    elif param_type == 'led_push':
                        value = (camera_attr.GetValue() if param_name == 'GammaEnable'
                                 else bool(camera_attr.GetIntValue()))
                    else:
                        continue
                    if 'Exposure' in param_name and 'Auto' not in param_name:
                        value *= 1e-3
                    self.settings.param(param_name).setValue(value)
                    if ('limits' in param and param_type in ['float', 'slide', 'int']
                            and not param.get('readonly', False)):
                        mn, mx = camera_attr.GetMin(), camera_attr.GetMax()
                        if 'Exposure' in param_name and 'Auto' not in param_name:
                            mn *= 1e-3
                            mx *= 1e-3
                        self.settings.param(param_name).setLimits([mn, mx])
                except Exception:
                    pass


if __name__ == '__main__':
    main(__file__, init=False)