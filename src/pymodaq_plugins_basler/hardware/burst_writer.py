"""
burst_writer.py
---------------
Dedicated writer thread for high-speed (≥1 kHz) burst acquisition to HDF5.

Design principles:
  - The grab callback puts frames into a queue and returns immediately.
  - This worker drains the queue in configurable chunks and writes to a
    single pre-allocated HDF5 dataset so that open/close overhead is paid
    only once per burst.
  - Display throttling (every Nth frame) is handled here so the grab
    callback never touches Qt signals directly during a burst.
  - A summary signal fires at the end of the burst, carrying everything
    needed for a LECO end-of-burst publish.
  - The writer blocks until the first frame arrives before starting its
    timer and termination logic, so time/frame-bounded bursts are not
    consumed by the wait for the hardware trigger.
"""

import queue
import threading
import time
import os
import json
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
from qtpy import QtCore

if not hasattr(QtCore, "pyqtSignal"):
    QtCore.pyqtSignal = QtCore.Signal  # type: ignore


# Queue item sentinel – put this into the queue to signal "burst is done"
_STOP_SENTINEL = object()


class BurstWriterSignals(QtCore.QObject):
    """All Qt signals emitted by :class:`BurstWriter`."""

    # Emitted for every Nth frame so the GUI can refresh at a sane rate.
    display_frame = QtCore.pyqtSignal(object)

    # Progress update: (frames_written, frames_dropped, elapsed_seconds)
    progress = QtCore.pyqtSignal(int, int, float)

    # Emitted once the first frame has been received – useful for GUI feedback.
    first_frame_received = QtCore.pyqtSignal()

    # Emitted once when the burst finishes (successfully or after an error).
    burst_finished = QtCore.pyqtSignal(dict)

    # Emitted on an unrecoverable write error.
    error = QtCore.pyqtSignal(str)


class BurstWriter(QtCore.QObject):
    """
    Worker object that must be moved to a dedicated :class:`QtCore.QThread`.

    The writer blocks in :meth:`run` until the first frame arrives via
    :meth:`enqueue`, then starts its timer and termination logic.  This means
    time/frame-bounded bursts are not consumed by the hardware-trigger latency.

    Parameters
    ----------
    h5_path : str
        Full path of the HDF5 file to create.
    frame_shape : tuple[int, int]
        (height, width) of each frame in numpy (row, col) order.
    dtype : np.dtype
        Data type of each frame (e.g. np.uint16).
    max_frames : int or None
        Hard cap on frames to write. Pass None for a time-bounded burst.
    max_seconds : float or None
        Time cap in seconds. Ignored if ``max_frames`` is not None.
    display_every_nth : int
        Emit a display signal every Nth frame.
    chunk_size : int
        Frames written to HDF5 per extend+write call.
    queue_maxsize : int
        Maximum frames held in the in-process queue before overflow.
    drop_oldest : bool
        If True, oldest frame is dropped on overflow. If False, newest is dropped.
    camera_meta : dict
        Arbitrary camera metadata written as HDF5 root attributes.
    first_frame_timeout : float
        Seconds to wait for the first frame before aborting. Default 30 s.
    """

    def __init__(
        self,
        h5_path: str,
        frame_shape: tuple,
        dtype=np.uint16,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
        display_every_nth: int = 40,
        chunk_size: int = 50,
        queue_maxsize: int = 500,
        drop_oldest: bool = False,
        camera_meta: Optional[dict] = None,
        first_frame_timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_frames is None and max_seconds is None:
            raise ValueError("One of max_frames or max_seconds must be set.")

        self.h5_path = h5_path
        self.frame_shape = frame_shape  # (H, W) — numpy row-major
        self.dtype = dtype
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.display_every_nth = display_every_nth
        self.chunk_size = chunk_size
        self.drop_oldest = drop_oldest
        self.camera_meta = camera_meta or {}
        self.first_frame_timeout = first_frame_timeout

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._frames_written: int = 0
        self._frames_dropped: int = 0
        self._frames_displayed: int = 0
        self._start_time: Optional[float] = None
        self._stop_requested: bool = False

        # Gate: run() blocks here until the first frame arrives.
        self._first_frame_event = threading.Event()

        self.signals = BurstWriterSignals()

    def enqueue(self, frame: np.ndarray, timestamp: int) -> bool:
        """
        Put a frame into the writer queue.  Thread-safe, returns in microseconds.
        Returns True if enqueued, False if dropped due to overflow.
        """
        # Release the first-frame gate on the very first call.
        if not self._first_frame_event.is_set():
            self._first_frame_event.set()

        item = (frame, timestamp)
        if self._queue.full():
            self._frames_dropped += 1
            if self.drop_oldest:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(item)
            return False
        else:
            self._queue.put_nowait(item)
            return True

    def request_stop(self):
        """Signal the writer loop to finish after draining the queue."""
        self._stop_requested = True
        # Also release the first-frame gate in case we are stopped while
        # waiting for the hardware trigger (e.g. user aborts).
        self._first_frame_event.set()
        self._queue.put_nowait(_STOP_SENTINEL)

    def run(self):
        h, w = self.frame_shape

        os.makedirs(os.path.dirname(os.path.abspath(self.h5_path)), exist_ok=True)

        try:
            with h5py.File(self.h5_path, "w") as f:

                # --- Pre-allocate resizable datasets ---
                frames_ds = f.create_dataset(
                    "frames",
                    shape=(0, h, w),
                    maxshape=(None, h, w),
                    dtype=self.dtype,
                    chunks=(self.chunk_size, h, w),
                )
                ts_ds = f.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.uint64,
                    chunks=(max(self.chunk_size * 10, 1000),),
                )

                # --- Root attributes written immediately so the file is
                #     well-formed even if the process is killed mid-burst ---
                f.attrs["format_version"] = "burst-v1.0"
                f.attrs["camera_model"] = self.camera_meta.get("camera_model", "")
                f.attrs["serial_number"] = self.camera_meta.get("serial_number", "")
                f.attrs["exposure_time_ms"] = self.camera_meta.get("exposure_time_ms", 0.0)
                f.attrs["gain"] = self.camera_meta.get("gain", 0.0)
                f.attrs["roi"] = self.camera_meta.get("roi", [0, 0, w, h])
                f.attrs["fps_target"] = self.camera_meta.get("fps_target", 1000)
                f.attrs["max_frames_requested"] = self.max_frames or -1
                f.attrs["max_seconds_requested"] = self.max_seconds or -1.0
                f.attrs["created_utc"] = datetime.utcnow().isoformat()
                f.attrs["sequence_uuid"] = self.camera_meta.get("sequence_uuid", "")
                f.attrs["fuzziness"] = self.camera_meta.get("fuzziness", 0.1)
                leco_meta = self.camera_meta.get("conduktor_metadata", {})
                if leco_meta:
                    f.attrs["conduktor_metadata"] = json.dumps(leco_meta)

                # --------------------------------------------------------
                # GATE: block here until the first frame arrives from the
                # hardware trigger (Line3 FrameStart), or until timeout /
                # stop is requested.  The acquisition timer starts only
                # after this gate opens, so the full max_frames / max_seconds
                # budget is available for actual image data.
                # --------------------------------------------------------
                arrived = self._first_frame_event.wait(timeout=self.first_frame_timeout)

                if not arrived:
                    self.signals.error.emit(
                        f"No frame received within {self.first_frame_timeout:.0f} s "
                        f"— check hardware trigger wiring on Line1 / Line3."
                    )
                    return

                if self._stop_requested and self._queue.empty():
                    # Stopped before any frame arrived (user abort while waiting).
                    self.signals.burst_finished.emit(self._build_summary())
                    return

                # Timer starts NOW — after the first frame has arrived.
                self._start_time = time.monotonic()
                self.signals.first_frame_received.emit()

                # --- Accumulation buffers ---
                frame_buf = np.empty((self.chunk_size, h, w), dtype=self.dtype)
                ts_buf = np.empty(self.chunk_size, dtype=np.uint64)
                buf_idx = 0
                last_progress_time = self._start_time

                while True:
                    elapsed = time.monotonic() - self._start_time

                    if self.max_frames is not None and self._frames_written >= self.max_frames:
                        break

                    if self.max_seconds is not None and self.max_frames is None:
                        if elapsed >= self.max_seconds:
                            break

                    if self._stop_requested and self._queue.empty():
                        break

                    try:
                        item = self._queue.get(timeout=0.005)
                    except queue.Empty:
                        continue

                    if item is _STOP_SENTINEL:
                        break

                    frame, ts = item

                    # Display throttle
                    total_seen = self._frames_written + buf_idx
                    if total_seen % self.display_every_nth == 0:
                        self.signals.display_frame.emit(frame)
                        self._frames_displayed += 1

                    # Buffer
                    frame_buf[buf_idx] = frame
                    ts_buf[buf_idx] = ts
                    buf_idx += 1

                    if buf_idx == self.chunk_size:
                        self._write_chunk(frames_ds, ts_ds, frame_buf, ts_buf, buf_idx)
                        buf_idx = 0

                    now = time.monotonic()
                    if now - last_progress_time >= 0.2:
                        self.signals.progress.emit(
                            self._frames_written,
                            self._frames_dropped,
                            now - self._start_time,
                        )
                        last_progress_time = now

                # Flush remainder
                if buf_idx > 0:
                    self._write_chunk(frames_ds, ts_ds, frame_buf, ts_buf, buf_idx)

                # Final metadata
                total_elapsed = time.monotonic() - self._start_time
                f.attrs["frames_written"] = self._frames_written
                f.attrs["frames_dropped"] = self._frames_dropped
                f.attrs["elapsed_seconds"] = total_elapsed
                f.attrs["actual_fps"] = (
                    self._frames_written / total_elapsed if total_elapsed > 0 else 0.0
                )

        except Exception as exc:
            self.signals.error.emit(str(exc))
            return

        summary = self._build_summary()
        self.signals.progress.emit(
            self._frames_written,
            self._frames_dropped,
            time.monotonic() - self._start_time,
        )
        self.signals.burst_finished.emit(summary)


    def _write_chunk(self, frames_ds, ts_ds, frame_buf, ts_buf, count):
        n = self._frames_written
        frames_ds.resize(n + count, axis=0)
        ts_ds.resize(n + count, axis=0)
        frames_ds[n: n + count] = frame_buf[:count]
        ts_ds[n: n + count] = ts_buf[:count]
        self._frames_written += count

    def _build_summary(self) -> dict:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        h5_dir = os.path.dirname(self.h5_path)
        h5_filename = os.path.basename(self.h5_path)
        return {
            "message_type": "detector",
            'metadata': {
                "format_version": "burst-v1.0",
                "actual_fps": self._frames_written / elapsed if elapsed > 0 else 0.0,
                "drop_rate_pct": (
                    100.0 * self._frames_dropped
                    / max(self._frames_written + self._frames_dropped, 1)
                ),
                "burst_metadata": {
                    "uuid": self.camera_meta.get("uuid", ""),
                    "sequence_uuid": self.camera_meta.get("sequence_uuid", ""),
                    "user_id": self.camera_meta.get("serial_number", ""),
                    "frames_written": self._frames_written,
                    "frames_dropped": self._frames_dropped,
                    "frames_displayed": self._frames_displayed,
                    "elapsed_seconds": elapsed,
                    "fps_target": self.camera_meta.get("fps_target", 1000),
                },
                "file_metadata": {
                    "filepath": h5_dir,
                    "filename": h5_filename,
                },
                "detector_metadata": {
                    "fuzziness": self.camera_meta.get("fuzziness", 0.1),
                    "gain": self.camera_meta.get("gain", 0.0),
                    "exposure_time": self.camera_meta.get("exposure_time_ms", 0.0),
                    "shape": self.camera_meta.get("roi", []),
                    "camera_model": self.camera_meta.get("camera_model", ""),
                    "serial_number": self.camera_meta.get("serial_number", ""),
                }
            }
        }