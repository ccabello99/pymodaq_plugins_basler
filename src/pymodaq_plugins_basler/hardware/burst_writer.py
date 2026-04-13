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
"""

import queue
import time
import os
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
    # Carries the raw numpy array (a reference, not a copy – don't modify it).
    display_frame = QtCore.pyqtSignal(object)

    # Progress update: (frames_written, frames_dropped, elapsed_seconds)
    progress = QtCore.pyqtSignal(int, int, float)

    # Emitted once when the burst finishes (successfully or after an error).
    # Payload is a dict – see _build_summary() for the schema.
    burst_finished = QtCore.pyqtSignal(dict)

    # Emitted on an unrecoverable write error.
    error = QtCore.pyqtSignal(str)


class BurstWriter(QtCore.QObject):
    """
    Worker object that must be moved to a dedicated :class:`QtCore.QThread`.

    Usage::

        self.burst_writer = BurstWriter(...)
        self.burst_thread = QtCore.QThread()
        self.burst_writer.moveToThread(self.burst_thread)
        self.burst_thread.started.connect(self.burst_writer.run)
        self.burst_thread.start()

    The :meth:`enqueue` method is thread-safe and is the only method that
    should be called from outside the worker thread during a burst.

    Parameters
    ----------
    h5_path : str
        Full path of the HDF5 file to create.
    frame_shape : tuple[int, int]
        (height, width) of each frame.
    dtype : np.dtype
        Data type of each frame (e.g. np.uint16).
    max_frames : int or None
        Hard cap on frames to write. When reached the burst stops.
        Pass None for a time-bounded burst (use ``max_seconds`` instead).
    max_seconds : float or None
        Time cap in seconds. Ignored if ``max_frames`` is not None.
    display_every_nth : int
        Emit a display signal every Nth frame. 40 → ~25 Hz display at 1 kHz.
    chunk_size : int
        Number of frames written to HDF5 in a single dataset extend+write.
        Larger = fewer I/O calls, but more latency on the final flush.
    queue_maxsize : int
        Maximum number of frames held in the in-process queue. When full,
        behaviour is controlled by ``drop_oldest``.
    drop_oldest : bool
        If True, when the queue is full the oldest frame is discarded to make
        room for the new one (preserves recency). If False, the newest frame
        is discarded (safer for ordered reconstructions).
    camera_meta : dict
        Arbitrary camera metadata written as HDF5 root attributes.
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_frames is None and max_seconds is None:
            raise ValueError("One of max_frames or max_seconds must be set.")

        self.h5_path = h5_path
        self.frame_shape = frame_shape  # (w, h)
        self.dtype = dtype
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.display_every_nth = display_every_nth
        self.chunk_size = chunk_size
        self.drop_oldest = drop_oldest
        self.camera_meta = camera_meta or {}

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._frames_written: int = 0
        self._frames_dropped: int = 0
        self._frames_displayed: int = 0
        self._start_time: Optional[float] = None
        self._stop_requested: bool = False

        self.signals = BurstWriterSignals()

    def enqueue(self, frame: np.ndarray, timestamp: int) -> bool:
        """
        Put a frame into the writer queue.  Returns True if enqueued,
        False if the frame was dropped due to queue overflow.

        This method is thread-safe and designed to be as fast as possible
        – it does nothing except put an item into a Queue.
        """
        item = (frame, timestamp)
        if self._queue.full():
            self._frames_dropped += 1
            if self.drop_oldest:
                try:
                    self._queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self._queue.put_nowait(item)
            # else: just drop the newest (don't put it in)
            return False
        else:
            self._queue.put_nowait(item)
            return True

    def request_stop(self):
        """Signal the writer loop to finish after draining the queue."""
        self._stop_requested = True
        self._queue.put_nowait(_STOP_SENTINEL)

    def run(self):
        self._start_time = time.monotonic()
        w, h = self.frame_shape

        os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)

        try:
            with h5py.File(self.h5_path, "w") as f:
                # --- Pre-allocate resizable datasets ---
                # chunk layout: one HDF5 chunk = chunk_size frames.
                # This is the single most important tuning knob for I/O perf.
                frames_ds = f.create_dataset(
                    "frames",
                    shape=(0, w, h),
                    maxshape=(None, w, h),
                    dtype=self.dtype,
                    chunks=(self.chunk_size, w, h),
                    # No compression: raw throughput is the priority.
                    # Swap to compression=lzf if storage is a constraint.
                )
                ts_ds = f.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.uint64,
                    chunks=(max(self.chunk_size * 10, 1000),),
                )

                # --- Root attributes ---
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

                # --- Accumulation buffers (write in chunks, not one-by-one) ---
                frame_buf = np.empty((self.chunk_size, w, h), dtype=self.dtype)
                ts_buf = np.empty(self.chunk_size, dtype=np.uint64)
                buf_idx = 0

                last_progress_time = time.monotonic()

                while True:
                    # --- Check termination conditions ---
                    elapsed = time.monotonic() - self._start_time

                    if self.max_frames is not None:
                        if self._frames_written >= self.max_frames:
                            break

                    if self.max_seconds is not None and self.max_frames is None:
                        if elapsed >= self.max_seconds:
                            break

                    if self._stop_requested and self._queue.empty():
                        break

                    # --- Drain queue ---
                    try:
                        item = self._queue.get(timeout=0.005)  # 5 ms timeout
                    except queue.Empty:
                        continue

                    if item is _STOP_SENTINEL:
                        break

                    frame, ts = item

                    # --- Display throttle ---
                    total_seen = self._frames_written + buf_idx
                    if total_seen % self.display_every_nth == 0:
                        self.signals.display_frame.emit(frame)
                        self._frames_displayed += 1

                    # --- Buffer the frame ---
                    frame_buf[buf_idx] = frame
                    ts_buf[buf_idx] = ts
                    buf_idx += 1

                    # --- Flush buffer when full ---
                    if buf_idx == self.chunk_size:
                        self._write_chunk(frames_ds, ts_ds, frame_buf, ts_buf, buf_idx)
                        buf_idx = 0

                    # --- Throttled progress signal (every 200 ms) ---
                    now = time.monotonic()
                    if now - last_progress_time >= 0.2:
                        self.signals.progress.emit(
                            self._frames_written,
                            self._frames_dropped,
                            now - self._start_time,
                        )
                        last_progress_time = now

                # --- Flush any remaining buffered frames ---
                if buf_idx > 0:
                    self._write_chunk(frames_ds, ts_ds, frame_buf, ts_buf, buf_idx)

                # --- Write final metadata ---
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

        # --- Emit final summary for LECO publishing ---
        summary = self._build_summary()
        self.signals.progress.emit(
            self._frames_written, self._frames_dropped, time.monotonic() - self._start_time
        )
        self.signals.burst_finished.emit(summary)


    def _write_chunk(self, frames_ds, ts_ds, frame_buf, ts_buf, count):
        """Extend the datasets and write ``count`` frames from the buffers."""
        n = self._frames_written
        frames_ds.resize(n + count, axis=0)
        ts_ds.resize(n + count, axis=0)
        frames_ds[n : n + count] = frame_buf[:count]
        ts_ds[n : n + count] = ts_buf[:count]
        self._frames_written += count

    def _build_summary(self) -> dict:
        """
        Build the end-of-burst summary dict.
        This is what gets published over LECO.
        """
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        return {
            "message_type": "burst_finished",
            "format_version": "burst-v1.0",
            "h5_path": self.h5_path,
            "camera_model": self.camera_meta.get("camera_model", ""),
            "serial_number": self.camera_meta.get("serial_number", ""),
            "frames_written": self._frames_written,
            "frames_dropped": self._frames_dropped,
            "frames_displayed": self._frames_displayed,
            "elapsed_seconds": elapsed,
            "actual_fps": self._frames_written / elapsed if elapsed > 0 else 0.0,
            "drop_rate_pct": (
                100.0 * self._frames_dropped / max(self._frames_written + self._frames_dropped, 1)
            ),
            "exposure_time_ms": self.camera_meta.get("exposure_time_ms", 0.0),
            "gain": self.camera_meta.get("gain", 0.0),
            "roi": self.camera_meta.get("roi", []),
            "fps_target": self.camera_meta.get("fps_target", 1000),
        }
