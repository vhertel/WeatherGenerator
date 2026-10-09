# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderOffgrid(DataReaderTimestep):
    """Reader that evaluates a stream at an arbitrary set of points.

    Instead of returning the stream's own data, this reader returns a fixed
    coordinate template repeated at a configurable frequency across the time
    window. Channel identity and normalization statistics are read from the
    stream's real dataset (see `_read_reference_metadata`), so the model sees
    a stream that is identical in every respect except where it is sampled.

    Configured entirely from the stream config::

        ERA5_offgrid:
          type: offgrid
          base_type: anemoi
          filenames: ['era5.zarr']
          offgrid:
            coords: /abs/path/offgrid_regular.npy
            geoinfos: /abs/path/offgrid_regular_geoinfos.npy
            frequency: 6h

    ``offgrid.coords`` and ``offgrid.geoinfos`` must be absolute paths.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        """Construct an offgrid reader.

        Parameters
        ----------
        tw_handler :
            Time window handler defining the inference window.
        filename :
            Resolved path to the stream's real dataset, used to read channel
            identity and normalization statistics.
        stream_info :
            Stream configuration. Must contain ``base_type`` and an ``offgrid``
            block with at least ``coords``.
        stage :
            Pipeline stage. Unused here; accepted because the generic reader
            dispatch in `_init_stream_datasets` always passes it.
        """
        name = stream_info.get("name", "<unnamed>")

        offgrid_cfg = stream_info.get("offgrid")
        if offgrid_cfg is None:
            msg = f"Stream '{name}' has type 'offgrid' but no 'offgrid' config block."
            raise ValueError(msg)

        # TODO
        # A single template is emitted per reader, so several filenames would
        # produce the same points several times over.
        filenames = stream_info.get("filenames", [])
        if len(filenames) > 1:
            msg = (
                f"Stream '{name}': offgrid streams accept at most one entry in "
                f"'filenames', got {len(filenames)}. The template would otherwise be "
                "emitted once per file."
            )
            raise ValueError(msg)

        coords_cfg = offgrid_cfg.get("coords")
        if coords_cfg is None:
            msg = f"Stream '{name}': offgrid.coords is required."
            raise ValueError(msg)
        coords_path = Path(coords_cfg)
        if not coords_path.exists():
            msg = f"Stream '{name}': offgrid.coords not found: {coords_path}"
            raise FileNotFoundError(msg)

        geoinfos_cfg = offgrid_cfg.get("geoinfos")
        geoinfos_path = Path(geoinfos_cfg) if geoinfos_cfg is not None else None
        if geoinfos_path is not None and not geoinfos_path.exists():
            msg = f"Stream '{name}': offgrid.geoinfos not found: {geoinfos_path}"
            raise FileNotFoundError(msg)

        # Sampling frequency of the template; defaults to the time window step.
        frequency = offgrid_cfg.get("frequency", tw_handler.t_window_step)
        period = parse_timedelta(frequency)

        # initialize base class with time window and frequency info
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time=tw_handler.t_start,
            data_end_time=tw_handler.t_end,
            period=period,
        )

        # load and validate coordinate template
        coords_arr = np.load(coords_path)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            msg = (
                f"Stream '{name}': template must be .npy with shape (N, 2) "
                f"[lat, lon], got {coords_arr.shape}"
            )
            raise ValueError(msg)

        # caches lats and lons
        self.latitudes = _clip_lat(coords_arr[:, 0])
        self.longitudes = _clip_lon(coords_arr[:, 1])
        self._latlon = np.stack([self.latitudes, self.longitudes], axis=1)

        # number of template points and available temporal samples
        self.n_points = len(self.latitudes)
        self.len = max(0, int((tw_handler.t_end - tw_handler.t_start) / period))

        # Read channel identity and normalization statistics from the real
        # dataset. Everything except the sampling locations comes from here.
        (
            self.source_channels,
            self.source_idx,
            self.target_channels,
            self.target_idx,
            self.target_channel_weights,
            self.mean,
            self.stdev,
        ) = _read_reference_metadata(filename, stream_info)

        # load optional geoinfos and validate row/column alignment
        geoinfo_channels_cfg = list(stream_info.get("geoinfo_channels") or [])
        if geoinfos_path is not None:
            geoinfo_arr = np.load(geoinfos_path)
            if geoinfo_arr.ndim != 2 or geoinfo_arr.shape[0] != self.n_points:
                msg = (
                    f"Stream '{name}': geoinfos file {geoinfos_path} has shape "
                    f"{geoinfo_arr.shape}, expected ({self.n_points}, G)."
                )
                raise ValueError(msg)
            if geoinfo_arr.shape[1] != len(geoinfo_channels_cfg):
                msg = (
                    f"Stream '{name}': geoinfos file {geoinfos_path} has "
                    f"{geoinfo_arr.shape[1]} columns but geoinfo_channels lists "
                    f"{len(geoinfo_channels_cfg)} entries."
                )
                raise ValueError(msg)
            self.geoinfos = geoinfo_arr.astype(np.float32)
        else:
            if len(geoinfo_channels_cfg) > 0:
                _logger.warning(
                    f"Stream '{name}': geoinfo_channels configured but no geoinfo "
                    ".npy file was provided; offgrid reader will use empty geoinfos."
                )
            geoinfo_channels_cfg = []
            self.geoinfos = np.zeros((self.n_points, 0), dtype=np.float32)

        # geoinfo metadata and normalization statistics
        self.geoinfo_channels = geoinfo_channels_cfg
        self.geoinfo_idx = np.arange(len(self.geoinfo_channels), dtype=np.int64)
        self.mean_geoinfo, self.stdev_geoinfo = _geoinfo_stats(self.geoinfos)

        _logger.info(
            f"{name}: offgrid reader active over {self.n_points} points "
            f"(base_type={stream_info['base_type']}, source={len(self.source_channels)}, "
            f"target={len(self.target_channels)}, geoinfo={len(self.geoinfo_channels)}, "
            f"steps={self.len})."
        )

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.n_points = 0
        self.latitudes = np.zeros(0, dtype=np.float32)
        self.longitudes = np.zeros(0, dtype=np.float32)
        self._latlon = np.zeros((0, 2), dtype=np.float32)
        self.geoinfos = np.zeros((0, 0), dtype=np.float32)

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """Get offgrid samples for one time window.

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData
            Window data with tiled coords/geoinfos, placeholder data fields,
            and datetimes.
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )
        assert t_idxs[0] >= 0, "index must be non-negative"

        n_steps = len(t_idxs)
        n_total = self.n_points * n_steps

        # repeat the template coordinates for each timestep in the window
        coords = np.tile(self._latlon, (n_steps, 1))

        # offgrid reader provides no atmospheric values; keep placeholder zeros
        data = np.zeros((n_total, len(channels_idx)), dtype=np.float32)

        # repeat geoinfos for each timestep
        geoinfos = np.tile(self.geoinfos, (n_steps, 1))

        # compute absolute times for each step, then repeat per grid point
        step_times = self.data_start_time + self.period * t_idxs
        datetimes = np.repeat(step_times, self.n_points)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd


def _geoinfo_stats(
    geoinfos: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-column mean and standard deviation of the template geoinfos.

    Standard deviations below 1e-6 are replaced by 1.0 so that normalization
    never divides by (near) zero.
    """
    if geoinfos.shape[1] == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)

    mean = np.mean(geoinfos, axis=0).astype(np.float32)
    stdev = np.std(geoinfos, axis=0).astype(np.float32)
    stdev[stdev < 1e-6] = 1.0
    return mean, stdev


def _read_reference_metadata(
    filename: Path, stream_info: dict
) -> tuple[
    list[str],
    NDArray[np.int64],
    list[str],
    NDArray[np.int64],
    list[float],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Read which channels the stream's real dataset exposes and their
    normalization statistics, without building a full reader for it.

    Offgrid never reads real data through this dataset -- it only needs to
    know which channels the model was trained on and their mean/stdev, so
    none of a full reader's windowed-indexing machinery (time window, period,
    empty-window fallback) applies here. This intentionally duplicates the
    small channel-selection rule from the underlying reader rather than
    reusing it, to avoid depending on that reader's internals.

    Returns
    -------
    Tuple of (source_channels, source_idx, target_channels, target_idx,
    target_channel_weights, mean, stdev), in that order.
    """
    name = stream_info.get("name", "<unnamed>")

    base_type = stream_info.get("base_type")
    if base_type != "anemoi":
        msg = (
            f"Stream '{name}': offgrid streams currently only support "
            f"base_type 'anemoi', got '{base_type}'."
        )
        raise ValueError(msg)

    # Lazy import, matching the convention in registry.py.
    import anemoi.datasets as anemoi_datasets

    ds0 = anemoi_datasets.open_dataset(filename)

    source_idx = _select_anemoi_channels(stream_info, ds0, "source")
    target_idx = _select_anemoi_channels(stream_info, ds0, "target")
    target_channels = [ds0.variables[i] for i in target_idx]

    channel_weights = stream_info.get("channel_weights")
    target_channel_weights = [
        channel_weights.get(ch, 1.0) if channel_weights else 1.0 for ch in target_channels
    ]

    return (
        [ds0.variables[i] for i in source_idx],
        source_idx,
        target_channels,
        target_idx,
        target_channel_weights,
        np.array(ds0.statistics["mean"], copy=True),
        np.array(ds0.statistics["stdev"], copy=True),
    )


def _select_anemoi_channels(stream_info: dict, ds0, ch_type: str) -> NDArray[np.int64]:
    """Select source/target channel indices from a raw anemoi dataset.

    Mirrors DataReaderAnemoi's rule: every non-forcing, non-constant-in-time
    variable is included unless it's named in `<ch_type>_exclude`, or
    `<ch_type>` is given as an explicit allow-list.
    """
    channels = stream_info.get(ch_type)
    channels_exclude = stream_info.get(ch_type + "_exclude", [])
    chs_idx = np.sort(
        [
            ds0.name_to_index[k]
            for (k, v) in ds0.typed_variables.items()
            if (
                not v.is_computed_forcing
                and not v.is_constant_in_time
                and (np.array([f == k for f in channels]).any() if channels is not None else True)
                and not np.array([f == k for f in channels_exclude]).any()
            )
        ]
    )
    return np.array(chs_idx, dtype=np.int64)


def _clip_lat(lats: NDArray) -> NDArray[np.float32]:
    """Clip latitudes to the range [-90, 90] and ensure periodicity."""
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray) -> NDArray[np.float32]:
    """Clip longitudes to the range [-180, 180] and ensure periodicity."""
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)
