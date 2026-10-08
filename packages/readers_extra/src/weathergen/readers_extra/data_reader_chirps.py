# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Data reader for CHIRPS v3 daily precipitation.

The CHIRPS stores are plain rioxarray-written zarr v2 groups holding a single
variable ``band`` with dims ``(time, latitude, longitude)`` on a regular
0.05 degree lat/lon grid (2000 x 7200, 50N-50S, 180W-180E), one store per year.
They are neither anemoi datasets nor WeatherGenerator obs datasets, hence this
reader.

The reader is laid out like the two core readers. From ``DataReaderAnemoi`` it
takes the channel selection (``source`` / ``target`` lists, ``*_exclude``, the
stage-specific lists the data sampler writes back), the target channel weights
and the shape of ``_get`` for gridded fields. From ``DataReaderObs`` it takes
the time handling: a per-window index built once in ``_setup_sample_index`` from
the time stamps of the data, instead of the start-plus-period arithmetic of
``DataReaderTimestep``.

Time
----
A value stamped day D in a store is the precipitation total from D 00:00 UTC to
D+1 00:00 UTC. The stores do not document this; it was measured against hourly
ERA5 and half-hourly IMERG precipitation and holds to within 1-3 hours (see
``scripts_vh/chirps/chirps_time_convention.py``).

The reader labels that day at D + ``daily_anchor``. The label decides which
model window the day falls into, so it has to be identical to the label the
ERA5 daily mean of the same day carries: 23:00 for the mean of the hourly states
D 00:00 .. D 23:00 (``rolling_average: [-23, 0]`` at ``frequency: 24:00:00``).
With identical labels the two streams share a window for every window start and
length, not only for windows that happen to begin at 00 UTC.

``DataReaderTimestep`` is not used because its arithmetic returns nothing for a
window that begins before the first label, which would drop 1 January of every
yearly store, and because it assumes an unbroken time axis, which 1990 does not
have (1990-01-01 is missing).

Values
------
* Ocean / no-data is the sentinel ``-9999.0``, not NaN. The ``nodata`` attribute
  is not CF-standard so nothing masks it automatically, and
  ``ReaderData.remove_nan_coords_and_geoinfos`` does not catch it either. Only
  non-negative values are returned, about 3.97M land points per day.
* Every valid point of the day is returned at native resolution. Subsampling is
  left to ``max_num_targets`` in the stream config, which draws the subset with
  the data sampler's own seeded generator.
* The stores carry no normalization statistics (anemoi datasets expose
  ``.statistics``, obs datasets ``means`` / ``vars``), so ``mean`` and ``stdev``
  come from the stream config and must be the statistics of the transformed
  field.
"""

import logging
from pathlib import Path
from typing import override

import numpy as np
import torch
import zarr
from numpy.typing import NDArray
from xarray.coding.times import decode_cf_datetime

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderChirps(DataReaderBase):
    "Reader for a single yearly CHIRPS v3 zarr store"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        """
        Construct data reader for one CHIRPS store

        Parameters
        ----------
        filename :
            filename (and path) of one ``chirpsv3_<year>.zarr`` store
        stream_info :
            information about stream. Keys beyond the common ones: ``daily_anchor``
            (required), ``mean`` and ``stdev`` (required), ``channels`` (default
            ``['tp']``) and ``transform`` (``null`` or ``log1p``).

        Returns
        -------
        None
        """

        super().__init__(tw_handler, stream_info)

        self.filename = filename
        self.z = zarr.open(str(filename), mode="r")
        self.data = self.z["band"]

        # caches lats and lons; the (2000 x 7200) point grid is never materialised
        self.latitudes = self.z["latitude"][:].astype(np.float32)
        self.longitudes = self.z["longitude"][:].astype(np.float32)

        # CHIRPS holds a single field
        self.channels = list(stream_info.get("channels", ["tp"]))
        assert len(self.channels) == 1, (
            f"{stream_info['name']}: CHIRPS has one field, got channels {self.channels}."
        )

        # NOTE: there is deliberately no init_empty() when the store does not overlap the data
        # loader window, as DataReaderAnemoi has. With one reader per year most readers are out
        # of range, and MultiStreamDataSampler takes the channels from the last reader and the
        # denormalization from the first one. Channels and statistics come from the stream
        # config, so they are always set; _get returns empty ReaderData outside this store.

        # select/filter requested source channels
        if stream_info.get(str(stage) + "_source_channels") is None:
            self.source_channels = self.select_channels("source")
        else:
            self.source_channels = stream_info.get(str(stage) + "_source_channels")
        self.source_idx = [self.channels.index(ch) for ch in self.source_channels]

        # select/filter requested target channels
        if stream_info.get(str(stage) + "_target_channels") is None:
            self.target_channels = self.select_channels("target")
        else:
            self.target_channels = stream_info.get(str(stage) + "_target_channels")
        self.target_idx = [self.channels.index(ch) for ch in self.target_channels]

        # get target channel weights from stream config
        if stream_info.get("target_channel_weights") is None:
            self.target_channel_weights = self.parse_target_channel_weights()
        else:
            self.target_channel_weights = stream_info.get("target_channel_weights")

        # CHIRPS carries no auxiliary fields
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        # load additional properties (transform, mean, stdev)
        self._load_properties()

        # Create index for samples
        self._setup_sample_index()

        self.len = len(self.dt)

        ds_name = stream_info["name"]
        _logger.info(
            f"{ds_name}: {Path(filename).name}: {self.len} days labelled {self.dt[0]} .. "
            f"{self.dt[-1]}, source channels: {self.source_channels}, "
            f"target channels: {self.target_channels}, transform: {self.transform}"
        )

    @override
    def length(self) -> int:
        return self.len

    def select_channels(self, ch_type: str) -> list[str]:
        """
        Select source or target channels

        Same rules as DataReaderAnemoi.select_channels: a channel is used if it is in the
        `source` / `target` list (when one is given; `source: []` declares a target-only
        stream) and not in `source_exclude` / `target_exclude`.

        Parameters
        ----------
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        list of selected channel names
        """

        channels = self.stream_info.get(ch_type)
        channels_exclude = self.stream_info.get(ch_type + "_exclude", []) or []

        unknown = [ch for ch in channels or [] if ch not in self.channels]
        assert len(unknown) == 0, (
            f"{self.stream_info['name']}: unknown {ch_type} channel(s) {unknown}; "
            f"available channels are {self.channels}."
        )

        return [
            ch
            for ch in self.channels
            if (ch in channels if channels is not None else True) and ch not in channels_exclude
        ]

    def _load_properties(self) -> None:
        """
        Transform and normalization statistics, from the stream config since the stores carry
        none. mean / stdev are those of the transformed field.
        """

        ds_name = self.stream_info["name"]

        # optional transform applied to raw mm/day before normalization
        self.transform = self.stream_info.get("transform")
        assert self.transform in (None, "none", "log1p"), (
            f"{ds_name}: unsupported CHIRPS transform {self.transform!r}."
        )

        mean, stdev = self.stream_info.get("mean"), self.stream_info.get("stdev")
        assert mean is not None and stdev is not None and len(mean) == len(stdev) == 1, (
            f"{ds_name}: 'mean' and 'stdev' of the transformed field must be set in the stream "
            "config (one value each); the CHIRPS stores carry no statistics."
        )
        self.mean = np.asarray(list(mean), dtype=np.float64)
        self.stdev = np.asarray(list(stdev), dtype=np.float64)

    def _setup_sample_index(self) -> None:
        """
        Dataset is divided into samples;
           - one sample per time window of the data loader
           - index arrays have one entry for each sample; they contain the first day and the
             day after the last day whose label lies in the window [start, end)
           - both are equal for a window without data in this store
        """

        ds_name = self.stream_info["name"]

        # day stamps of the store
        time = self.z["time"]
        days = np.asarray(
            decode_cf_datetime(time[:], time.attrs["units"], time.attrs.get("calendar")),
            dtype="datetime64[ns]",
        )
        assert np.all(days == days.astype("datetime64[D]")) and np.all(np.diff(days) > 0), (
            f"{self.filename}: expected increasing day stamps at 00 UTC."
        )

        # label of each day: stamp + daily_anchor; the total of the day is reported at this time
        # raised rather than asserted: without it the labels, and with them the pairing with
        # the other streams, would be undefined
        if self.stream_info.get("daily_anchor") is None:
            raise ValueError(
                f"{ds_name}: 'daily_anchor' must be set in the stream config. It is the time of "
                "day a CHIRPS day is labelled with and has to equal the label of the ERA5 daily "
                "mean (23:00:00 for rolling_average [-23, 0]); otherwise the two can land in "
                "different windows."
            )
        anchor = parse_timedelta(self.stream_info["daily_anchor"])
        assert np.timedelta64(0, "h") <= anchor < np.timedelta64(24, "h"), (
            f"{ds_name}: daily_anchor must lie within the day, got {anchor}."
        )
        self.dt = days + anchor

        # start and end of all time windows of the data loader
        tw_handler = self.time_window_handler
        num_windows = int(tw_handler.get_index_range().end) + 1
        t_starts = tw_handler.t_start + tw_handler.t_window_step * np.arange(num_windows)
        t_ends = t_starts + tw_handler.t_window_len

        # windows follow the convention [t_start, t_end) where endpoint is excluded
        self.indices_start = np.searchsorted(self.dt, t_starts.astype(self.dt.dtype), side="left")
        self.indices_end = np.searchsorted(self.dt, t_ends.astype(self.dt.dtype), side="left")

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        if len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        if idx < 0 or idx >= len(self.indices_start):
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        didx_start = self.indices_start[idx]
        didx_end = self.indices_end[idx]

        if didx_start == didx_end:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        coords, data, datetimes = [], [], []
        for didx in range(didx_start, didx_end):
            field = self.data[didx]

            # only valid points: drops the -9999 sentinel (and NaN, should a store contain it)
            mask = field >= 0.0
            lat_idx, lon_idx = np.divmod(np.flatnonzero(mask), field.shape[1])

            # construct lat/lon coords
            coords += [np.stack([self.latitudes[lat_idx], self.longitudes[lon_idx]], axis=1)]
            data += [field[mask]]
            # date time matching #data points of data
            datetimes += [np.full(lat_idx.size, self.dt[didx])]

        data = np.concatenate(data).astype(np.float32)
        if self.transform == "log1p":
            np.log1p(data, out=data)

        # one column per requested channel (CHIRPS has a single field)
        data = np.repeat(data[:, None], len(channels_idx), axis=1)
        geoinfos = np.zeros((data.shape[0], len(self.geoinfo_idx)), dtype=np.float32)

        rd = ReaderData(
            coords=np.concatenate(coords),
            geoinfos=geoinfos,
            data=data,
            datetimes=np.concatenate(datetimes),
        )

        dtr = self.time_window_handler.window(idx)
        check_reader_data(rd, dtr)

        return rd

    @override
    def denormalize_source_channels(self, source: NDArray) -> NDArray:
        """
        Denormalize source channels and undo the transform, so that output is in mm/day
        """
        source = super().denormalize_source_channels(source)
        if self.transform == "log1p":
            source = _expm1(source)
        return source

    @override
    def denormalize_target_channels(self, target: NDArray) -> NDArray:
        """
        Denormalize target channels and undo the transform, so that output is in mm/day
        """
        target = super().denormalize_target_channels(target)
        if self.transform == "log1p":
            target = _expm1(target)
        return target


def _expm1(data: NDArray | torch.Tensor) -> NDArray | torch.Tensor:
    """
    Invert log1p for whichever array type the caller uses.

    The denormalization hooks are typed for numpy, but write_output calls them with the
    model's torch tensors, which are still on the GPU at that point; np.expm1 would try to
    convert those to numpy and raise.
    """
    return torch.expm1(data) if isinstance(data, torch.Tensor) else np.expm1(data)
