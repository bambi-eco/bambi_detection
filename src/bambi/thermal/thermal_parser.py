# -*- coding: utf-8 -*-
"""
Thermal image parser for DJI and FLIR radiometric JPEGs.

Based on SanNianYiSi/thermal_parser (MIT licence).

The DJI SDK DLL directory (https://www.dji.com/at/downloads/softwares/dji-thermal-sdk)
is supplied explicitly via the constructor.  The caller is responsible for locating
the SDK on the current system and passing the path in.

Metadata (camera model, FLIR calibration params, DJI measurement params) is read
via Pillow and pure-Python binary/XMP parsing — no exiftool required.

Typical usage::

    thermal = Thermal(
        sdk_dir="/path/to/dji_sdk/utility/bin/windows/release_x64",
    )
    temp_array = thermal.parse("/path/to/DJI_20260301_T.JPG")
    thermal.close()

*sdk_dir* is optional.  When omitted the DJI SDK is not loaded and only the
pure-Python R-JPEG fallback is available.
"""

import os
import re
import platform
from ctypes import (
    CDLL, POINTER, Structure, cast, create_string_buffer, pointer, sizeof,
    c_float, c_int, c_int16, c_int32, c_uint8, c_uint32, c_void_p,
)
from io import BufferedIOBase, BytesIO
from typing import BinaryIO, Dict, List, Optional, Tuple, Union, Any

import numpy as np
from PIL import Image

__all__ = ['Thermal', 'parse_dji_rjpeg', 'apply_colormap']

# ---------------------------------------------------------------------------
# DJI SDK ctypes structs
# ---------------------------------------------------------------------------

DIRP_HANDLE = c_void_p
DIRP_VERBOSE_LEVEL_NONE = 0


class dirp_rjpeg_version_t(Structure):
    _fields_ = [('rjpeg', c_uint32), ('header', c_uint32), ('curve', c_uint32)]


class dirp_resolotion_t(Structure):
    _fields_ = [('width', c_uint32), ('height', c_uint32)]


class dirp_measurement_params_t(Structure):
    _fields_ = [
        ('distance', c_float),
        ('humidity', c_float),
        ('emissivity', c_float),
        ('reflection', c_float),
    ]


# ---------------------------------------------------------------------------
# FLIR APP1 parsing (pure Python)
# ---------------------------------------------------------------------------

SEGMENT_SEP = b'\xff'
APP1_MARKER = b'\xe1'
MAGIC_FLIR_DEF = b'FLIR\x00'
CHUNK_APP1_BYTES_COUNT = len(APP1_MARKER)
CHUNK_LENGTH_BYTES_COUNT = 2
CHUNK_MAGIC_BYTES_COUNT = len(MAGIC_FLIR_DEF)
CHUNK_SKIP_BYTES_COUNT = 1
CHUNK_NUM_BYTES_COUNT = 1
CHUNK_TOT_BYTES_COUNT = 1
CHUNK_METADATA_LENGTH = (
    CHUNK_APP1_BYTES_COUNT + CHUNK_LENGTH_BYTES_COUNT + CHUNK_MAGIC_BYTES_COUNT
    + CHUNK_SKIP_BYTES_COUNT + CHUNK_NUM_BYTES_COUNT + CHUNK_TOT_BYTES_COUNT
)


def _unpack_flir(path_or_stream: Union[str, BinaryIO]) -> np.ndarray:
    if isinstance(path_or_stream, str) and os.path.isfile(path_or_stream):
        with open(path_or_stream, 'rb') as f:
            return _unpack_flir(f)
    stream = path_or_stream
    flir_app1 = _extract_flir_app1(stream)
    records = _parse_flir_app1(flir_app1)
    return _parse_thermal(flir_app1, records)


def _extract_flir_app1(stream: BinaryIO) -> BinaryIO:
    stream.read(2)
    chunks_count: Optional[int] = None
    chunks: Dict[int, bytes] = {}
    while True:
        b = stream.read(1)
        if b == b'':
            break
        if b != SEGMENT_SEP:
            continue
        parsed = _parse_flir_chunk(stream, chunks_count)
        if not parsed:
            continue
        chunks_count, chunk_num, chunk = parsed
        if chunks.get(chunk_num) is not None:
            raise ValueError('Invalid FLIR: duplicate chunk number')
        chunks[chunk_num] = chunk
        if chunk_num == chunks_count:
            break
    if chunks_count is None:
        raise ValueError('Invalid FLIR: no metadata encountered')
    data = b''.join(chunks[i] for i in range(chunks_count + 1))
    s = BytesIO(data)
    s.seek(0)
    return s


def _parse_flir_chunk(
    stream: BinaryIO, chunks_count: Optional[int]
) -> Optional[Tuple[int, int, bytes]]:
    marker = stream.read(CHUNK_APP1_BYTES_COUNT)
    length_bytes = stream.read(CHUNK_LENGTH_BYTES_COUNT)
    length = int.from_bytes(length_bytes, 'big') - CHUNK_METADATA_LENGTH
    magic = stream.read(CHUNK_MAGIC_BYTES_COUNT)
    if not (marker == APP1_MARKER and magic == MAGIC_FLIR_DEF):
        stream.seek(-len(marker) - len(length_bytes) - len(magic), 1)
        return None
    stream.seek(1, 1)
    chunk_num = int.from_bytes(stream.read(CHUNK_NUM_BYTES_COUNT), 'big')
    chunks_tot = int.from_bytes(stream.read(CHUNK_TOT_BYTES_COUNT), 'big')
    if chunks_count is None:
        chunks_count = chunks_tot
    if chunk_num < 0 or chunk_num > chunks_tot or chunks_tot != chunks_count:
        raise ValueError(f'Invalid FLIR: inconsistent chunk count ({chunks_tot})')
    return chunks_tot, chunk_num, stream.read(length + 1)


def _parse_thermal(stream: BinaryIO, records: dict) -> np.ndarray:
    _, _, raw = _parse_raw_data(stream, records[1])
    return raw


def _parse_flir_app1(stream: BinaryIO) -> dict:
    stream.read(4)
    stream.seek(16, 1)
    stream.read(4)
    record_dir_offset = int.from_bytes(stream.read(4), 'big')
    record_dir_count = int.from_bytes(stream.read(4), 'big')
    stream.seek(28, 1)
    stream.read(4)
    stream.seek(record_dir_offset)
    stream.read(32 * record_dir_count)
    details: dict = {}
    for nr in range(record_dir_count):
        d = _parse_flir_record_metadata(stream, nr)
        if d:
            details[d[1]] = d
    return details


def _parse_flir_record_metadata(stream: BinaryIO, record_nr: int):
    stream.seek(32 * record_nr)
    rtype = int.from_bytes(stream.read(2), 'big')
    if rtype < 1:
        return None
    stream.read(2); stream.read(4); stream.read(4)
    offset = int.from_bytes(stream.read(4), 'big')
    length = int.from_bytes(stream.read(4), 'big')
    stream.read(4); stream.read(4); stream.read(4)
    return 32 * record_nr, rtype, offset, length


def _parse_raw_data(stream: BinaryIO, metadata: tuple):
    _, _, offset, length = metadata
    stream.seek(offset)
    stream.seek(2, 1)
    width = int.from_bytes(stream.read(2), 'little')
    height = int.from_bytes(stream.read(2), 'little')
    stream.seek(offset + 32)
    raw_bytes = stream.read(length)
    img = Image.open(BytesIO(raw_bytes))
    arr = np.array(img)
    if arr.shape != (height, width):
        raise ValueError(
            f'FLIR shape mismatch: metadata {(height, width)} vs data {arr.shape}'
        )
    if img.format == 'PNG':
        # PNG encodes uint16 big-endian; swap to little-endian sensor values
        arr = ((arr >> 8) | ((arr & 0x00FF) << 8)).astype(arr.dtype)
    return width, height, arr


def _read_flir_camera_params(stream: BinaryIO, records: dict) -> dict:
    """Read FLIR calibration parameters from FLIR APP1 record type 32.

    Offsets follow the FLIR FFF CameraParams binary layout (all float32 LE).
    Temperature fields are stored in Kelvin and converted to °C here.
    Humidity may be stored as a fraction (0-1) or percentage (>2); normalised
    to the 0-100 scale expected by :meth:`Thermal.parse_flir`.
    """
    rec = records.get(32)
    if rec is None:
        return {}
    _, _, offset, length = rec
    if length < 136:
        return {}
    stream.seek(offset)
    data = np.frombuffer(stream.read(length), dtype='<f4')
    if len(data) < 34:
        return {}

    hum_raw = float(data[7])  # offset 28
    return {
        'emissivity':                       float(data[0]),
        'object_distance':                  float(data[1]),
        'reflected_apparent_temperature':   float(data[2]) - 273.15,
        'atmospheric_temperature':          float(data[3]) - 273.15,
        'ir_window_temperature':            float(data[4]) - 273.15,
        'ir_window_transmission':           float(data[5]),
        'relative_humidity':                hum_raw if hum_raw > 2 else hum_raw * 100,
        'planck_r1':  float(data[24]),  # offset 96
        'planck_b':   float(data[25]),
        'planck_f':   float(data[26]),
        'planck_o':   float(data[27]),
        'planck_r2':  float(data[28]),
        'atx':        float(data[29]),
        'ata1':       float(data[30]),
        'ata2':       float(data[31]),
        'atb1':       float(data[32]),
        'atb2':       float(data[33]),
    }


# ---------------------------------------------------------------------------
# Pure-Python DJI R-JPEG extraction (no SDK required)
# ---------------------------------------------------------------------------

def _dji_block_score(arr: np.ndarray) -> float:
    """Score a uint16 array as potential DJI raw thermal data (Kelvin × 64).

    For −100 °C … 500 °C, raw values fall in ≈ [11 082, 49 482].
    Returns a value in [0.0, 1.0].  Penalises extreme outliers (values
    near 0 or near 65 535) which appear when garbage bytes are included
    at block boundaries.
    """
    in_range = float(np.mean((arr >= 11_000) & (arr <= 50_000)))
    if in_range < 0.90:
        return 0.0
    extreme = float(np.mean((arr < 500) | (arr > 64_500)))
    return in_range * max(0.0, 1.0 - extreme * 200.0)


# Candidate trailing-byte counts, ordered by likelihood.
# The DJI H20T R-JPEG appends a 512-byte footer after the thermal block;
# other models may differ — we score every candidate and pick the best.
_DJI_TRAILING_CANDIDATES = [512, 0, 256, 128, 768, 64, 1024, 32, 2048]


def parse_dji_rjpeg(filepath: str, dtype=np.float32) -> np.ndarray:
    """Extract temperature data from a DJI R-JPEG without the DJI SDK.

    DJI stores raw uint16 sensor values (Kelvin × 64) as a contiguous block
    at the END of the R-JPEG file, immediately before an optional proprietary
    footer.  Conversion formula:
        T [°C] = raw_uint16 / 64.0 − 273.15

    Supported cameras: M30T, H20T/ZH20T, M3T, M3TD, M4T, H30T, M2EA, H20N.
    For FLIR-based cameras use :meth:`Thermal.parse_flir` instead.
    For H20T / M3T which use a proprietary compressed format this fallback
    will fail — the DJI Thermal SDK is required.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    height, width = _jpeg_dimensions(data)
    n_pixels = height * width
    expected_bytes = n_pixels * 2

    if len(data) < expected_bytes:
        raise ValueError(
            f'File ({len(data)} B) is too small for a {width}×{height} '
            f'thermal payload ({expected_bytes} B).'
        )

    best_score = 0.0
    best_raw: Optional[np.ndarray] = None

    for trailing in _DJI_TRAILING_CANDIDATES:
        start = len(data) - expected_bytes - trailing
        if start < 0:
            continue
        block = np.frombuffer(data[start:start + expected_bytes], dtype=np.uint16)
        score = _dji_block_score(block)
        if score > best_score:
            best_score = score
            best_raw = block

    if best_raw is None or best_score == 0.0:
        raise ValueError(
            f"Could not locate the thermal payload in "
            f"'{os.path.basename(filepath)}'.\n\n"
            "The DJI H20T/ZH20T and similar cameras use a proprietary R-JPEG\n"
            "format that requires the official DJI Thermal SDK for reliable\n"
            "decoding.  Pass a valid sdk_dir to Thermal() to enable SDK support."
        )

    temp = best_raw.astype(np.float32) / 64.0 - 273.15
    arr2d = temp.reshape(height, width)
    if dtype == np.int16:
        return (arr2d * 10).astype(np.int16)
    return arr2d.astype(dtype)


def _jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Return (height, width) by scanning JPEG SOF markers."""
    i = 0
    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = int.from_bytes(data[i + 5:i + 7], 'big')
            w = int.from_bytes(data[i + 7:i + 9], 'big')
            return h, w
        if marker == 0xD8:
            i += 2
        elif marker in (0xD9, 0xDA):
            break
        else:
            seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
            i += 2 + seg_len
    raise ValueError('Could not determine JPEG image dimensions')


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

ABSOLUTE_ZERO = 273.15

_CMAP_ALIASES = {
    'white-hotspot': 'gray',
    'black-hotspot': 'gray_r',
}

COLORMAPS = [
    'white-hotspot',
    'black-hotspot',
    'plasma',
    'inferno',
    'magma',
    'viridis',
    'jet',
]


def apply_colormap(
    temp_array: np.ndarray,
    colormap: str = 'white-hotspot',
    lo_threshold: Optional[float] = None,
    hi_threshold: Optional[float] = None,
) -> Image.Image:
    """Convert a 2-D temperature array (°C) to an RGB PIL Image.

    Pixels outside [lo_threshold, hi_threshold] are rendered black.
    If both thresholds are None the full temperature range is used for
    normalisation and no pixels are masked.

    :param temp_array: H×W float32 array of temperatures in °C.
    :param colormap: One of :data:`COLORMAPS` or any matplotlib colormap name.
    :param lo_threshold: Lower clip temperature (°C). Pixels below → black.
    :param hi_threshold: Upper clip temperature (°C). Pixels above → black.
    :returns: PIL RGB image of the same spatial size.
    """
    import matplotlib as mpl

    arr = np.asarray(temp_array, dtype=np.float32)

    mask = np.zeros(arr.shape, dtype=bool)
    if lo_threshold is not None:
        mask |= arr < lo_threshold
    if hi_threshold is not None:
        mask |= arr > hi_threshold

    lo = lo_threshold if lo_threshold is not None else float(arr.min())
    hi = hi_threshold if hi_threshold is not None else float(arr.max())
    if hi <= lo:
        hi = lo + 1.0

    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    mpl_name = _CMAP_ALIASES.get(colormap, colormap)
    if hasattr(mpl, 'colormaps'):
        cmap = mpl.colormaps[mpl_name]
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(mpl_name)

    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb[mask] = 0
    return Image.fromarray(rgb, mode='RGB')


# ---------------------------------------------------------------------------
# DJI XMP metadata reader (pure Python, no exiftool)
# ---------------------------------------------------------------------------

_XMP_MARKER = b'http://ns.adobe.com/xap/1.0/\x00'


def _extract_jpeg_xmp(filepath: str) -> Optional[bytes]:
    """Scan all JPEG APP1 segments and return the XMP payload bytes, or None.

    Pillow only exposes the *first* APP1 segment via img.info; DJI writes EXIF
    as the first APP1 and XMP as the second, so img.info['xmp'] is always empty
    for DJI files.  This function scans the raw bytes directly.
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i < len(data) - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xDA:  # SOS — entropy-coded data begins; stop
            break
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
        if marker == 0xE1:  # APP1
            payload = data[i + 4: i + 2 + seg_len]
            if payload.startswith(_XMP_MARKER):
                return payload[len(_XMP_MARKER):]
        i += 2 + seg_len
    return None


_IIRP_SIG = b'iirp'
_IIRP_PARAMS_OFFSET = 32  # byte offset inside the APP4 payload where floats begin

# Confirmed offsets (float32 LE) for DJI M3T and compatible cameras:
#   +0  reflected apparent temperature (°C)
#   +4  object distance (m)
#   +8  emissivity (0–1)
#   +12 relative humidity (fraction 0–1; multiply by 100 to get %)
_IIRP_N_FLOATS = 4


def _read_dji_iirp_params(filepath: str) -> dict:
    """Read DJI thermal measurement parameters from the APP4 ``iirp`` binary block.

    DJI embeds an ``iirp`` (infrared image reference parameters) block in a
    JPEG APP4 segment.  Parameters are stored as four consecutive float32
    little-endian values starting at payload byte 32.

    Emissivity is already in the 0–1 range (unlike XMP, which stores it as a
    percentage).  Humidity is stored as a fraction (0–1) and is converted here
    to the percentage expected by :meth:`Thermal.parse_dirp2`.
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        i = 2
        while i < len(data) - 4:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xDA:
                break
            if marker in (0xD8, 0xD9):
                i += 2
                continue
            seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
            if marker == 0xE4:  # APP4
                payload = data[i + 4: i + 2 + seg_len]
                end = _IIRP_PARAMS_OFFSET + _IIRP_N_FLOATS * 4
                if len(payload) >= end and payload[4:8] == _IIRP_SIG:
                    floats = np.frombuffer(payload[_IIRP_PARAMS_OFFSET:end], dtype='<f4')
                    hum_raw = float(floats[3])
                    return {
                        'reflected_apparent_temperature': float(floats[0]),
                        'object_distance':                float(floats[1]),
                        'emissivity':                     float(floats[2]),
                        'relative_humidity': hum_raw * 100.0 if hum_raw <= 2.0 else hum_raw,
                    }
            i += 2 + seg_len
    except Exception:
        pass
    return {}


def _read_dji_xmp_params(filepath: str) -> dict:
    """Extract DJI thermal measurement parameters for *filepath*.

    Priority order:
    1. XMP ``drone-dji:SelfData`` / standard XMP tags (older DJI cameras).
    2. APP4 ``iirp`` binary block (M3T and other modern DJI thermal cameras).

    Emissivity from XMP is stored as a percentage (0–100) and is divided by
    100 here.  Emissivity from ``iirp`` is already in the 0–1 range.
    """
    import xml.etree.ElementTree as ET

    _measurement_keys = {
        'object_distance', 'relative_humidity',
        'emissivity', 'reflected_apparent_temperature',
    }

    try:
        with Image.open(filepath) as img:
            width, height = img.size

        result: Dict[str, Any] = {'image_width': width, 'image_height': height}

        xmp_bytes = _extract_jpeg_xmp(filepath)
        if xmp_bytes:
            xmp_str = xmp_bytes.decode('utf-8', errors='replace').rstrip('\x00')
            root = ET.fromstring(xmp_str)

            flat: Dict[str, str] = {}
            for elem in root.iter():
                local_tag = elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag
                if elem.text and elem.text.strip():
                    flat[local_tag] = elem.text.strip()
                for k, v in elem.attrib.items():
                    local_k = k.split('}', 1)[-1] if '}' in k else k
                    flat[local_k] = v

            _tag_candidates = {
                'object_distance':                ['CalibrateDistance', 'ObjectDistance'],
                'relative_humidity':              ['Humidity', 'RelativeHumidity'],
                'emissivity':                     ['Emissivity'],
                'reflected_apparent_temperature': ['ReflectTemperature', 'ReflectedTemperature',
                                                   'ReflectedApparentTemperature'],
            }
            for param, candidates in _tag_candidates.items():
                for tag in candidates:
                    if tag in flat:
                        nums = re.findall(r'[-\d]+\.?\d*', flat[tag])
                        if nums:
                            try:
                                result[param] = float(nums[0])
                                break
                            except ValueError:
                                pass

            if 'emissivity' in result:
                result['emissivity'] /= 100.0  # XMP stores as percentage

        # Fill any missing measurement params from the APP4 iirp block
        missing = _measurement_keys - set(result)
        if missing:
            iirp = _read_dji_iirp_params(filepath)
            for k in missing:
                if k in iirp:
                    result[k] = iirp[k]

        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main Thermal class
# ---------------------------------------------------------------------------

class Thermal:
    """Parse DJI and FLIR radiometric JPEG files into temperature arrays.

    :param dtype: Output dtype — ``np.float32`` (temperatures in °C, default)
        or ``np.int16`` (temperatures × 10 in tenths of °C).
    :param sdk_dir: Directory containing the DJI Thermal SDK shared libraries
        (``libdirp.dll`` / ``libdirp.so`` and siblings).  When ``None`` the
        SDK is not loaded; DJI cameras that require it will raise an error.
    :param exiftool_path: Accepted for backwards compatibility but no longer used.
        All metadata is now read via Pillow and pure-Python binary parsing.
    """

    DJI_XT2 = 'XT2'
    DJI_ZH20T = 'ZH20T'
    DJI_XTS = 'XT S'
    DJI_XTR = 'FLIR'
    FLIR_B60 = 'Flir b60'
    FLIR_E40 = 'FLIR E40'
    FLIR_T640 = 'FLIR T640'
    FLIR = 'FLIR'
    FLIR_DEFAULT = '*'
    FLIR_AX8 = 'FLIR AX8'
    DJI_M2EA = 'MAVIC2-ENTERPRISE-ADVANCED'
    DJI_H20N = 'ZH20N'
    DJI_M3T = 'M3T'
    DJI_M3TD = 'M3TD'
    DJI_M30T = 'M30T'
    DJI_H30T = 'H30T'
    DJI_M4T = 'M4T'

    DIRP_SUCCESS = 0

    def __init__(
        self,
        dtype=np.float32,
        sdk_dir: Optional[str] = None,
        exiftool_path: Optional[str] = None,
    ):
        if dtype.__name__ not in {np.float32.__name__, np.int16.__name__}:
            raise ValueError(f'dtype must be np.float32 or np.int16, got {dtype}')
        self._dtype = dtype
        self._filepath_exiftool = exiftool_path

        self._dll_dirp = None
        self._dll_dirp_sub = None
        self._dll_iirp = None
        self._sdk_loaded = False

        if sdk_dir is not None:
            ext = 'so' if platform.system() == 'Linux' else 'dll'
            self._filepath_dirp = os.path.join(sdk_dir, f'libdirp.{ext}')
            self._filepath_dirp_sub = os.path.join(sdk_dir, f'libv_dirp.{ext}')
            self._filepath_iirp = os.path.join(sdk_dir, f'libv_iirp.{ext}')
            self._load_sdk(sdk_dir)

    def _load_sdk(self, sdk_dir: str) -> None:
        """Load the DJI Thermal SDK DLLs from *sdk_dir*."""
        if not (os.path.isfile(self._filepath_dirp)
                and os.path.isfile(self._filepath_dirp_sub)
                and os.path.isfile(self._filepath_iirp)):
            return

        _dll_dir_cookie = None
        _old_ldpath = None
        try:
            if hasattr(os, 'add_dll_directory'):
                # Windows: register sdk_dir so the loader resolves transitive
                # dependencies (libv_girp.dll, MicroIA_*.dll, …).
                _dll_dir_cookie = os.add_dll_directory(sdk_dir)
            elif platform.system() == 'Linux':
                # Linux: prepend sdk_dir so dlopen() finds .so dependencies.
                # Restored in the finally block — the change is transient.
                _old_ldpath = os.environ.get('LD_LIBRARY_PATH', '')
                os.environ['LD_LIBRARY_PATH'] = (
                    sdk_dir + ':' + _old_ldpath if _old_ldpath else sdk_dir
                )
            self._dll_dirp = CDLL(self._filepath_dirp)
            self._dll_dirp_sub = CDLL(self._filepath_dirp_sub)
            self._dll_iirp = CDLL(self._filepath_iirp)
            self._sdk_loaded = True
            self._setup_sdk_functions()
        except OSError:
            self._sdk_loaded = False
            self._dll_dirp = None
        finally:
            if _dll_dir_cookie is not None:
                _dll_dir_cookie.close()
            if _old_ldpath is not None:
                os.environ['LD_LIBRARY_PATH'] = _old_ldpath

    def close(self) -> None:
        """Explicitly unload the SDK DLLs so the files are no longer locked."""
        if not self._sdk_loaded:
            return
        try:
            if platform.system() == 'Windows':
                import ctypes as _ct
                _free = _ct.windll.kernel32.FreeLibrary
                for dll in (self._dll_iirp, self._dll_dirp_sub, self._dll_dirp):
                    if dll is not None:
                        _free(dll._handle)
            else:
                import ctypes as _ct
                # CDLL(None) opens the current process image (libc on Linux),
                # which exports dlclose even on glibc >= 2.34 where libdl.so
                # is merged into libc.
                _dl = _ct.CDLL(None)
                for dll in (self._dll_iirp, self._dll_dirp_sub, self._dll_dirp):
                    if dll is not None:
                        _dl.dlclose(dll._handle)
        except Exception:
            pass
        self._dll_dirp = None
        self._dll_dirp_sub = None
        self._dll_iirp = None
        self._sdk_loaded = False

    def _setup_sdk_functions(self) -> None:
        dll = self._dll_dirp
        dll.dirp_set_verbose_level.argtypes = [c_int]
        dll.dirp_set_verbose_level(DIRP_VERBOSE_LEVEL_NONE)

        dll.dirp_create_from_rjpeg.argtypes = [POINTER(c_uint8), c_int32, POINTER(DIRP_HANDLE)]
        dll.dirp_create_from_rjpeg.restype = c_int32

        dll.dirp_destroy.argtypes = [DIRP_HANDLE]
        dll.dirp_destroy.restype = c_int32

        dll.dirp_get_rjpeg_version.argtypes = [DIRP_HANDLE, POINTER(dirp_rjpeg_version_t)]
        dll.dirp_get_rjpeg_version.restype = c_int32

        dll.dirp_get_rjpeg_resolution.argtypes = [DIRP_HANDLE, POINTER(dirp_resolotion_t)]
        dll.dirp_get_rjpeg_resolution.restype = c_int32

        dll.dirp_get_measurement_params.argtypes = [DIRP_HANDLE, POINTER(dirp_measurement_params_t)]
        dll.dirp_get_measurement_params.restype = c_int32

        dll.dirp_set_measurement_params.argtypes = [DIRP_HANDLE, POINTER(dirp_measurement_params_t)]
        dll.dirp_set_measurement_params.restype = c_int32

        dll.dirp_measure.argtypes = [DIRP_HANDLE, POINTER(c_int16), c_int32]
        dll.dirp_measure.restype = c_int32

        dll.dirp_measure_ex.argtypes = [DIRP_HANDLE, POINTER(c_float), c_int32]
        dll.dirp_measure_ex.restype = c_int32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, filepath_image: str) -> np.ndarray:
        """Parse a radiometric JPEG and return a 2-D temperature array (°C).

        Detects camera model from EXIF, then routes to the appropriate decoder.
        Falls back to speculative SDK decoding or the pure-Python R-JPEG extractor
        when the model is unrecognised.
        """
        if not isinstance(filepath_image, str) or not os.path.exists(filepath_image):
            raise FileNotFoundError(f'File not found: {filepath_image}')

        camera_model = self._get_camera_model(filepath_image)

        _flir_models = {
            self.FLIR, self.FLIR_DEFAULT, self.FLIR_T640,
            self.FLIR_E40, self.FLIR_B60, self.FLIR_AX8,
            self.DJI_XT2, self.DJI_XTR,
        }
        _dji_sdk_models = {
            self.DJI_ZH20T, self.DJI_XTS,
            self.DJI_M2EA, self.DJI_H20N,
            self.DJI_M3T, self.DJI_M3TD, self.DJI_M30T,
            self.DJI_H30T, self.DJI_M4T,
        }

        if camera_model and (camera_model in _flir_models or self.FLIR in camera_model):
            kwargs = self._extract_flir_params(filepath_image)
            return self.parse_flir(filepath_image, **kwargs)

        if camera_model and camera_model in _dji_sdk_models:
            if self._sdk_loaded:
                kwargs = self._extract_dji_params(filepath_image, camera_model)
                return self.parse_dirp2(filepath_image, **kwargs)
            else:
                return parse_dji_rjpeg(filepath_image, dtype=self._dtype)

        # Unknown / undetectable model: try SDK speculatively, then pure-Python.
        if self._sdk_loaded:
            for _m2ea in (True, False):
                try:
                    return self.parse_dirp2(filepath_image, m2ea_mode=_m2ea)
                except Exception:
                    pass

        raise ValueError(
            f"Cannot decode '{os.path.basename(filepath_image)}'.\n\n"
            "DJI thermal cameras (H20T, M3T, M3TD, M30T, …) store temperature\n"
            "data in a proprietary compressed format that requires the DJI\n"
            "Thermal SDK.  Construct Thermal(sdk_dir=...) with the path to\n"
            "the SDK DLL directory to enable full decoding support."
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _get_camera_model(self, filepath: str) -> Optional[str]:
        try:
            from bambi.util.image_utils import get_exif_data
            return get_exif_data(filepath).get('Model') or None
        except Exception:
            return None

    def _extract_flir_params(self, filepath: str) -> dict:
        try:
            with open(filepath, 'rb') as f:
                flir_app1 = _extract_flir_app1(f)
            records = _parse_flir_app1(flir_app1)
            params = _read_flir_camera_params(flir_app1, records)
            if params:
                return params
        except Exception:
            pass
        return {}

    def _extract_dji_params(self, filepath: str, camera_model: str) -> dict:
        kwargs = _read_dji_xmp_params(filepath)
        if camera_model == self.DJI_M30T:
            kwargs.pop('image_width', None)
            kwargs.pop('image_height', None)
        if camera_model in {
            self.DJI_M2EA, self.DJI_H20N, self.DJI_M3T, self.DJI_M3TD,
            self.DJI_M30T, self.DJI_H30T, self.DJI_M4T,
        }:
            kwargs['m2ea_mode'] = True
        return kwargs

    # ------------------------------------------------------------------
    # FLIR Planck conversion
    # ------------------------------------------------------------------

    def parse_flir(
        self,
        filepath_image: str,
        emissivity: float = 1.0,
        object_distance: float = 1.0,
        atmospheric_temperature: float = 20.0,
        reflected_apparent_temperature: float = 20.0,
        ir_window_temperature: float = 20.0,
        ir_window_transmission: float = 1.0,
        relative_humidity: float = 50.0,
        planck_r1: float = 21106.77,
        planck_b: float = 1501.0,
        planck_f: float = 1.0,
        planck_o: float = -7340.0,
        planck_r2: float = 0.012545258,
        ata1: float = 0.006569,
        ata2: float = 0.01262,
        atb1: float = -0.002276,
        atb2: float = -0.00667,
        atx: float = 1.9,
    ) -> np.ndarray:
        raw = _unpack_flir(filepath_image)

        emiss_wind = 1 - ir_window_transmission
        refl_wind = 0.0
        h2o = (relative_humidity / 100) * np.exp(
            1.5587 + 0.06939 * atmospheric_temperature
            - 0.00027816 * atmospheric_temperature ** 2
            + 0.00000068455 * atmospheric_temperature ** 3
        )
        tau1 = atx * np.exp(-np.sqrt(object_distance / 2) * (ata1 + atb1 * np.sqrt(h2o))) + (
            1 - atx
        ) * np.exp(-np.sqrt(object_distance / 2) * (ata2 + atb2 * np.sqrt(h2o)))
        tau2 = tau1

        def _rad(T):
            return planck_r1 / (planck_r2 * (np.exp(planck_b / (T + ABSOLUTE_ZERO)) - planck_f)) - planck_o

        raw_refl1_attn = (1 - emissivity) / emissivity * _rad(reflected_apparent_temperature)
        raw_atm1_attn = (1 - tau1) / emissivity / tau1 * _rad(atmospheric_temperature)
        raw_wind_attn = emiss_wind / emissivity / tau1 / ir_window_transmission * _rad(ir_window_temperature)
        raw_refl2_attn = refl_wind / emissivity / tau1 / ir_window_transmission * _rad(reflected_apparent_temperature)
        raw_atm2_attn = (1 - tau2) / emissivity / tau1 / ir_window_transmission / tau2 * _rad(atmospheric_temperature)

        raw_obj = (
            raw / emissivity / tau1 / ir_window_transmission / tau2
            - raw_atm1_attn - raw_atm2_attn - raw_wind_attn
            - raw_refl1_attn - raw_refl2_attn
        )
        val = planck_r1 / (planck_r2 * (raw_obj + planck_o)) + planck_f
        if np.any(val <= 0):
            raise ValueError(
                f'Planck formula produced non-positive values — file may be corrupt: {filepath_image}'
            )
        temperature = planck_b / np.log(val) - ABSOLUTE_ZERO
        return temperature.astype(self._dtype)

    # ------------------------------------------------------------------
    # DJI SDK path (requires libdirp DLLs)
    # ------------------------------------------------------------------

    def parse_dirp2(
        self,
        filepath_image: str,
        image_height: int = 512,
        image_width: int = 640,
        object_distance: float = 5.0,
        relative_humidity: float = 70.0,
        emissivity: float = 1.0,
        reflected_apparent_temperature: float = 23.0,
        m2ea_mode: bool = False,
    ) -> np.ndarray:
        if not self._sdk_loaded:
            raise RuntimeError(
                "DJI SDK DLLs are not loaded.  Construct Thermal(sdk_dir=...) "
                "with the path to the SDK DLL directory."
            )
        with open(filepath_image, 'rb') as f:
            raw_bytes = f.read()
        raw_buf = create_string_buffer(raw_bytes, len(raw_bytes))
        raw_size = c_int32(len(raw_bytes))
        raw_ptr = cast(raw_buf, POINTER(c_uint8))
        handle = DIRP_HANDLE()
        dll = self._dll_dirp

        ret = dll.dirp_create_from_rjpeg(raw_ptr, raw_size, pointer(handle))
        if ret != self.DIRP_SUCCESS:
            raise RuntimeError(f'dirp_create_from_rjpeg failed: {ret}')

        ver = dirp_rjpeg_version_t()
        res = dirp_resolotion_t()
        if dll.dirp_get_rjpeg_version(handle, pointer(ver)) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_get_rjpeg_version failed')
        if dll.dirp_get_rjpeg_resolution(handle, pointer(res)) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_get_rjpeg_resolution failed')
        sdk_w = int(res.width) or image_width
        sdk_h = int(res.height) or image_height

        if not m2ea_mode:
            params = dirp_measurement_params_t()
            if dll.dirp_get_measurement_params(handle, pointer(params)) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_get_measurement_params failed')
            params.distance = float(object_distance)
            params.humidity = float(relative_humidity)
            params.emissivity = float(emissivity)
            params.reflection = float(reflected_apparent_temperature)
            if dll.dirp_set_measurement_params(handle, pointer(params)) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_set_measurement_params failed')

        n = sdk_w * sdk_h
        if self._dtype.__name__ == np.float32.__name__:
            data = np.zeros(n, dtype=np.float32)
            ptr = data.ctypes.data_as(POINTER(c_float))
            if dll.dirp_measure_ex(handle, ptr, c_int32(n * sizeof(c_float))) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_measure_ex failed')
            temp = data.reshape(sdk_h, sdk_w)
        else:
            data = np.zeros(n, dtype=np.int16)
            ptr = data.ctypes.data_as(POINTER(c_int16))
            if dll.dirp_measure(handle, ptr, c_int32(n * sizeof(c_int16))) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_measure failed')
            temp = data.reshape(sdk_h, sdk_w) / 10.0

        if dll.dirp_destroy(handle) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_destroy failed')

        return np.array(temp, dtype=self._dtype)
