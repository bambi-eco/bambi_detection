# -*- coding: utf-8 -*-
"""
Thermal image parser for DJI radiometric JPEGs.

Based on SanNianYiSi/thermal_parser (MIT licence).

The DJI SDK DLL directory (https://www.dji.com/at/downloads/softwares/dji-thermal-sdk)
is supplied explicitly via the constructor.  The caller is responsible for locating
the SDK on the current system and passing the path in.

Metadata (camera model, DJI measurement params) is read
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
from typing import Dict, Optional, Tuple, Any

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
    """Parse DJI radiometric JPEG files into temperature arrays.

    :param dtype: Output dtype — ``np.float32`` (temperatures in °C, default)
        or ``np.int16`` (temperatures × 10 in tenths of °C).
    :param sdk_dir: Directory containing the DJI Thermal SDK shared libraries
        (``libdirp.dll`` / ``libdirp.so`` and siblings).  When ``None`` the
        SDK is not loaded; DJI cameras that require it will raise an error.
    :param exiftool_path: Accepted for backwards compatibility but no longer used.
        All metadata is now read via Pillow and pure-Python binary parsing.
    """

    DJI_ZH20T = 'ZH20T'
    DJI_XTS = 'XT S'
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
        """Parse a DJI radiometric JPEG and return a 2-D temperature array (°C).

        Detects camera model from EXIF, then routes to the appropriate decoder.
        Falls back to speculative SDK decoding or the pure-Python R-JPEG extractor
        when the model is unrecognised.
        """
        if not isinstance(filepath_image, str) or not os.path.exists(filepath_image):
            raise FileNotFoundError(f'File not found: {filepath_image}')

        camera_model = self._get_camera_model(filepath_image)

        _dji_sdk_models = {
            self.DJI_ZH20T, self.DJI_XTS,
            self.DJI_M2EA, self.DJI_H20N,
            self.DJI_M3T, self.DJI_M3TD, self.DJI_M30T,
            self.DJI_H30T, self.DJI_M4T,
        }

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
