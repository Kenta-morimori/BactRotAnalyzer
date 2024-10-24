from dataclasses import dataclass
from typing import Final, List


@dataclass
class ROTATION_FEATURES:
    rot_long_axis: str = "rot_long_axis"
    rot_short_axis: str = "rot_short_axis"
    angle_FFT_peak: str = "angle_FFT_peak"
    angular_velosity_mean: str = "angular_velosity_mean"
    SD_FFT_Amp_decrease: str = "SD_FFT_Amp_decrease"
    SD_window_data_num_mean: str = "SD_window_data_num_mean"
    SD_FFT_Amp_refpoints: str = "SD_FFT_Amp_refpoints"


SD_WIDTH_DEPEND_COLS: Final[List[str]] = [
    ROTATION_FEATURES.SD_FFT_Amp_decrease,
    ROTATION_FEATURES.SD_window_data_num_mean,
]
