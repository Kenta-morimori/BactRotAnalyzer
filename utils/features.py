from dataclasses import dataclass
from typing import Final, List


@dataclass
class ROTATION_FEATURES:
    rot_long_axis: str = "rot_long_axis"
    rot_short_axis: str = "rot_short_axis"
    rot_aspect_ratio: str = "rot_aspect_ratio"
    angle_FFT_peak: str = "angle_FFT_peak"
    rot_angular_velosity_th: str = "rot_angular_velosity_th"
    angular_velosity_mean_rot_part: str = "angular_velosity_mean_rot_part"
    angular_velosity_sd_rot_part: str = "angular_velosity_sd_rot_part"
    angular_velosity_mean: str = "angular_velosity_mean"
    angular_velosity_sd: str = "angular_velosity_sd"
    SD_FFT_Amp_decrease: str = "SD_FFT_Amp_decrease"
    SD_window_data_num_mean: str = "SD_window_data_num_mean"
    SD_FFT_Amp_refpoints: str = "SD_FFT_Amp_refpoints"


SD_WIDTH_DEPEND_COLS: Final[List[str]] = [
    ROTATION_FEATURES.SD_FFT_Amp_decrease,
    ROTATION_FEATURES.SD_window_data_num_mean,
]


IGNORE_PLOT_COLS: Final[List[str]] = [
    ROTATION_FEATURES.rot_angular_velosity_th,
]
