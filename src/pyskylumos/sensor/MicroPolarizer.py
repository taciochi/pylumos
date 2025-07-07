from typing import Dict

from numpy.typing import NDArray
from numpy import float32, random, zeros_like, deg2rad, cos

from src.pyskylumos.sensor.SlicingPattern import SlicingPattern


class MicroPolarizer:
    __extinction_ratio: float
    __tolerance: float
    __wire_grid_orientations_slicing: Dict[int, SlicingPattern]

    def __init__(
            self,
            extinction_ratio: float,
            tolerance: float,
            wire_grid_orientations_slicing: Dict[int, SlicingPattern]
    ) -> None:
        """
        :param extinction_ratio: The polarizer's extinction ratio (e.g., 0.99).
        :param tolerance: Max random angular offset (in radians or degrees, see usage) for manufacturing defects.
        :param wire_grid_orientations_slicing: Dictionary specifying which rows/cols correspond to each orientation angle.
        """
        self.__tolerance = tolerance
        self.__extinction_ratio = extinction_ratio
        self.__wire_grid_orientations_slicing = wire_grid_orientations_slicing

    def get_intensity_on_pixel(
            self,
            degree_of_polarization: NDArray[NDArray[float32]],
            angle_of_polarization: NDArray[NDArray[float32]],
            radiance: NDArray[NDArray[float32]]
    ) -> NDArray[float32]:
        angle_map = zeros_like(radiance, dtype=float32)

        # 2) Fill angle_map using the SlicingPattern dictionary
        #    The dictionary keys (orientation_angle) are presumably in degrees, so convert to radians as needed.
        for orientation_angle_deg, slicing_pattern in self.__wire_grid_orientations_slicing.items():
            orientation_angle_rad = deg2rad(orientation_angle_deg)  # convert to radians
            angle_map[
            :,
            slicing_pattern.start_row::slicing_pattern.step,
            slicing_pattern.start_column::slicing_pattern.step
            ] = orientation_angle_rad

        # 3) Optionally add random "defects" to simulate slight orientation errors
        #    If self.__tolerance is in degrees, you'll need deg2rad(...) here.
        #    If it is already in radians, use it directly. Example below assumes it is in radians.
        time_dim, row_dim, col_dim = radiance.shape
        # random array in [-1, +1], scaled by tolerance
        defects = (self.__tolerance * (1 - 2 * random.rand(row_dim, col_dim).astype(float32)))[None, :, :]
        # broadcast defects across time dimension
        angle_map += defects

        # 4) Compute intensity: I = 0.5 * radiance * [1 + (extinction_ratio * DoP) * cos(2 * (AoP - angle_map))]
        #    Note AoP and angle_map are in radians.
        intensity_on_pixel = 0.5 * radiance * (
                1.0 + (
                self.__extinction_ratio * degree_of_polarization
        ) * cos(
            2.0 * (angle_of_polarization - angle_map)
        )
        )

        return intensity_on_pixel
