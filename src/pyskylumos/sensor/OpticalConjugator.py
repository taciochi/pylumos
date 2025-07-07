from typing import Tuple, Optional

from numpy.typing import NDArray
from numpy import linspace, ones, angle, arctan, pi, absolute, arcsin, rad2deg, float32, flip


class OpticalConjugator:
    __lens_conjugation_type: str
    __number_pixels_vertical: int
    __number_pixels_horizontal: int
    __lens_focal_length_micrometers: float
    __sensor_pixel_size_square_micrometers: float

    def __init__(
            self,
            lens_conjugation_type: str,
            number_pixels_vertical: int,
            number_pixels_horizontal: int,
            lens_focal_length_micrometers: float,
            sensor_pixel_size_square_micrometers: float
    ) -> None:
        self.__lens_conjugation_type = lens_conjugation_type
        self.__number_pixels_vertical = number_pixels_vertical
        self.__number_pixels_horizontal = number_pixels_horizontal
        self.__lens_focal_length_micrometers = lens_focal_length_micrometers
        self.__sensor_pixel_size_square_micrometers = sensor_pixel_size_square_micrometers

    @property
    def lens_conjugation_type(self) -> str:
        return self.__lens_conjugation_type

    @property
    def sensor_pixel_size_square_micrometers(self) -> float:
        return self.__sensor_pixel_size_square_micrometers

    # noinspection PyTypeChecker
    def __get_complex_sensor_plane(
            self,
    ) -> NDArray[complex]:
        start_x: float = (self.__number_pixels_horizontal - 1) / 2
        stop_x: float = -start_x
        x_pixels: NDArray[float32] = linspace(start=start_x, stop=stop_x,
                                              num=self.__number_pixels_horizontal).astype(float32)
        del start_x, stop_x

        start_y: float = (self.__number_pixels_vertical - 1) / 2
        stop_y: float = -start_y
        y_pixels: NDArray[float32] = linspace(start=start_y, stop=stop_y,
                                              num=self.__number_pixels_vertical).astype(float32)
        del start_y, stop_y

        x_micrometers: NDArray[float32] = self.__sensor_pixel_size_square_micrometers * x_pixels
        y_micrometers: NDArray[float32] = self.__sensor_pixel_size_square_micrometers * y_pixels

        real: NDArray[float32] = ones(shape=(self.__number_pixels_vertical, 1)) * x_micrometers
        imaginary: NDArray[float32] = ones(shape=(1, self.__number_pixels_horizontal)) * y_micrometers.T[:, None]
        complex_plane: NDArray[complex] = real + 1j * imaginary

        return complex_plane

    def __apply_conjugation(
            self,
            complex_sensor_plane: NDArray[complex],
            custom_lens_conjugation: Optional[callable]
    ) -> NDArray[float32]:
        half_pi: float = pi / 2
        match self.__lens_conjugation_type:
            case 'thin':
                return half_pi - arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                ).astype(float32)
            case 'stereographic':
                return half_pi - 2 * arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                ).astype(float32)
            case 'equi_angle':
                return half_pi - (absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers).astype(float32)
            case 'equi_solid_angle':
                return half_pi - 2 * arcsin(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                ).astype(float32)
            case 'orthogonal':
                return half_pi - arcsin(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                )
            case 'custom':
                return custom_lens_conjugation(
                    complex_sensor_plane=complex_sensor_plane,
                    lens_focal_length_micrometers=self.__lens_focal_length_micrometers
                )
            case _:
                exit('Invalid lens projection type')

    def get_azimuth_altitude(
            self,
            altitude_min_clip: Optional[float],
            custom_lens_conjugation: Optional[callable] = None,
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        complex_sensor_plane: NDArray[complex] = self.__get_complex_sensor_plane()
        azimuth: NDArray[float32] = angle(z=complex_sensor_plane, deg=True)
        altitude: NDArray[float32] = rad2deg(
            self.__apply_conjugation(
                complex_sensor_plane=complex_sensor_plane,
                custom_lens_conjugation=custom_lens_conjugation
            )
        )

        if altitude_min_clip is not None:
            altitude = altitude.clip(min=altitude_min_clip)

        return flip(azimuth, axis=1), altitude
