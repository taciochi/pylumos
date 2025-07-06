from dataclasses import dataclass


@dataclass
class SlicingPattern:
    start_row: int
    start_column: int
    step: int
