"""Recognize compact tire sizes shared by alignment and duration safeguards."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TireSize:
    width: int
    aspect: int
    construction: str
    rim: int

    @property
    def part_units(self) -> tuple[int, int]:
        def short_number_units(value: int) -> int:
            return 1 if value < 20 or value % 10 == 0 else 2

        remainder = self.width % 100
        width_units = (
            2 if remainder == 0 else (3 if remainder < 10 else 1 + short_number_units(remainder))
        )
        return (
            width_units + short_number_units(self.aspect),
            len(self.construction) + short_number_units(self.rim),
        )

    @property
    def spoken_units(self) -> int:
        return sum(self.part_units)


def parse_tire_size(text: str) -> TireSize | None:
    """Require the full size syntax, so ranges, years and model IDs stay distinct."""
    core = text.strip().strip(".,;:!?()[]{}\"'，。！？；：")
    match = re.fullmatch(r"(\d{3})[/-](\d{2})\s*(ZR|R)(\d{2})", core, re.IGNORECASE)
    if not match:
        return None
    width, aspect, construction, rim = match.groups()
    size = TireSize(int(width), int(aspect), construction.upper(), int(rim))
    if not (
        125 <= size.width <= 395
        and size.width % 5 == 0
        and 20 <= size.aspect <= 95
        and size.aspect % 5 == 0
        and 10 <= size.rim <= 30
    ):
        return None
    return size
