from macrec.systems.base import System
from macrec.systems.rewoo import ReWOOSystem

SYSTEMS: list[type[System]] = [value for value in globals().values() if isinstance(value, type) and issubclass(value, System) and value != System]
