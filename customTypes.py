from dataclasses import dataclass
from typing import List
from collections import namedtuple

from torch import Tensor
import torch

def customWeatherDecoder(weatherDict):
    return namedtuple('X', weatherDict.keys())(*weatherDict.values())

@dataclass
class HourlyUnits:
    time: str
    temperature_2m: str
    precipitation: str
    cloud_cover: str
    sunshine_duration: str
    def to_dict(self):
        return {
            "time": self.time,
            "temperature_2m": self.temperature_2m,
            "precipitation": self.precipitation,
            "cloud_cover": self.cloud_cover,
            "sunshine_duration": self.sunshine_duration
        }

@dataclass
class Hourly:
    time: List[str]
    temperature_2m: List[float]
    precipitation: List[float]
    cloud_cover: List[int]
    sunshine_duration: List[float]
    def to_dict(self):
        return {
            "time": self.time,
            "temperature_2m": self.temperature_2m,
            "precipitation": self.precipitation,
            "cloud_cover": self.cloud_cover,
            "sunshine_duration": self.sunshine_duration
        }

@dataclass
class DailyUnits:
    time: str
    sunrise: str
    sunset: str
    sunshine_duration: str

    def to_dict(self):
        return {
            "time": self.time,
            "sunrise": self.sunrise,
            "sunset": self.sunset,
            "sunshine_duration": self.sunshine_duration
        }


@dataclass
class Daily:
    time: List[str]
    sunrise: List[str]
    sunset: List[str]
    sunshine_duration: List[float]
    def to_dict(self):
        return {
            "time": self.time,
            "sunrise": self.sunrise,
            "sunset": self.sunset,
            "sunshine_duration": self.sunshine_duration
        }

@dataclass
class WeatherData:
    latitude: float
    longitude: float
    generationtime_ms: float
    utc_offset_seconds: int
    timezone: str
    timezone_abbreviation: str
    elevation: float
    hourly_units: HourlyUnits
    hourly: Hourly
    daily_units: DailyUnits
    daily: Daily
    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "generationtime_ms": self.generationtime_ms,
            "utc_offset_seconds": self.utc_offset_seconds,
            "timezone": self.timezone,
            "timezone_abbreviation": self.timezone_abbreviation,
            "elevation": self.elevation,
            "hourly_units": self.hourly_units.to_dict(),
            "hourly": self.hourly.to_dict(),
            "daily_units": self.daily_units.to_dict(),
            "daily": self.daily.to_dict()
        }



@dataclass
class DocumentDesciption():
    kalenderwoche: int
    jahr: int
    starttag: int
    endtag: int
    
@dataclass
class DocumentList:
    docs:list[DocumentDesciption]

def weather_dict_to_feature_vec(data:dict)-> Tensor:
   tensor = torch.tensor((), dtype=torch.int32)
   tensor.new_ones((2, 3))
   return tensor


