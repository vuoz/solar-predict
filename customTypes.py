from dataclasses import dataclass
from typing import List
from collections import namedtuple

def customWeatherDecoder(weatherDict):
    return namedtuple('X', weatherDict.keys())(*weatherDict.values())

@dataclass
class HourlyUnits:
    time: str
    temperature_2m: str
    precipitation: str
    cloud_cover: str
    sunshine_duration: str

@dataclass
class Hourly:
    time: List[str]
    temperature_2m: List[float]
    precipitation: List[float]
    cloud_cover: List[int]
    sunshine_duration: List[float]

@dataclass
class DailyUnits:
    time: str
    sunrise: str
    sunset: str
    sunshine_duration: str

@dataclass
class Daily:
    time: List[str]
    sunrise: List[str]
    sunset: List[str]
    sunshine_duration: List[float]

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



@dataclass
class DocumentDesciption():
    kalenderwoche: int
    jahr: int
    starttag: int
    endtag: int
    
@dataclass
class DocumentList:
    docs:list[DocumentDesciption]

