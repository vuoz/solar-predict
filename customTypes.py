from dataclasses import dataclass

@dataclass
class DocumentDesciption():
    kalenderwoche: int
    jahr: int
    starttag: int
    endtag: int
    
@dataclass
class DocumentList:
    docs:list[DocumentDesciption]


@dataclass
class MinMaxWeather():
    power_production: tuple[float,float]
    percipitation: tuple[float,float]
    temp: tuple[float,float]
    cloud_cover: tuple[float,float]
    wind_speed: tuple[float,float]
    irradiance: tuple[float,float]
    humidity: tuple[float,float]
    diffuse_radiation: tuple[float,float]
    direct_normal_irradiance: tuple[float,float]
    diffuse_radiation_instant: tuple[float,float]
    direct_normal_irradiance_instant: tuple[float,float]
    global_tilted_irradiance_instant: tuple[float,float]

@dataclass
class MinMaxSeasons():
    winter: MinMaxWeather
    spring: MinMaxWeather
    summer: MinMaxWeather
    autumn: MinMaxWeather
