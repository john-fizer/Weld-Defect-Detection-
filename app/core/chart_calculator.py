"""
Core chart calculation engine using Swiss Ephemeris
Handles natal chart calculations with multiple house and zodiac systems
"""
import swisseph as swe
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pytz
from pathlib import Path
from app.config import settings


# Planet constants
PLANETS = {
    'Sun': swe.SUN,
    'Moon': swe.MOON,
    'Mercury': swe.MERCURY,
    'Venus': swe.VENUS,
    'Mars': swe.MARS,
    'Jupiter': swe.JUPITER,
    'Saturn': swe.SATURN,
    'Uranus': swe.URANUS,
    'Neptune': swe.NEPTUNE,
    'Pluto': swe.PLUTO,
    'North Node': swe.TRUE_NODE,
    'Chiron': swe.CHIRON,
}

# House system mapping
HOUSE_SYSTEMS = {
    'placidus': b'P',
    'koch': b'K',
    'equal': b'E',
    'whole_sign': b'W',
    'vedic_whole': b'W',  # Whole sign with sidereal zodiac
}

# Zodiac signs
SIGNS = [
    'Aries', 'Taurus', 'Gemini', 'Cancer',
    'Leo', 'Virgo', 'Libra', 'Scorpio',
    'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
]

# Aspect constants
ASPECTS = {
    'conjunction': (0, 8),
    'sextile': (60, 6),
    'square': (90, 8),
    'trine': (120, 8),
    'opposition': (180, 8),
}


class ChartCalculator:
    """Main chart calculation engine"""

    def __init__(self):
        """Initialize Swiss Ephemeris"""
        # Set ephemeris path
        ephe_path = Path(settings.ephemeris_path)
        if ephe_path.exists():
            swe.set_ephe_path(str(ephe_path))
        else:
            # Use default Swiss Ephemeris data
            swe.set_ephe_path(None)

    def calculate_natal_chart(
        self,
        birth_datetime: datetime,
        latitude: float,
        longitude: float,
        house_system: str = 'placidus',
        zodiac_system: str = 'tropical'
    ) -> Dict:
        """
        Calculate complete natal chart

        Args:
            birth_datetime: Birth date and time (timezone aware)
            latitude: Birth location latitude
            longitude: Birth location longitude
            house_system: House system to use
            zodiac_system: 'tropical' or 'sidereal'

        Returns:
            Dictionary containing planets, houses, aspects, and additional data
        """
        # Convert datetime to Julian Day
        jd = self._datetime_to_jd(birth_datetime)

        # Set sidereal mode if needed
        if zodiac_system == 'sidereal':
            swe.set_sid_mode(swe.SIDM_LAHIRI)  # Lahiri ayanamsa

        # Calculate planetary positions
        planets = self._calculate_planets(jd, zodiac_system)

        # Calculate houses
        houses = self._calculate_houses(jd, latitude, longitude, house_system, zodiac_system)

        # Assign planets to houses
        planets = self._assign_houses_to_planets(planets, houses, house_system)

        # Calculate aspects
        aspects = self._calculate_aspects(planets)

        return {
            'planets': planets,
            'houses': houses,
            'aspects': aspects,
            'julian_day': jd,
            'house_system': house_system,
            'zodiac_system': zodiac_system,
        }

    def _datetime_to_jd(self, dt: datetime) -> float:
        """Convert datetime to Julian Day"""
        # Ensure datetime is in UTC
        if dt.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")

        dt_utc = dt.astimezone(pytz.UTC)

        jd = swe.julday(
            dt_utc.year,
            dt_utc.month,
            dt_utc.day,
            dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
        )
        return jd

    def _calculate_planets(self, jd: float, zodiac_system: str) -> Dict:
        """Calculate all planetary positions"""
        planets_data = {}
        flag = swe.FLG_SIDEREAL if zodiac_system == 'sidereal' else swe.FLG_SWIEPH

        for planet_name, planet_id in PLANETS.items():
            try:
                result, ret = swe.calc_ut(jd, planet_id, flag)

                longitude = result[0]
                latitude = result[1]
                speed = result[3]

                # Determine sign and degree
                sign_num = int(longitude / 30)
                degree_in_sign = longitude % 30

                planets_data[planet_name] = {
                    'longitude': longitude,
                    'latitude': latitude,
                    'speed': speed,
                    'sign': SIGNS[sign_num],
                    'sign_num': sign_num,
                    'degree': degree_in_sign,
                    'degree_absolute': longitude,
                    'retrograde': speed < 0,
                    'house': None,  # Will be assigned later
                }

            except Exception as e:
                print(f"Error calculating {planet_name}: {e}")
                continue

        return planets_data

    def _calculate_houses(
        self,
        jd: float,
        latitude: float,
        longitude: float,
        house_system: str,
        zodiac_system: str
    ) -> Dict:
        """Calculate house cusps and angles"""
        house_flag = HOUSE_SYSTEMS.get(house_system, b'P')

        # Calculate houses
        cusps, ascmc = swe.houses(jd, latitude, longitude, house_flag)

        # Handle sidereal zodiac
        if zodiac_system == 'sidereal':
            ayanamsa = swe.get_ayanamsa_ut(jd)
            cusps = [(c - ayanamsa) % 360 for c in cusps]
            ascmc = [(a - ayanamsa) % 360 for a in ascmc]

        houses_data = {}

        # Store house cusps
        for i, cusp in enumerate(cusps[1:13], start=1):  # Houses 1-12
            sign_num = int(cusp / 30)
            degree_in_sign = cusp % 30

            houses_data[i] = {
                'cusp': cusp,
                'sign': SIGNS[sign_num],
                'sign_num': sign_num,
                'degree': degree_in_sign,
            }

        # Store angles (ASC, MC, etc.)
        houses_data['angles'] = {
            'ASC': ascmc[0],
            'MC': ascmc[1],
            'ARMC': ascmc[2],
            'Vertex': ascmc[3],
        }

        return houses_data

    def _assign_houses_to_planets(
        self,
        planets: Dict,
        houses: Dict,
        house_system: str
    ) -> Dict:
        """Assign house placement to each planet"""

        if house_system == 'whole_sign' or house_system == 'vedic_whole':
            # Whole sign houses: based on sign of Ascendant
            asc_sign = int(houses['angles']['ASC'] / 30)

            for planet_name, planet_data in planets.items():
                planet_sign = planet_data['sign_num']
                # Calculate house number
                house_num = ((planet_sign - asc_sign) % 12) + 1
                planet_data['house'] = house_num
        else:
            # Quadrant houses: based on house cusps
            for planet_name, planet_data in planets.items():
                planet_long = planet_data['longitude']
                house_num = self._find_house_for_longitude(planet_long, houses)
                planet_data['house'] = house_num

        return planets

    def _find_house_for_longitude(self, longitude: float, houses: Dict) -> int:
        """Find which house a longitude falls into (for quadrant systems)"""
        longitude = longitude % 360

        for house_num in range(1, 13):
            current_cusp = houses[house_num]['cusp']
            next_house = (house_num % 12) + 1
            next_cusp = houses[next_house]['cusp']

            # Handle wrapping around 360
            if next_cusp < current_cusp:
                if longitude >= current_cusp or longitude < next_cusp:
                    return house_num
            else:
                if current_cusp <= longitude < next_cusp:
                    return house_num

        return 1  # Default to 1st house if not found

    def _calculate_aspects(self, planets: Dict) -> List[Dict]:
        """Calculate aspects between planets"""
        aspects_list = []
        planet_names = list(planets.keys())

        for i, planet1_name in enumerate(planet_names):
            for planet2_name in planet_names[i + 1:]:
                planet1_long = planets[planet1_name]['longitude']
                planet2_long = planets[planet2_name]['longitude']

                # Calculate angular separation
                separation = abs(planet1_long - planet2_long)
                if separation > 180:
                    separation = 360 - separation

                # Check for aspects
                for aspect_name, (angle, orb) in ASPECTS.items():
                    diff = abs(separation - angle)
                    if diff <= orb:
                        aspects_list.append({
                            'planet1': planet1_name,
                            'planet2': planet2_name,
                            'aspect': aspect_name,
                            'angle': angle,
                            'orb': diff,
                            'applying': self._is_applying(
                                planets[planet1_name],
                                planets[planet2_name],
                                angle
                            ),
                        })

        return aspects_list

    def _is_applying(self, planet1: Dict, planet2: Dict, aspect_angle: float) -> bool:
        """Determine if aspect is applying or separating"""
        # Simplified: compare speeds
        # A more accurate version would check actual motion
        if planet1['speed'] > planet2['speed']:
            return True
        return False

    def calculate_decans(self, planets: Dict) -> Dict:
        """
        Calculate decan placements for planets
        Each sign has 3 decans (10 degrees each)
        """
        decan_rulers = {
            # Fire triplicity (Aries, Leo, Sagittarius)
            'Aries': ['Mars', 'Sun', 'Jupiter'],
            'Leo': ['Sun', 'Jupiter', 'Mars'],
            'Sagittarius': ['Jupiter', 'Mars', 'Sun'],
            # Earth triplicity
            'Taurus': ['Venus', 'Mercury', 'Saturn'],
            'Virgo': ['Mercury', 'Saturn', 'Venus'],
            'Capricorn': ['Saturn', 'Venus', 'Mercury'],
            # Air triplicity
            'Gemini': ['Mercury', 'Venus', 'Saturn'],
            'Libra': ['Venus', 'Saturn', 'Mercury'],
            'Aquarius': ['Saturn', 'Mercury', 'Venus'],
            # Water triplicity
            'Cancer': ['Moon', 'Mars', 'Jupiter'],
            'Scorpio': ['Mars', 'Jupiter', 'Moon'],
            'Pisces': ['Jupiter', 'Moon', 'Mars'],
        }

        decans_data = {}
        for planet_name, planet_data in planets.items():
            degree = planet_data['degree']
            sign = planet_data['sign']

            # Determine decan (0-9.99 = 1st, 10-19.99 = 2nd, 20-29.99 = 3rd)
            decan_num = int(degree / 10)
            decan_ruler = decan_rulers[sign][decan_num]

            decans_data[planet_name] = {
                'decan': decan_num + 1,
                'ruler': decan_ruler,
                'degree_in_decan': degree % 10,
            }

        return decans_data

    def close(self):
        """Clean up Swiss Ephemeris resources"""
        swe.close()
