"""
Vedic astrology calculations including Nakshatras
"""
from typing import Dict, Tuple


# Nakshatra data: (name, ruling planet, degrees start, degrees end)
NAKSHATRAS = [
    ('Ashwini', 'Ketu', 0.0, 13.333333),
    ('Bharani', 'Venus', 13.333333, 26.666667),
    ('Krittika', 'Sun', 26.666667, 40.0),
    ('Rohini', 'Moon', 40.0, 53.333333),
    ('Mrigashira', 'Mars', 53.333333, 66.666667),
    ('Ardra', 'Rahu', 66.666667, 80.0),
    ('Punarvasu', 'Jupiter', 80.0, 93.333333),
    ('Pushya', 'Saturn', 93.333333, 106.666667),
    ('Ashlesha', 'Mercury', 106.666667, 120.0),
    ('Magha', 'Ketu', 120.0, 133.333333),
    ('Purva Phalguni', 'Venus', 133.333333, 146.666667),
    ('Uttara Phalguni', 'Sun', 146.666667, 160.0),
    ('Hasta', 'Moon', 160.0, 173.333333),
    ('Chitra', 'Mars', 173.333333, 186.666667),
    ('Swati', 'Rahu', 186.666667, 200.0),
    ('Vishakha', 'Jupiter', 200.0, 213.333333),
    ('Anuradha', 'Saturn', 213.333333, 226.666667),
    ('Jyeshtha', 'Mercury', 226.666667, 240.0),
    ('Mula', 'Ketu', 240.0, 253.333333),
    ('Purva Ashadha', 'Venus', 253.333333, 266.666667),
    ('Uttara Ashadha', 'Sun', 266.666667, 280.0),
    ('Shravana', 'Moon', 280.0, 293.333333),
    ('Dhanishta', 'Mars', 293.333333, 306.666667),
    ('Shatabhisha', 'Rahu', 306.666667, 320.0),
    ('Purva Bhadrapada', 'Jupiter', 320.0, 333.333333),
    ('Uttara Bhadrapada', 'Saturn', 333.333333, 346.666667),
    ('Revati', 'Mercury', 346.666667, 360.0),
]

# Nakshatra pada rulership (each nakshatra has 4 padas)
# Pada rulers follow the order: Fire, Earth, Air, Water signs
PADA_SIGNS = {
    1: 'Fire',
    2: 'Earth',
    3: 'Air',
    4: 'Water',
}


class VedicCalculator:
    """Vedic astrology calculations"""

    def calculate_nakshatras(self, planets: Dict) -> Dict:
        """
        Calculate Nakshatra placements for all planets

        Args:
            planets: Dictionary of planet data with sidereal longitudes

        Returns:
            Dictionary with nakshatra data for each planet
        """
        nakshatras_data = {}

        for planet_name, planet_data in planets.items():
            longitude = planet_data['longitude'] % 360

            nakshatra_info = self._get_nakshatra(longitude)
            nakshatras_data[planet_name] = nakshatra_info

        return nakshatras_data

    def _get_nakshatra(self, longitude: float) -> Dict:
        """
        Get Nakshatra information for a given longitude

        Args:
            longitude: Sidereal longitude (0-360)

        Returns:
            Dictionary with nakshatra name, ruler, pada, etc.
        """
        # Each nakshatra is 13°20' (13.333333°)
        nakshatra_length = 360 / 27  # 13.333333

        # Find which nakshatra
        nakshatra_num = int(longitude / nakshatra_length)
        if nakshatra_num >= 27:
            nakshatra_num = 26

        nakshatra_name, ruler, start_deg, end_deg = NAKSHATRAS[nakshatra_num]

        # Calculate position within nakshatra
        degree_in_nakshatra = longitude - (nakshatra_num * nakshatra_length)

        # Calculate pada (each pada is 3°20' or 3.333333°)
        pada_length = nakshatra_length / 4
        pada = int(degree_in_nakshatra / pada_length) + 1
        if pada > 4:
            pada = 4

        degree_in_pada = degree_in_nakshatra % pada_length

        return {
            'nakshatra': nakshatra_name,
            'nakshatra_num': nakshatra_num + 1,
            'ruler': ruler,
            'pada': pada,
            'pada_element': PADA_SIGNS[pada],
            'degree_in_nakshatra': degree_in_nakshatra,
            'degree_in_pada': degree_in_pada,
        }

    def calculate_vimshottari_dasha(
        self,
        moon_longitude: float,
        birth_datetime
    ) -> Dict:
        """
        Calculate Vimshottari Dasha periods based on Moon's nakshatra

        Args:
            moon_longitude: Sidereal longitude of Moon
            birth_datetime: Birth datetime

        Returns:
            Dictionary with dasha periods
        """
        # Dasha periods in years
        dasha_periods = {
            'Ketu': 7,
            'Venus': 20,
            'Sun': 6,
            'Moon': 10,
            'Mars': 7,
            'Rahu': 18,
            'Jupiter': 16,
            'Saturn': 19,
            'Mercury': 17,
        }

        # Dasha sequence
        dasha_sequence = ['Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury']

        # Get Moon's nakshatra
        moon_nak = self._get_nakshatra(moon_longitude)
        ruler = moon_nak['ruler']

        # Find starting dasha
        start_index = dasha_sequence.index(ruler)

        # Calculate balance of first dasha based on Moon's position in nakshatra
        degree_in_nak = moon_nak['degree_in_nakshatra']
        nakshatra_length = 360 / 27
        proportion_completed = degree_in_nak / nakshatra_length
        first_dasha_years = dasha_periods[ruler]
        balance_years = first_dasha_years * (1 - proportion_completed)

        # Build dasha timeline
        dashas = []
        current_date = birth_datetime
        years_elapsed = balance_years

        # First dasha (partial)
        dashas.append({
            'planet': ruler,
            'start_date': current_date,
            'duration_years': balance_years,
        })

        # Subsequent dashas
        for i in range(1, len(dasha_sequence)):
            planet = dasha_sequence[(start_index + i) % len(dasha_sequence)]
            duration = dasha_periods[planet]

            # Calculate approximate start date
            from datetime import timedelta
            start_date = birth_datetime + timedelta(days=years_elapsed * 365.25)

            dashas.append({
                'planet': planet,
                'start_date': start_date,
                'duration_years': duration,
            })

            years_elapsed += duration

        return {
            'current_dasha': ruler,
            'balance_years': balance_years,
            'dashas': dashas,
        }

    def calculate_yogas(self, planets: Dict, houses: Dict) -> list:
        """
        Calculate Vedic yogas (planetary combinations)

        Args:
            planets: Planet positions
            houses: House data

        Returns:
            List of identified yogas
        """
        yogas = []

        # This is a simplified version - real yoga calculation is complex
        # Here we'll identify some basic yogas

        # Raja Yoga: Angular house lords in combination
        # Dhana Yoga: Wealth combinations
        # etc.

        # Example: Check for planets in kendras (1, 4, 7, 10)
        kendras = [1, 4, 7, 10]
        kendra_planets = []

        for planet_name, planet_data in planets.items():
            if planet_data.get('house') in kendras:
                kendra_planets.append(planet_name)

        if len(kendra_planets) >= 2:
            yogas.append({
                'name': 'Kendra Yoga',
                'description': f'Multiple planets in angular houses: {", ".join(kendra_planets)}',
                'planets': kendra_planets,
            })

        # Check for planets in trines (1, 5, 9)
        trikonas = [1, 5, 9]
        trikona_planets = []

        for planet_name, planet_data in planets.items():
            if planet_data.get('house') in trikonas:
                trikona_planets.append(planet_name)

        if len(trikona_planets) >= 2:
            yogas.append({
                'name': 'Trikona Yoga',
                'description': f'Multiple planets in trine houses: {", ".join(trikona_planets)}',
                'planets': trikona_planets,
            })

        return yogas

    def get_planetary_strength(self, planet_data: Dict) -> Dict:
        """
        Calculate Shadbala (six-fold strength) - simplified version

        Args:
            planet_data: Planet position data

        Returns:
            Dictionary with strength metrics
        """
        # This is a very simplified version
        # Real Shadbala calculation is extremely complex

        strength = {
            'exalted': self._is_exalted(planet_data),
            'debilitated': self._is_debilitated(planet_data),
            'own_sign': self._is_in_own_sign(planet_data),
        }

        return strength

    def _is_exalted(self, planet_data: Dict) -> bool:
        """Check if planet is in exaltation"""
        exaltations = {
            'Sun': 'Aries',
            'Moon': 'Taurus',
            'Mercury': 'Virgo',
            'Venus': 'Pisces',
            'Mars': 'Capricorn',
            'Jupiter': 'Cancer',
            'Saturn': 'Libra',
        }
        # Simplified - just checking sign
        return planet_data.get('sign') == exaltations.get(planet_data.get('planet_name'))

    def _is_debilitated(self, planet_data: Dict) -> bool:
        """Check if planet is debilitated"""
        debilitations = {
            'Sun': 'Libra',
            'Moon': 'Scorpio',
            'Mercury': 'Pisces',
            'Venus': 'Virgo',
            'Mars': 'Cancer',
            'Jupiter': 'Capricorn',
            'Saturn': 'Aries',
        }
        return planet_data.get('sign') == debilitations.get(planet_data.get('planet_name'))

    def _is_in_own_sign(self, planet_data: Dict) -> bool:
        """Check if planet is in own sign"""
        rulerships = {
            'Sun': ['Leo'],
            'Moon': ['Cancer'],
            'Mercury': ['Gemini', 'Virgo'],
            'Venus': ['Taurus', 'Libra'],
            'Mars': ['Aries', 'Scorpio'],
            'Jupiter': ['Sagittarius', 'Pisces'],
            'Saturn': ['Capricorn', 'Aquarius'],
        }
        planet_name = planet_data.get('planet_name', '')
        return planet_data.get('sign') in rulerships.get(planet_name, [])
