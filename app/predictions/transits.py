"""
Transit calculation and prediction engine
"""
import swisseph as swe
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.core.chart_calculator import ChartCalculator, PLANETS, ASPECTS


class TransitsEngine:
    """Calculate transits and predict events"""

    def __init__(self):
        self.chart_calc = ChartCalculator()

    def calculate_current_transits(
        self,
        current_datetime: datetime,
        house_system: str = 'placidus',
        zodiac_system: str = 'tropical'
    ) -> Dict:
        """
        Calculate current planetary positions (transits)

        Args:
            current_datetime: Current date/time
            house_system: House system
            zodiac_system: Tropical or sidereal

        Returns:
            Current planetary positions
        """
        # For transits, we don't need location-specific houses
        # Just planetary positions
        jd = self.chart_calc._datetime_to_jd(current_datetime)

        flag = swe.FLG_SIDEREAL if zodiac_system == 'sidereal' else swe.FLG_SWIEPH
        if zodiac_system == 'sidereal':
            swe.set_sid_mode(swe.SIDM_LAHIRI)

        planets = {}
        for planet_name, planet_id in PLANETS.items():
            try:
                result, ret = swe.calc_ut(jd, planet_id, flag)

                longitude = result[0]
                speed = result[3]

                sign_num = int(longitude / 30)
                degree_in_sign = longitude % 30

                from app.core.chart_calculator import SIGNS
                planets[planet_name] = {
                    'longitude': longitude,
                    'speed': speed,
                    'sign': SIGNS[sign_num],
                    'sign_num': sign_num,
                    'degree': degree_in_sign,
                    'retrograde': speed < 0,
                }

            except Exception as e:
                print(f"Error calculating transit for {planet_name}: {e}")
                continue

        return {
            'planets': planets,
            'datetime': current_datetime,
            'julian_day': jd,
        }

    def calculate_transit_to_natal_aspects(
        self,
        natal_chart: Dict,
        transit_datetime: datetime,
        orb_override: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Calculate aspects between transiting planets and natal planets

        Args:
            natal_chart: Natal chart data
            transit_datetime: Date to calculate transits for
            orb_override: Optional custom orbs

        Returns:
            List of transit-to-natal aspects
        """
        transits = self.calculate_current_transits(transit_datetime)
        aspects = []

        # Tighter orbs for transits
        orbs = orb_override or {
            'conjunction': 2.0,
            'sextile': 2.0,
            'square': 2.0,
            'trine': 2.0,
            'opposition': 2.0,
        }

        natal_planets = natal_chart['planets']
        transit_planets = transits['planets']

        for transit_name, transit_data in transit_planets.items():
            for natal_name, natal_data in natal_planets.items():
                transit_long = transit_data['longitude']
                natal_long = natal_data['longitude']

                separation = abs(transit_long - natal_long)
                if separation > 180:
                    separation = 360 - separation

                for aspect_name, (aspect_angle, _) in ASPECTS.items():
                    orb = orbs.get(aspect_name, 2.0)
                    diff = abs(separation - aspect_angle)

                    if diff <= orb:
                        aspects.append({
                            'transiting_planet': transit_name,
                            'natal_planet': natal_name,
                            'aspect': aspect_name,
                            'angle': aspect_angle,
                            'orb': diff,
                            'is_applying': transit_data['speed'] > 0,
                            'is_retrograde': transit_data['retrograde'],
                            'exact_date': None,  # Would calculate in production
                        })

        return aspects

    def calculate_transit_to_houses(
        self,
        natal_chart: Dict,
        transit_datetime: datetime
    ) -> List[Dict]:
        """
        Calculate which natal houses transiting planets are in

        Args:
            natal_chart: Natal chart with houses
            transit_datetime: Transit date

        Returns:
            List of transit house positions
        """
        transits = self.calculate_current_transits(transit_datetime)
        houses = natal_chart.get('houses', {})

        transit_houses = []

        for planet_name, planet_data in transits['planets'].items():
            planet_long = planet_data['longitude']

            # Find house
            house_num = self.chart_calc._find_house_for_longitude(planet_long, houses)

            transit_houses.append({
                'planet': planet_name,
                'house': house_num,
                'longitude': planet_long,
                'sign': planet_data['sign'],
            })

        return transit_houses

    def predict_transit_events(
        self,
        natal_chart: Dict,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Predict events based on transits over a date range

        Args:
            natal_chart: Natal chart data
            start_date: Start of prediction period
            end_date: End of prediction period

        Returns:
            List of predicted events
        """
        predictions = []

        # Sample dates throughout period
        current_date = start_date
        delta = timedelta(days=1)  # Check daily for transits

        while current_date <= end_date:
            aspects = self.calculate_transit_to_natal_aspects(natal_chart, current_date)

            # Focus on slow-moving outer planet transits (more significant)
            outer_planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

            for aspect in aspects:
                if aspect['transiting_planet'] in outer_planets:
                    # Tight orb = more significant
                    if aspect['orb'] < 1.0:
                        prediction = self._interpret_transit_aspect(aspect, current_date)
                        if prediction:
                            predictions.append(prediction)

            current_date += delta

        return predictions

    def _interpret_transit_aspect(self, aspect: Dict, date: datetime) -> Optional[Dict]:
        """
        Interpret a transit aspect and predict event type

        Args:
            aspect: Aspect data
            date: Date of aspect

        Returns:
            Prediction dictionary or None
        """
        transit_planet = aspect['transiting_planet']
        natal_planet = aspect['natal_planet']
        aspect_type = aspect['aspect']

        # Event category mapping (simplified)
        # Real version would be much more sophisticated
        event_categories = {
            ('Saturn', 'Sun', 'conjunction'): 'career_termination',
            ('Saturn', 'Sun', 'square'): 'career_termination',
            ('Jupiter', 'Sun', 'conjunction'): 'career_promotion',
            ('Jupiter', 'Sun', 'trine'): 'career_promotion',
            ('Uranus', 'Sun', 'conjunction'): 'career_new_job',
            ('Uranus', 'Sun', 'square'): 'career_new_job',
            ('Neptune', 'Venus', 'conjunction'): 'relationship_start',
            ('Saturn', 'Venus', 'square'): 'relationship_breakup',
            ('Pluto', 'Sun', 'conjunction'): 'personal_spiritual',
            ('Jupiter', 'Jupiter', 'conjunction'): 'financial_windfall',  # Jupiter return
        }

        key = (transit_planet, natal_planet, aspect_type)
        event_category = event_categories.get(key)

        if event_category:
            # Calculate confidence based on orb
            confidence = 0.7 if aspect['orb'] < 0.5 else 0.5

            # Retrograde transits may indicate delays
            if aspect['is_retrograde']:
                confidence *= 0.8

            return {
                'date': date,
                'event_category': event_category,
                'description': f'Transit {transit_planet} {aspect_type} natal {natal_planet}',
                'confidence': confidence,
                'aspect_data': aspect,
            }

        return None

    def find_exact_transit_dates(
        self,
        transiting_planet: str,
        natal_planet_longitude: float,
        aspect_angle: float,
        start_date: datetime,
        end_date: datetime,
        zodiac_system: str = 'tropical'
    ) -> List[datetime]:
        """
        Find exact dates when a transit aspect becomes exact
        Handles retrograde motion (can have up to 3 exact hits)

        Args:
            transiting_planet: Name of transiting planet
            natal_planet_longitude: Longitude of natal planet
            aspect_angle: Aspect angle (0, 60, 90, 120, 180)
            start_date: Start search date
            end_date: End search date
            zodiac_system: Tropical or sidereal

        Returns:
            List of exact aspect dates
        """
        exact_dates = []

        # Target longitude for exact aspect
        target_longitude = (natal_planet_longitude + aspect_angle) % 360

        # Sample daily
        current_date = start_date
        previous_separation = None

        while current_date <= end_date:
            transits = self.calculate_current_transits(current_date, zodiac_system=zodiac_system)

            if transiting_planet in transits['planets']:
                transit_long = transits['planets'][transiting_planet]['longitude']

                # Calculate separation from target
                separation = abs(transit_long - target_longitude)
                if separation > 180:
                    separation = 360 - separation

                # Check if we crossed exact aspect
                if previous_separation is not None:
                    if previous_separation > separation and separation < 1.0:
                        # Found exact aspect (or very close)
                        exact_dates.append(current_date)

                previous_separation = separation

            current_date += timedelta(days=1)

        return exact_dates
