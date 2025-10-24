"""
Secondary Progressions calculation and prediction engine
"""
import swisseph as swe
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz
from app.core.chart_calculator import ChartCalculator, PLANETS, ASPECTS


class ProgressionsEngine:
    """Calculate secondary progressions and predict events"""

    def __init__(self):
        self.chart_calc = ChartCalculator()

    def calculate_progressed_chart(
        self,
        birth_datetime: datetime,
        target_date: datetime,
        latitude: float,
        longitude: float,
        house_system: str = 'placidus',
        zodiac_system: str = 'tropical'
    ) -> Dict:
        """
        Calculate progressed chart for a given date
        Secondary progressions: 1 day after birth = 1 year of life

        Args:
            birth_datetime: Original birth datetime
            target_date: Date to calculate progressions for
            latitude: Birth latitude
            longitude: Birth longitude
            house_system: House system to use
            zodiac_system: Tropical or sidereal

        Returns:
            Progressed chart data
        """
        # Calculate age in years
        age_delta = target_date - birth_datetime
        age_years = age_delta.total_seconds() / (365.25 * 24 * 3600)

        # Progressed date = birth + age in days
        progressed_datetime = birth_datetime + timedelta(days=age_years)

        # Calculate chart for progressed date
        progressed_chart = self.chart_calc.calculate_natal_chart(
            progressed_datetime,
            latitude,
            longitude,
            house_system,
            zodiac_system
        )

        progressed_chart['age_years'] = age_years
        progressed_chart['target_date'] = target_date
        progressed_chart['progressed_date'] = progressed_datetime

        return progressed_chart

    def calculate_progressed_to_natal_aspects(
        self,
        natal_chart: Dict,
        progressed_chart: Dict,
        orb_override: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Calculate aspects between progressed planets and natal planets

        Args:
            natal_chart: Natal chart data
            progressed_chart: Progressed chart data
            orb_override: Optional custom orbs for aspects

        Returns:
            List of progressed-to-natal aspects
        """
        aspects = []

        # Use custom orbs or default to tighter orbs for progressions
        orbs = orb_override or {
            'conjunction': 1.0,
            'sextile': 1.0,
            'square': 1.0,
            'trine': 1.0,
            'opposition': 1.0,
        }

        natal_planets = natal_chart['planets']
        progressed_planets = progressed_chart['planets']

        for prog_name, prog_data in progressed_planets.items():
            for natal_name, natal_data in natal_planets.items():
                prog_long = prog_data['longitude']
                natal_long = natal_data['longitude']

                # Calculate angular separation
                separation = abs(prog_long - natal_long)
                if separation > 180:
                    separation = 360 - separation

                # Check for aspects
                for aspect_name, (aspect_angle, _) in ASPECTS.items():
                    orb = orbs.get(aspect_name, 1.0)
                    diff = abs(separation - aspect_angle)

                    if diff <= orb:
                        aspects.append({
                            'progressed_planet': prog_name,
                            'natal_planet': natal_name,
                            'aspect': aspect_name,
                            'angle': aspect_angle,
                            'orb': diff,
                            'exact_date': self._calculate_exact_aspect_date(
                                natal_chart,
                                prog_name,
                                natal_name,
                                aspect_angle,
                                progressed_chart['target_date']
                            ),
                        })

        return aspects

    def _calculate_exact_aspect_date(
        self,
        natal_chart: Dict,
        prog_planet: str,
        natal_planet: str,
        aspect_angle: float,
        current_date: datetime
    ) -> Optional[datetime]:
        """
        Calculate approximate date when progressed aspect becomes exact
        This is a simplified version - real calculation would need iteration
        """
        # This would require iterative calculation
        # For now, return None or approximate based on current separation
        return None

    def predict_progression_events(
        self,
        natal_chart: Dict,
        start_date: datetime,
        end_date: datetime,
        latitude: float,
        longitude: float
    ) -> List[Dict]:
        """
        Predict events based on secondary progressions over a date range

        Args:
            natal_chart: Natal chart data
            start_date: Start of prediction period
            end_date: End of prediction period
            latitude: Birth latitude
            longitude: Birth longitude

        Returns:
            List of predicted events with dates
        """
        predictions = []

        # Sample key dates throughout the period
        # In production, would calculate continuously
        current_date = start_date
        delta = timedelta(days=30)  # Check every month

        while current_date <= end_date:
            prog_chart = self.calculate_progressed_chart(
                natal_chart['birth_datetime'],
                current_date,
                latitude,
                longitude
            )

            aspects = self.calculate_progressed_to_natal_aspects(natal_chart, prog_chart)

            # Analyze significant aspects
            for aspect in aspects:
                # Major aspects to Sun, Moon, ASC, MC are most significant
                if aspect['natal_planet'] in ['Sun', 'Moon'] or aspect['orb'] < 0.5:
                    prediction = self._interpret_progression_aspect(aspect, current_date)
                    if prediction:
                        predictions.append(prediction)

            current_date += delta

        return predictions

    def _interpret_progression_aspect(self, aspect: Dict, date: datetime) -> Optional[Dict]:
        """
        Interpret a progressed aspect and predict event type

        Args:
            aspect: Aspect data
            date: Date of aspect

        Returns:
            Prediction dictionary or None
        """
        # Simplified interpretation logic
        # Real version would use more sophisticated rules and ML

        prog_planet = aspect['progressed_planet']
        natal_planet = aspect['natal_planet']
        aspect_type = aspect['aspect']

        # Event category mapping (simplified)
        event_categories = {
            ('Sun', 'conjunction'): 'career_promotion',
            ('Sun', 'square'): 'career_termination',
            ('Venus', 'conjunction'): 'relationship_start',
            ('Venus', 'square'): 'relationship_breakup',
            ('Mars', 'square'): 'personal_health',
            ('Jupiter', 'trine'): 'financial_windfall',
        }

        key = (natal_planet, aspect_type)
        event_category = event_categories.get(key)

        if event_category:
            return {
                'date': date,
                'event_category': event_category,
                'description': f'Progressed {prog_planet} {aspect_type} natal {natal_planet}',
                'confidence': 0.6,  # Base confidence
                'aspect_data': aspect,
            }

        return None

    def calculate_progressed_moon_phase(self, progressed_chart: Dict, natal_chart: Dict) -> Dict:
        """
        Calculate progressed Moon phase relative to progressed Sun
        Progressed New Moon/Full Moon are significant timing indicators
        """
        prog_sun = progressed_chart['planets']['Sun']['longitude']
        prog_moon = progressed_chart['planets']['Moon']['longitude']

        phase_angle = (prog_moon - prog_sun) % 360

        # Determine phase
        if 0 <= phase_angle < 45:
            phase_name = 'New Moon'
        elif 45 <= phase_angle < 90:
            phase_name = 'Waxing Crescent'
        elif 90 <= phase_angle < 135:
            phase_name = 'First Quarter'
        elif 135 <= phase_angle < 180:
            phase_name = 'Waxing Gibbous'
        elif 180 <= phase_angle < 225:
            phase_name = 'Full Moon'
        elif 225 <= phase_angle < 270:
            phase_name = 'Waning Gibbous'
        elif 270 <= phase_angle < 315:
            phase_name = 'Last Quarter'
        else:
            phase_name = 'Waning Crescent'

        return {
            'phase_angle': phase_angle,
            'phase_name': phase_name,
            'is_new_moon': 0 <= phase_angle < 15,
            'is_full_moon': 165 <= phase_angle < 195,
        }
