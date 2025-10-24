"""
Zodiacal Releasing (ZR) - Hellenistic timing technique
Based on ancient time-lord systems, primarily from Vettius Valens
"""
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# ZR period lengths in years
ZR_PERIODS = {
    'Aries': 15,
    'Taurus': 8,
    'Gemini': 20,
    'Cancer': 25,
    'Leo': 19,
    'Virgo': 20,
    'Libra': 8,
    'Scorpio': 15,
    'Sagittarius': 12,
    'Capricorn': 27,
    'Aquarius': 30,
    'Pisces': 12,
}

# Planetary rulers for ZR
ZR_RULERS = {
    'Aries': 'Mars',
    'Taurus': 'Venus',
    'Gemini': 'Mercury',
    'Cancer': 'Moon',
    'Leo': 'Sun',
    'Virgo': 'Mercury',
    'Libra': 'Venus',
    'Scorpio': 'Mars',
    'Sagittarius': 'Jupiter',
    'Capricorn': 'Saturn',
    'Aquarius': 'Saturn',
    'Pisces': 'Jupiter',
}


class ZodiacalReleasingEngine:
    """Calculate Zodiacal Releasing periods and predict peak periods"""

    def __init__(self):
        self.signs = list(ZR_PERIODS.keys())

    def calculate_zr_from_lot_of_fortune(
        self,
        natal_chart: Dict,
        birth_datetime: datetime,
        is_day_chart: Optional[bool] = None
    ) -> Dict:
        """
        Calculate ZR from the Lot of Fortune
        This is the most common starting point for ZR

        Args:
            natal_chart: Natal chart data
            birth_datetime: Birth datetime
            is_day_chart: True if day chart (Sun above horizon), False if night

        Returns:
            ZR periods and timeline
        """
        # Calculate Lot of Fortune
        lot_of_fortune = self._calculate_lot_of_fortune(natal_chart, is_day_chart)

        # Get ZR starting sign
        start_sign = lot_of_fortune['sign']

        # Calculate full ZR timeline
        timeline = self._calculate_zr_timeline(start_sign, birth_datetime)

        return {
            'lot_of_fortune': lot_of_fortune,
            'start_sign': start_sign,
            'timeline': timeline,
        }

    def calculate_zr_from_lot_of_spirit(
        self,
        natal_chart: Dict,
        birth_datetime: datetime,
        is_day_chart: Optional[bool] = None
    ) -> Dict:
        """
        Calculate ZR from the Lot of Spirit
        Used for career and livelihood timing

        Args:
            natal_chart: Natal chart data
            birth_datetime: Birth datetime
            is_day_chart: True if day chart, False if night

        Returns:
            ZR periods and timeline
        """
        lot_of_spirit = self._calculate_lot_of_spirit(natal_chart, is_day_chart)
        start_sign = lot_of_spirit['sign']
        timeline = self._calculate_zr_timeline(start_sign, birth_datetime)

        return {
            'lot_of_spirit': lot_of_spirit,
            'start_sign': start_sign,
            'timeline': timeline,
        }

    def _calculate_lot_of_fortune(
        self,
        natal_chart: Dict,
        is_day_chart: Optional[bool] = None
    ) -> Dict:
        """
        Calculate Lot of Fortune
        Day: ASC + Moon - Sun
        Night: ASC + Sun - Moon
        """
        planets = natal_chart['planets']
        asc = natal_chart['houses']['angles']['ASC']
        sun = planets['Sun']['longitude']
        moon = planets['Moon']['longitude']

        # Determine day/night if not provided
        if is_day_chart is None:
            # Day chart if Sun is above horizon (houses 7-12)
            sun_house = planets['Sun'].get('house', 1)
            is_day_chart = sun_house in [7, 8, 9, 10, 11, 12]

        # Calculate Lot of Fortune
        if is_day_chart:
            lot = (asc + moon - sun) % 360
        else:
            lot = (asc + sun - moon) % 360

        sign_num = int(lot / 30)
        sign = self.signs[sign_num]

        return {
            'longitude': lot,
            'sign': sign,
            'sign_num': sign_num,
            'degree': lot % 30,
        }

    def _calculate_lot_of_spirit(
        self,
        natal_chart: Dict,
        is_day_chart: Optional[bool] = None
    ) -> Dict:
        """
        Calculate Lot of Spirit
        Day: ASC + Sun - Moon
        Night: ASC + Moon - Sun
        """
        planets = natal_chart['planets']
        asc = natal_chart['houses']['angles']['ASC']
        sun = planets['Sun']['longitude']
        moon = planets['Moon']['longitude']

        if is_day_chart is None:
            sun_house = planets['Sun'].get('house', 1)
            is_day_chart = sun_house in [7, 8, 9, 10, 11, 12]

        # Lot of Spirit is opposite calculation from Fortune
        if is_day_chart:
            lot = (asc + sun - moon) % 360
        else:
            lot = (asc + moon - sun) % 360

        sign_num = int(lot / 30)
        sign = self.signs[sign_num]

        return {
            'longitude': lot,
            'sign': sign,
            'sign_num': sign_num,
            'degree': lot % 30,
        }

    def _calculate_zr_timeline(
        self,
        start_sign: str,
        birth_datetime: datetime,
        levels: int = 4
    ) -> List[Dict]:
        """
        Calculate ZR timeline with nested levels

        Args:
            start_sign: Starting sign for ZR
            birth_datetime: Birth datetime
            levels: Number of nested levels to calculate

        Returns:
            List of ZR periods with dates
        """
        timeline = []
        current_date = birth_datetime

        # Start with Level 1 periods
        start_index = self.signs.index(start_sign)

        # Calculate major periods (Level 1)
        for i in range(12):  # All 12 signs
            sign_index = (start_index + i) % 12
            sign = self.signs[sign_index]
            period_length_years = ZR_PERIODS[sign]
            ruler = ZR_RULERS[sign]

            period_start = current_date
            period_end = current_date + timedelta(days=period_length_years * 365.25)

            period_data = {
                'level': 1,
                'sign': sign,
                'ruler': ruler,
                'start_date': period_start,
                'end_date': period_end,
                'duration_years': period_length_years,
                'is_peak': self._is_peak_period(sign, start_sign),
            }

            # Calculate sub-periods (Level 2) if requested
            if levels >= 2:
                period_data['sub_periods'] = self._calculate_sub_periods(
                    sign,
                    period_start,
                    period_end,
                    level=2
                )

            timeline.append(period_data)
            current_date = period_end

        return timeline

    def _calculate_sub_periods(
        self,
        parent_sign: str,
        start_date: datetime,
        end_date: datetime,
        level: int
    ) -> List[Dict]:
        """
        Calculate nested sub-periods within a major period

        Args:
            parent_sign: Sign of parent period
            start_date: Start of parent period
            end_date: End of parent period
            level: Current level (2, 3, 4, etc.)

        Returns:
            List of sub-periods
        """
        sub_periods = []
        parent_index = self.signs.index(parent_sign)
        total_duration = (end_date - start_date).total_seconds()

        # Calculate proportional durations for sub-periods
        total_sub_years = sum(ZR_PERIODS.values())

        current_date = start_date

        for i in range(12):
            sign_index = (parent_index + i) % 12
            sign = self.signs[sign_index]
            period_years = ZR_PERIODS[sign]
            ruler = ZR_RULERS[sign]

            # Proportional duration
            proportion = period_years / total_sub_years
            duration_seconds = total_duration * proportion
            sub_end = current_date + timedelta(seconds=duration_seconds)

            sub_period = {
                'level': level,
                'sign': sign,
                'ruler': ruler,
                'start_date': current_date,
                'end_date': sub_end,
                'duration_years': period_years * proportion,
            }

            sub_periods.append(sub_period)
            current_date = sub_end

        return sub_periods

    def _is_peak_period(self, current_sign: str, start_sign: str) -> bool:
        """
        Determine if period is a peak period
        Peak periods occur when returning to the starting sign or angular relationship
        """
        start_index = self.signs.index(start_sign)
        current_index = self.signs.index(current_sign)

        # Distance from start
        distance = (current_index - start_index) % 12

        # Peak at: same sign (0), square (3, 9), opposition (6), trine (4, 8)
        peak_distances = [0, 3, 4, 6, 8, 9]

        return distance in peak_distances

    def find_current_period(self, timeline: List[Dict], target_date: datetime) -> Optional[Dict]:
        """
        Find which ZR period is active at a given date

        Args:
            timeline: ZR timeline
            target_date: Date to check

        Returns:
            Active period or None
        """
        for period in timeline:
            if period['start_date'] <= target_date <= period['end_date']:
                # Also check sub-periods
                current_sub = None
                if 'sub_periods' in period:
                    for sub in period['sub_periods']:
                        if sub['start_date'] <= target_date <= sub['end_date']:
                            current_sub = sub
                            break

                return {
                    'major_period': period,
                    'sub_period': current_sub,
                }

        return None

    def predict_zr_events(
        self,
        timeline: List[Dict],
        event_category_mapping: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Predict events based on ZR periods

        Args:
            timeline: ZR timeline
            event_category_mapping: Mapping of signs/rulers to event categories

        Returns:
            List of predicted events
        """
        predictions = []

        # Default event mapping (simplified)
        default_mapping = {
            'Sun': 'career_promotion',
            'Moon': 'family_birth',
            'Mercury': 'personal_education',
            'Venus': 'relationship_start',
            'Mars': 'personal_health',
            'Jupiter': 'financial_windfall',
            'Saturn': 'career_termination',
        }

        mapping = event_category_mapping or default_mapping

        for period in timeline:
            # Peak periods are most significant
            if period.get('is_peak'):
                ruler = period['ruler']
                event_category = mapping.get(ruler)

                if event_category:
                    predictions.append({
                        'date_start': period['start_date'],
                        'date_end': period['end_date'],
                        'event_category': event_category,
                        'description': f'ZR peak period: {period["sign"]} ruled by {ruler}',
                        'confidence': 0.7,
                        'period_data': period,
                    })

        return predictions
