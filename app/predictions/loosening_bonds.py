"""
Loosening of Bonds (LB) - Hellenistic timing refinement technique
Refines Zodiacal Releasing timing by identifying critical transition points
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class LooseningBondsEngine:
    """
    Calculate Loosening of Bonds periods
    LB identifies when ZR periods transition and bonds "loosen"
    These are critical timing windows for predicted events
    """

    def __init__(self):
        self.signs = [
            'Aries', 'Taurus', 'Gemini', 'Cancer',
            'Leo', 'Virgo', 'Libra', 'Scorpio',
            'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
        ]

        # Planetary rulers
        self.rulers = {
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

    def identify_loosening_periods(
        self,
        zr_timeline: List[Dict],
        natal_chart: Dict
    ) -> List[Dict]:
        """
        Identify Loosening of Bonds periods within ZR timeline

        Args:
            zr_timeline: Zodiacal Releasing timeline
            natal_chart: Natal chart data

        Returns:
            List of loosening periods with dates and intensity
        """
        loosening_periods = []

        for i, period in enumerate(zr_timeline):
            # Check for various loosening conditions

            # 1. Transition between major periods
            if i < len(zr_timeline) - 1:
                next_period = zr_timeline[i + 1]

                # Loosening occurs at transition point
                transition_loosening = {
                    'type': 'major_transition',
                    'date': period['end_date'],
                    'from_sign': period['sign'],
                    'to_sign': next_period['sign'],
                    'from_ruler': period['ruler'],
                    'to_ruler': next_period['ruler'],
                    'intensity': self._calculate_transition_intensity(period, next_period),
                }

                loosening_periods.append(transition_loosening)

            # 2. Check for sub-period transitions
            if 'sub_periods' in period:
                sub_loosening = self._identify_sub_period_loosening(period['sub_periods'])
                loosening_periods.extend(sub_loosening)

            # 3. Check for activation by transits
            transit_activations = self._identify_transit_activations(
                period,
                natal_chart
            )
            loosening_periods.extend(transit_activations)

        return loosening_periods

    def _calculate_transition_intensity(
        self,
        current_period: Dict,
        next_period: Dict
    ) -> float:
        """
        Calculate intensity of loosening at period transition
        Higher intensity = more significant events likely

        Args:
            current_period: Current ZR period
            next_period: Next ZR period

        Returns:
            Intensity score (0.0 to 1.0)
        """
        intensity = 0.5  # Base intensity

        # Increase if moving to/from peak period
        if current_period.get('is_peak') or next_period.get('is_peak'):
            intensity += 0.2

        # Increase if rulers are in hard aspect relationship
        current_ruler = current_period['ruler']
        next_ruler = next_period['ruler']

        if self._rulers_in_hard_aspect(current_ruler, next_ruler):
            intensity += 0.15

        # Increase for major level transitions (vs sub-period transitions)
        if current_period.get('level', 1) == 1:
            intensity += 0.15

        return min(intensity, 1.0)

    def _rulers_in_hard_aspect(self, ruler1: str, ruler2: str) -> bool:
        """
        Check if two rulers are traditionally in hard aspect
        Simplified version - real version would check natal positions
        """
        # Traditional enmities
        hard_pairs = [
            ('Sun', 'Saturn'),
            ('Moon', 'Mars'),
            ('Mars', 'Venus'),
            ('Mercury', 'Jupiter'),
        ]

        return (ruler1, ruler2) in hard_pairs or (ruler2, ruler1) in hard_pairs

    def _identify_sub_period_loosening(self, sub_periods: List[Dict]) -> List[Dict]:
        """
        Identify loosening periods within sub-periods

        Args:
            sub_periods: List of sub-periods

        Returns:
            List of sub-period loosening points
        """
        loosening = []

        for i in range(len(sub_periods) - 1):
            current = sub_periods[i]
            next_sub = sub_periods[i + 1]

            loosening.append({
                'type': 'sub_transition',
                'date': current['end_date'],
                'from_sign': current['sign'],
                'to_sign': next_sub['sign'],
                'from_ruler': current['ruler'],
                'to_ruler': next_sub['ruler'],
                'level': current.get('level', 2),
                'intensity': 0.3,  # Lower intensity than major transitions
            })

        return loosening

    def _identify_transit_activations(
        self,
        period: Dict,
        natal_chart: Dict
    ) -> List[Dict]:
        """
        Identify when transits activate the period's ruler
        These are loosening moments within the period

        Args:
            period: ZR period
            natal_chart: Natal chart

        Returns:
            List of transit activation points
        """
        activations = []

        # This is simplified - real version would calculate actual transits
        # For now, just mark key points within the period

        ruler = period['ruler']
        start_date = period['start_date']
        end_date = period['end_date']
        duration = end_date - start_date

        # Mark quarter points within period (simplified)
        quarter_points = [0.25, 0.5, 0.75]

        for fraction in quarter_points:
            activation_date = start_date + (duration * fraction)

            activations.append({
                'type': 'transit_activation',
                'date': activation_date,
                'ruler': ruler,
                'sign': period['sign'],
                'intensity': 0.4,
                'description': f'Transit activation of {ruler} period',
            })

        return activations

    def refine_prediction_timing(
        self,
        zr_prediction: Dict,
        loosening_periods: List[Dict]
    ) -> Dict:
        """
        Refine a ZR prediction using LB timing

        Args:
            zr_prediction: A ZR-based prediction with date range
            loosening_periods: List of loosening periods

        Returns:
            Refined prediction with more specific timing
        """
        pred_start = zr_prediction['date_start']
        pred_end = zr_prediction['date_end']

        # Find loosening periods within prediction window
        relevant_loosening = [
            lb for lb in loosening_periods
            if pred_start <= lb['date'] <= pred_end
        ]

        if not relevant_loosening:
            # No refinement possible
            return zr_prediction

        # Sort by intensity
        relevant_loosening.sort(key=lambda x: x.get('intensity', 0.5), reverse=True)

        # Use highest intensity loosening as refined date
        best_loosening = relevant_loosening[0]

        refined = zr_prediction.copy()
        refined['refined_date'] = best_loosening['date']
        refined['loosening_data'] = best_loosening
        refined['confidence'] = min(zr_prediction.get('confidence', 0.5) + 0.2, 1.0)

        return refined

    def predict_lb_events(
        self,
        loosening_periods: List[Dict],
        min_intensity: float = 0.6
    ) -> List[Dict]:
        """
        Generate event predictions based on loosening periods

        Args:
            loosening_periods: List of loosening periods
            min_intensity: Minimum intensity threshold for predictions

        Returns:
            List of predictions
        """
        predictions = []

        # Event type mapping based on ruler
        ruler_events = {
            'Sun': 'career_promotion',
            'Moon': 'family_birth',
            'Mercury': 'personal_education',
            'Venus': 'relationship_start',
            'Mars': 'personal_health',
            'Jupiter': 'financial_windfall',
            'Saturn': 'career_termination',
        }

        for lb in loosening_periods:
            intensity = lb.get('intensity', 0.5)

            if intensity >= min_intensity:
                # Major transition - make prediction
                ruler = lb.get('to_ruler') or lb.get('ruler')
                event_category = ruler_events.get(ruler)

                if event_category:
                    predictions.append({
                        'date': lb['date'],
                        'event_category': event_category,
                        'description': f'Loosening of Bonds: {lb["type"]}',
                        'confidence': intensity,
                        'lb_data': lb,
                    })

        return predictions

    def calculate_critical_dates(
        self,
        zr_timeline: List[Dict],
        natal_chart: Dict,
        target_year: int
    ) -> List[Dict]:
        """
        Calculate all critical dates (LB points) for a given year

        Args:
            zr_timeline: ZR timeline
            natal_chart: Natal chart
            target_year: Year to find critical dates for

        Returns:
            List of critical dates with details
        """
        # Get all loosening periods
        all_loosening = self.identify_loosening_periods(zr_timeline, natal_chart)

        # Filter for target year
        critical_dates = []

        for lb in all_loosening:
            if lb['date'].year == target_year:
                critical_dates.append({
                    'date': lb['date'],
                    'type': lb['type'],
                    'intensity': lb.get('intensity', 0.5),
                    'description': self._describe_critical_date(lb),
                    'data': lb,
                })

        # Sort by date
        critical_dates.sort(key=lambda x: x['date'])

        return critical_dates

    def _describe_critical_date(self, loosening: Dict) -> str:
        """Generate human-readable description of loosening period"""
        lb_type = loosening['type']

        if lb_type == 'major_transition':
            return (f"Major period transition from {loosening['from_sign']} "
                   f"to {loosening['to_sign']} "
                   f"(intensity: {loosening.get('intensity', 0.5):.2f})")

        elif lb_type == 'sub_transition':
            return (f"Sub-period transition from {loosening['from_sign']} "
                   f"to {loosening['to_sign']}")

        elif lb_type == 'transit_activation':
            return f"Transit activation of {loosening['ruler']} period"

        return "Loosening period"
