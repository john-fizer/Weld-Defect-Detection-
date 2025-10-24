"""
LLM integration for holographic chart interpretation
Synthesizes insights across multiple house systems and techniques
"""
from typing import Dict, List, Optional
import json
from anthropic import Anthropic
from openai import OpenAI
from app.config import settings


class ChartNarrator:
    """
    LLM-powered chart interpretation engine
    Generates unified holographic narratives from multi-system data
    """

    def __init__(self, provider: str = "anthropic"):
        """
        Initialize narrator with LLM provider

        Args:
            provider: "anthropic" or "openai"
        """
        self.provider = provider

        if provider == "anthropic":
            self.client = Anthropic(api_key=settings.anthropic_api_key)
            self.model = "claude-3-5-sonnet-20241022"
        elif provider == "openai":
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = "gpt-4"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def synthesize_multi_system_interpretation(
        self,
        chart_data: Dict,
        prompt_version: str = "v1"
    ) -> Dict:
        """
        Generate holographic interpretation across all house systems

        Args:
            chart_data: Dictionary containing charts from multiple systems
            prompt_version: Version of prompt template to use

        Returns:
            Interpretation with text, themes, and confidence
        """
        # Prepare chart data for LLM
        chart_summary = self._prepare_chart_summary(chart_data)

        # Get prompt
        prompt = self._build_synthesis_prompt(chart_summary, prompt_version)

        # Generate interpretation
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            interpretation_text = response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert astrologer synthesizing insights from multiple chart systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
            interpretation_text = response.choices[0].message.content

        # Extract themes and confidence
        themes = self._extract_themes(interpretation_text)
        confidence = self._estimate_confidence(interpretation_text)

        return {
            'interpretation_text': interpretation_text,
            'key_themes': themes,
            'confidence_score': confidence,
            'llm_model': self.model,
            'prompt_version': prompt_version,
        }

    def interpret_prediction_confluence(
        self,
        predictions: Dict[str, List[Dict]],
        time_window_days: int = 30
    ) -> Dict:
        """
        Interpret when multiple prediction techniques agree (holographic confirmation)

        Args:
            predictions: Dict of predictions by technique
                         {'progressions': [...], 'transits': [...], 'zr': [...], 'lb': [...]}
            time_window_days: Window for considering predictions as confluent

        Returns:
            Interpretation of confluent predictions
        """
        # Find temporal confluence
        confluent_events = self._find_confluent_predictions(predictions, time_window_days)

        if not confluent_events:
            return {
                'interpretation_text': 'No significant confluence detected between techniques.',
                'confidence_score': 0.0,
            }

        # Build prompt for confluence interpretation
        prompt = self._build_confluence_prompt(confluent_events)

        # Generate interpretation
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            interpretation_text = response.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing astrological timing techniques."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )
            interpretation_text = response.choices[0].message.content

        # High confidence when multiple techniques agree
        confluence_count = sum(len(events) for events in confluent_events.values())
        confidence = min(0.5 + (confluence_count * 0.1), 0.95)

        return {
            'interpretation_text': interpretation_text,
            'confluent_events': confluent_events,
            'confidence_score': confidence,
            'llm_model': self.model,
        }

    def _prepare_chart_summary(self, chart_data: Dict) -> str:
        """Prepare concise chart summary for LLM"""
        summary_parts = []

        # Include data from each system
        for system_name, chart in chart_data.items():
            planets_summary = []
            for planet, data in chart.get('planets', {}).items():
                planets_summary.append(
                    f"{planet} in {data['sign']} {data['degree']:.1f}° (House {data.get('house', '?')})"
                )

            summary_parts.append(f"=== {system_name.upper()} ===")
            summary_parts.append("\n".join(planets_summary))

            # Include nakshatra data if available
            if 'nakshatras' in chart:
                nak_summary = []
                for planet, nak_data in chart['nakshatras'].items():
                    nak_summary.append(
                        f"{planet}: {nak_data['nakshatra']} (pada {nak_data['pada']}) ruled by {nak_data['ruler']}"
                    )
                summary_parts.append("Nakshatras:")
                summary_parts.append("\n".join(nak_summary))

            summary_parts.append("")  # Blank line

        return "\n".join(summary_parts)

    def _build_synthesis_prompt(self, chart_summary: str, version: str) -> str:
        """Build prompt for holographic synthesis"""
        if version == "v1":
            return f"""You are analyzing a natal chart calculated using multiple house systems and astrological traditions. Your task is to provide a HOLOGRAPHIC interpretation where different systems COMPLEMENT and CLARIFY each other, rather than contradict.

{chart_summary}

Please provide a synthesized interpretation that:

1. Identifies where different systems REINFORCE the same themes (holographic confirmation)
2. Explains how apparent differences between systems actually provide COMPLEMENTARY perspectives
3. Uses Vedic nakshatras to add depth and nuance to Western placements
4. Integrates decan and degree theory to refine interpretations
5. Highlights the UNIFIED core truth emerging from all systems

Focus on:
- Major life themes and purpose
- Personality traits and strengths
- Challenges and growth areas
- Relationship patterns
- Career inclinations

Structure your response as a flowing narrative, not bullet points. Emphasize synthesis over analysis.
"""
        else:
            # Future prompt versions for A/B testing
            return self._build_synthesis_prompt(chart_summary, "v1")

    def _build_confluence_prompt(self, confluent_events: Dict) -> str:
        """Build prompt for interpreting confluent predictions"""
        events_summary = json.dumps(confluent_events, indent=2, default=str)

        return f"""You are analyzing astrological timing predictions from multiple techniques. The following predictions show TEMPORAL CONFLUENCE - multiple techniques agreeing on similar timing:

{events_summary}

When multiple independent techniques agree, this represents HOLOGRAPHIC CONFIRMATION of likely events.

Please provide an interpretation that:

1. Explains the significance of this confluence
2. Identifies the most likely event types and timing
3. Describes what themes/areas of life are activated
4. Rates the strength of the indication (weak/moderate/strong)
5. Provides actionable guidance

Be direct and specific. Focus on the unified message across techniques.
"""

    def _extract_themes(self, interpretation_text: str) -> List[str]:
        """Extract key themes from interpretation text"""
        # Simplified - could use NLP in production
        themes = []

        theme_keywords = {
            'career': ['career', 'profession', 'vocation', 'work'],
            'relationships': ['relationship', 'love', 'partnership', 'marriage'],
            'family': ['family', 'children', 'parent', 'home'],
            'spirituality': ['spiritual', 'philosophy', 'meaning', 'purpose'],
            'health': ['health', 'body', 'vitality', 'wellness'],
            'finances': ['money', 'wealth', 'financial', 'resources'],
            'creativity': ['creative', 'art', 'expression', 'innovation'],
        }

        text_lower = interpretation_text.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def _estimate_confidence(self, interpretation_text: str) -> float:
        """Estimate confidence score from interpretation text"""
        # Simplified heuristic
        confidence = 0.6  # Base confidence

        # Increase for strong language
        strong_words = ['clearly', 'definitely', 'strongly', 'significant', 'prominent']
        weak_words = ['possibly', 'perhaps', 'might', 'could', 'maybe']

        text_lower = interpretation_text.lower()

        strong_count = sum(1 for word in strong_words if word in text_lower)
        weak_count = sum(1 for word in weak_words if word in text_lower)

        confidence += strong_count * 0.05
        confidence -= weak_count * 0.05

        return max(0.0, min(1.0, confidence))

    def _find_confluent_predictions(
        self,
        predictions: Dict[str, List[Dict]],
        time_window_days: int
    ) -> Dict:
        """
        Find predictions that cluster together in time

        Args:
            predictions: Predictions by technique
            time_window_days: Window for confluence

        Returns:
            Clustered confluent predictions
        """
        from datetime import timedelta

        confluent = {}
        all_predictions = []

        # Flatten all predictions with source
        for technique, preds in predictions.items():
            for pred in preds:
                all_predictions.append({
                    'technique': technique,
                    'data': pred,
                    'date': pred.get('date') or pred.get('date_start'),
                })

        # Sort by date
        all_predictions.sort(key=lambda x: x['date'])

        # Find clusters
        i = 0
        while i < len(all_predictions):
            cluster_start = all_predictions[i]['date']
            cluster_end = cluster_start + timedelta(days=time_window_days)
            cluster = [all_predictions[i]]

            # Find all predictions within window
            j = i + 1
            while j < len(all_predictions) and all_predictions[j]['date'] <= cluster_end:
                cluster.append(all_predictions[j])
                j += 1

            # If cluster has multiple techniques, it's confluent
            techniques_in_cluster = set(p['technique'] for p in cluster)
            if len(techniques_in_cluster) >= 2:
                cluster_key = cluster_start.strftime("%Y-%m-%d")
                confluent[cluster_key] = cluster

            i = j if j > i + 1 else i + 1

        return confluent
