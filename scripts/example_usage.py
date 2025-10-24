"""
Example usage of the Holographic Astrology Platform
Demonstrates multi-system synthesis and prediction confluence
"""
from datetime import datetime
import pytz
from app.core.chart_calculator import ChartCalculator
from app.core.vedic import VedicCalculator
from app.predictions.progressions import ProgressionsEngine
from app.predictions.transits import TransitsEngine
from app.predictions.zodiacal_releasing import ZodiacalReleasingEngine
from app.predictions.loosening_bonds import LooseningBondsEngine
from app.llm.narrator import ChartNarrator


def example_holographic_chart_calculation():
    """
    Example: Calculate chart using multiple house systems
    Shows how different systems provide complementary perspectives
    """
    print("=" * 80)
    print("HOLOGRAPHIC CHART CALCULATION EXAMPLE")
    print("=" * 80)

    # Birth data
    birth_date = "1990-01-15"
    birth_time = "14:30"
    latitude = 40.7128
    longitude = -74.0060
    timezone_str = "America/New_York"

    # Create timezone-aware datetime
    tz = pytz.timezone(timezone_str)
    dt_naive = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    birth_datetime = tz.localize(dt_naive)

    # Initialize calculators
    chart_calc = ChartCalculator()
    vedic_calc = VedicCalculator()

    # Calculate charts in different systems
    house_systems = ['placidus', 'whole_sign', 'vedic_whole']

    all_charts = {}

    for house_system in house_systems:
        zodiac = 'sidereal' if 'vedic' in house_system else 'tropical'

        print(f"\n--- Calculating {house_system.upper()} ({zodiac}) ---")

        chart = chart_calc.calculate_natal_chart(
            birth_datetime,
            latitude,
            longitude,
            house_system=house_system,
            zodiac_system=zodiac
        )

        # Add decans
        chart['decans'] = chart_calc.calculate_decans(chart['planets'])

        # Add Vedic data for sidereal charts
        if zodiac == 'sidereal':
            chart['nakshatras'] = vedic_calc.calculate_nakshatras(chart['planets'])

        all_charts[house_system] = chart

        # Print sample planetary positions
        print(f"\nSun position:")
        sun = chart['planets']['Sun']
        print(f"  {sun['sign']} {sun['degree']:.2f}° (House {sun.get('house', '?')})")

        if zodiac == 'sidereal' and 'nakshatras' in chart:
            sun_nak = chart['nakshatras']['Sun']
            print(f"  Nakshatra: {sun_nak['nakshatra']} (pada {sun_nak['pada']}) ruled by {sun_nak['ruler']}")

    print("\n" + "=" * 80)
    print("HOLOGRAPHIC SYNTHESIS")
    print("=" * 80)
    print("\nNotice how different house systems place planets in different houses,")
    print("but these perspectives COMPLEMENT each other to reveal the full picture.")
    print("For example:")
    print("- Placidus shows temporal emphasis (time-based houses)")
    print("- Whole Sign shows sign-based themes (equal dignity)")
    print("- Vedic adds nakshatra rulers for deeper insight")
    print("\nThis is the 'holographic' approach - multiple views of the same truth.")

    return all_charts


def example_prediction_confluence():
    """
    Example: Generate predictions from multiple techniques
    Shows how confluence increases confidence
    """
    print("\n\n" + "=" * 80)
    print("PREDICTION CONFLUENCE EXAMPLE")
    print("=" * 80)

    # For this example, we'll use simplified mock data
    # In production, you'd use actual calculated natal chart

    natal_chart = {
        'birth_datetime': datetime(1990, 1, 15, 14, 30, tzinfo=pytz.UTC),
        'planets': {
            'Sun': {'longitude': 295.5, 'sign': 'Capricorn', 'degree': 25.5, 'house': 10},
            'Moon': {'longitude': 145.2, 'sign': 'Leo', 'degree': 25.2, 'house': 5},
            'Venus': {'longitude': 315.8, 'sign': 'Aquarius', 'degree': 15.8, 'house': 11},
        },
        'houses': {
            1: {'cusp': 75.0, 'sign': 'Gemini'},
            # ... other houses
            'angles': {'ASC': 75.0, 'MC': 345.0}
        }
    }

    # Prediction time range
    start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 12, 31, tzinfo=pytz.UTC)

    print("\nCalculating predictions from multiple techniques...")

    # Initialize engines
    prog_engine = ProgressionsEngine()
    transit_engine = TransitsEngine()
    zr_engine = ZodiacalReleasingEngine()
    lb_engine = LooseningBondsEngine()

    # Get predictions
    predictions = {}

    print("\n1. Secondary Progressions...")
    predictions['progressions'] = prog_engine.predict_progression_events(
        natal_chart,
        start_date,
        end_date,
        40.7128,
        -74.0060
    )
    print(f"   Found {len(predictions['progressions'])} progression indicators")

    print("\n2. Transits...")
    predictions['transits'] = transit_engine.predict_transit_events(
        natal_chart,
        start_date,
        end_date
    )
    print(f"   Found {len(predictions['transits'])} transit indicators")

    print("\n3. Zodiacal Releasing...")
    zr_result = zr_engine.calculate_zr_from_lot_of_fortune(
        natal_chart,
        natal_chart['birth_datetime']
    )
    predictions['zr'] = zr_engine.predict_zr_events(zr_result['timeline'])
    print(f"   Found {len(predictions['zr'])} ZR period indicators")

    print("\n4. Loosening of Bonds...")
    lb_periods = lb_engine.identify_loosening_periods(
        zr_result['timeline'],
        natal_chart
    )
    predictions['lb'] = lb_engine.predict_lb_events(lb_periods)
    print(f"   Found {len(predictions['lb'])} LB timing indicators")

    print("\n" + "-" * 80)
    print("ANALYZING CONFLUENCE...")
    print("-" * 80)

    narrator = ChartNarrator(provider="anthropic")  # Would need API key

    # In production, this would call the LLM
    print("\nWhen multiple techniques agree on timing, confidence increases dramatically.")
    print("For example:")
    print("  - If progressions, transits, AND ZR all indicate career change in March 2024")
    print("  - This represents 'holographic confirmation' across independent methods")
    print("  - Confidence level: HIGH")
    print("\nThis multi-technique approach is what makes the system powerful.")

    return predictions


def example_feedback_loop():
    """
    Example: How feedback improves the system
    Shows the closed-loop ML training process
    """
    print("\n\n" + "=" * 80)
    print("FEEDBACK LOOP EXAMPLE")
    print("=" * 80)

    print("\nThe system continuously improves through user feedback:")
    print("\n1. User receives prediction (e.g., 'career change in March 2024')")
    print("   - Technique: Secondary Progressions")
    print("   - Confidence: 65%")

    print("\n2. Event occurs (or doesn't)")
    print("   - User reports: Event occurred on March 15, 2024")
    print("   - Accuracy: Prediction was 15 days early")

    print("\n3. Feedback is stored in modular database")
    print("   - Stored in: progression_feedback table")
    print("   - Isolated from: transit_feedback, zr_feedback, lb_feedback")

    print("\n4. ML model retrains on new data")
    print("   - Progression model learns this pattern")
    print("   - Other techniques unaffected (modular)")

    print("\n5. Next prediction is more accurate")
    print("   - New confidence: 72%")
    print("   - Refined timing based on learned patterns")

    print("\n6. Performance monitoring")
    print("   - If a technique performs poorly (< 30% accuracy)")
    print("   - System recommends: 'REMOVE_OR_REDESIGN'")
    print("   - Keeps only what works!")

    print("\n" + "=" * 80)


def example_cnn_alternative():
    """
    Example: Why direct data approach is better than CNN for chart reading
    """
    print("\n\n" + "=" * 80)
    print("ML ARCHITECTURE: Why NOT use CNN for chart reading")
    print("=" * 80)

    print("\nYou asked about training a CNN to read chart wheels.")
    print("Here's why the direct data approach is superior:")

    print("\n❌ CNN Approach (Visual):")
    print("  1. Generate chart wheel image")
    print("  2. Train CNN to recognize visual patterns")
    print("  3. Extract approximate positions from image")
    print("  4. Interpret based on visual recognition")
    print("  Downsides: Computationally expensive, less accurate, harder to debug")

    print("\n✅ Direct Data Approach (Mathematical):")
    print("  1. Calculate exact planetary positions (Swiss Ephemeris)")
    print("  2. Extract precise features (degrees, aspects, houses)")
    print("  3. Train models on EXACT data + outcomes")
    print("  4. Interpret based on learned correlations")
    print("  Benefits: Precise, fast, interpretable, modular")

    print("\n💡 Where CNNs COULD be useful:")
    print("  - Pattern recognition in multi-chart overlays")
    print("  - Visual aspect configuration detection (Grand Trines, etc.)")
    print("  - Anomaly detection in complex chart comparisons")
    print("  - As a SECONDARY validation layer, not primary")

    print("\nCurrent architecture uses:")
    print("  - Structured ML (Random Forest, Gradient Boosting)")
    print("  - LLMs for holistic interpretation (Claude, GPT-4)")
    print("  - Modular feedback loops per technique")
    print("  - Mathematical precision throughout")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run examples
    example_holographic_chart_calculation()
    example_prediction_confluence()
    example_feedback_loop()
    example_cnn_alternative()

    print("\n\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Multiple house systems = Holographic perspective (complementary, not contradictory)")
    print("2. Multiple techniques = Confluence analysis (agreement = higher confidence)")
    print("3. Modular feedback = Independent refinement (keep what works, remove what doesn't)")
    print("4. Direct data > Visual parsing (mathematical precision beats image recognition)")
    print("\nThis architecture enables continuous improvement while maintaining precision.")
