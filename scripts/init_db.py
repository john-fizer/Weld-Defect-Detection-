"""
Initialize database with all tables
"""
from app.models.database import init_db, engine
from app.models.chart import NatalChart, ChartCalculation, ChartInterpretation, InterpretationFeedback
from app.models.predictions import (
    Prediction, ProgressionPrediction, TransitPrediction,
    ZodiacalReleasingPrediction, LooseningBondsPrediction,
    ProgressionFeedback, TransitFeedback, ZRFeedback, LBFeedback
)
from app.models.events import UserEvent, EventPredictionCorrelation


def main():
    """Initialize all database tables"""
    print("Initializing database...")

    # Create all tables
    init_db()

    print("Database initialized successfully!")
    print("\nCreated tables:")
    print("  - natal_charts")
    print("  - chart_calculations")
    print("  - chart_interpretations")
    print("  - interpretation_feedback")
    print("  - predictions")
    print("  - progression_predictions")
    print("  - transit_predictions")
    print("  - zodiacal_releasing_predictions")
    print("  - loosening_bonds_predictions")
    print("  - progression_feedback")
    print("  - transit_feedback")
    print("  - zr_feedback")
    print("  - lb_feedback")
    print("  - user_events")
    print("  - event_prediction_correlations")

    print("\nModular feedback architecture ready!")
    print("Each technique has isolated feedback collection for independent refinement.")


if __name__ == "__main__":
    main()
