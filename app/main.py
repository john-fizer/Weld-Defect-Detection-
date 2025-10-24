"""
Main FastAPI application
Holographic Astrology Platform
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import pytz

from app.models.database import get_db, init_db
from app.core.chart_calculator import ChartCalculator
from app.core.vedic import VedicCalculator
from app.predictions.progressions import ProgressionsEngine
from app.predictions.transits import TransitsEngine
from app.predictions.zodiacal_releasing import ZodiacalReleasingEngine
from app.predictions.loosening_bonds import LooseningBondsEngine
from app.llm.narrator import ChartNarrator
from app.ml.feedback_processor import FeedbackProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Holographic Astrology Platform",
    description="Next-gen astrology with multi-system synthesis and ML feedback loops",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("Database initialized")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Holographic Astrology Platform API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Example endpoints - in production these would be more extensive

@app.post("/chart/calculate")
async def calculate_chart(
    birth_date: str,  # ISO format
    birth_time: str,  # HH:MM format
    latitude: float,
    longitude: float,
    timezone: str,
    house_systems: Optional[list] = None,
    include_vedic: bool = True,
):
    """
    Calculate natal chart with multiple house systems

    Example:
        POST /chart/calculate
        {
            "birth_date": "1990-01-15",
            "birth_time": "14:30",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone": "America/New_York",
            "house_systems": ["placidus", "whole_sign", "vedic_whole"],
            "include_vedic": true
        }
    """
    try:
        # Parse datetime
        tz = pytz.timezone(timezone)
        dt_naive = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
        birth_datetime = tz.localize(dt_naive)

        # Initialize calculators
        chart_calc = ChartCalculator()
        vedic_calc = VedicCalculator()

        # Default house systems
        if house_systems is None:
            house_systems = ['placidus', 'whole_sign']

        # Calculate charts for each system
        charts = {}

        for house_system in house_systems:
            # Determine zodiac system
            zodiac = 'sidereal' if 'vedic' in house_system else 'tropical'

            chart = chart_calc.calculate_natal_chart(
                birth_datetime,
                latitude,
                longitude,
                house_system=house_system,
                zodiac_system=zodiac
            )

            # Add decans
            chart['decans'] = chart_calc.calculate_decans(chart['planets'])

            # Add Vedic data if requested
            if include_vedic and zodiac == 'sidereal':
                chart['nakshatras'] = vedic_calc.calculate_nakshatras(chart['planets'])

            charts[house_system] = chart

        return {
            "birth_info": {
                "date": birth_date,
                "time": birth_time,
                "location": {"latitude": latitude, "longitude": longitude},
                "timezone": timezone,
            },
            "charts": charts,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/interpretation/synthesize")
async def synthesize_interpretation(
    charts: dict,
    llm_provider: str = "anthropic"
):
    """
    Generate holographic interpretation across multiple chart systems

    Example:
        POST /interpretation/synthesize
        {
            "charts": {...chart data from /chart/calculate...},
            "llm_provider": "anthropic"
        }
    """
    try:
        narrator = ChartNarrator(provider=llm_provider)
        interpretation = narrator.synthesize_multi_system_interpretation(charts)

        return interpretation

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predictions/progressions")
async def get_progression_predictions(
    natal_chart: dict,
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
):
    """
    Get progression-based predictions

    Example:
        POST /predictions/progressions
        {
            "natal_chart": {...},
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    """
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        prog_engine = ProgressionsEngine()
        predictions = prog_engine.predict_progression_events(
            natal_chart,
            start_dt,
            end_dt,
            latitude,
            longitude
        )

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predictions/all")
async def get_all_predictions(
    natal_chart: dict,
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
):
    """
    Get predictions from all techniques with confluence analysis

    Example:
        POST /predictions/all
        {
            "natal_chart": {...},
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    """
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Initialize all prediction engines
        prog_engine = ProgressionsEngine()
        transit_engine = TransitsEngine()
        zr_engine = ZodiacalReleasingEngine()
        lb_engine = LooseningBondsEngine()

        # Get predictions from each technique
        predictions = {}

        # Progressions
        predictions['progressions'] = prog_engine.predict_progression_events(
            natal_chart, start_dt, end_dt, latitude, longitude
        )

        # Transits
        predictions['transits'] = transit_engine.predict_transit_events(
            natal_chart, start_dt, end_dt
        )

        # Zodiacal Releasing
        zr_result = zr_engine.calculate_zr_from_lot_of_fortune(
            natal_chart,
            natal_chart.get('birth_datetime', start_dt)
        )
        predictions['zr'] = zr_engine.predict_zr_events(zr_result['timeline'])

        # Loosening of Bonds
        lb_periods = lb_engine.identify_loosening_periods(
            zr_result['timeline'],
            natal_chart
        )
        predictions['lb'] = lb_engine.predict_lb_events(lb_periods)

        # Analyze confluence
        narrator = ChartNarrator()
        confluence_analysis = narrator.interpret_prediction_confluence(predictions)

        return {
            "predictions_by_technique": predictions,
            "confluence_analysis": confluence_analysis,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ml/performance")
async def get_ml_performance(
    db: Session = Depends(get_db)
):
    """
    Get ML performance metrics for all techniques

    Example:
        GET /ml/performance
    """
    try:
        processor = FeedbackProcessor(db)
        performance = processor.compare_technique_performance()

        return {
            "performance_comparison": performance.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feedback/submit")
async def submit_feedback(
    prediction_id: int,
    event_occurred: bool,
    accuracy_rating: int,
    actual_event_date: Optional[str] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback on a prediction

    This feeds into the closed-loop ML training pipeline
    """
    # This would create appropriate feedback record based on prediction type
    # Implementation depends on database session and models

    return {
        "status": "feedback_received",
        "prediction_id": prediction_id,
        "message": "Feedback will be used to improve future predictions"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
