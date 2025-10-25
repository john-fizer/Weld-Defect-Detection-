"""Database models for trading system."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Trade(Base):
    """Trade execution record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    action = Column(String(50), nullable=False)  # e.g., 'iron_condor', 'straddle'
    quantity = Column(Integer, default=1)
    fill_price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    legs = Column(Text)  # JSON string of legs
    status = Column(String(20), default="filled")  # filled, canceled, rejected

    def __repr__(self):
        return f"<Trade(id={self.id}, ticker={self.ticker}, strategy={self.strategy})>"


class Position(Base):
    """Open position record."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    ticker = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    quantity = Column(Integer, default=1)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    legs = Column(Text)  # JSON string
    status = Column(String(20), default="open")  # open, closed

    # Greeks
    delta = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)

    def __repr__(self):
        return f"<Position(id={self.id}, ticker={self.ticker}, status={self.status})>"


class PerformanceMetric(Base):
    """Performance metrics snapshot."""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    total_pnl = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    open_positions = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)

    def __repr__(self):
        return f"<PerformanceMetric(timestamp={self.timestamp}, pnl={self.total_pnl})>"


class AgentLog(Base):
    """Agent execution log."""

    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    details = Column(Text)  # JSON string
    success = Column(Boolean, default=True)

    def __repr__(self):
        return f"<AgentLog(agent={self.agent_name}, action={self.action})>"
