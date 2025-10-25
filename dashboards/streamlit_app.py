"""Streamlit dashboard for multi-agent trading system."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from storage.db import get_session, init_db
from storage.models import Trade, Position, PerformanceMetric
from storage.portfolio import Portfolio


# Page config
st.set_page_config(
    page_title="Multi-Agent Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

# Initialize DB
try:
    init_db()
except:
    pass


# Title
st.title("🤖 Multi-Agent Trading System Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Positions", "Planner", "Analytics", "Settings"])

# Initialize session
session = get_session()


def load_portfolio():
    """Load portfolio data."""
    portfolio = Portfolio()
    portfolio.update_from_database()
    return portfolio


def load_recent_trades(limit=20):
    """Load recent trades."""
    trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
    return trades


def load_open_positions():
    """Load open positions."""
    positions = session.query(Position).filter(Position.status == "open").all()
    return positions


def load_performance_metrics(days=30):
    """Load performance metrics."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    metrics = session.query(PerformanceMetric).filter(
        PerformanceMetric.timestamp >= cutoff
    ).order_by(PerformanceMetric.timestamp).all()
    return metrics


# === OVERVIEW PAGE ===
if page == "Overview":
    st.header("Portfolio Overview")

    # Load data
    portfolio = load_portfolio()
    summary = portfolio.get_summary()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Account Value",
            f"${summary['account_value']:,.2f}",
            delta=f"{summary['total_pnl_pct']:.2%}",
        )

    with col2:
        st.metric(
            "Daily P&L",
            f"${summary['daily_pnl']:,.2f}",
            delta=f"{summary['daily_pnl_pct']:.2%}",
        )

    with col3:
        st.metric(
            "Open Positions",
            summary['open_positions'],
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{summary['max_drawdown']:.2%}",
            delta=None,
            delta_color="inverse",
        )

    # Recent trades
    st.subheader("Recent Trades")
    trades = load_recent_trades(10)

    if trades:
        trades_data = []
        for t in trades:
            trades_data.append({
                "Time": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Ticker": t.ticker,
                "Strategy": t.strategy,
                "Action": t.action,
                "Qty": t.quantity,
                "Fill Price": f"${t.fill_price:.2f}",
                "Status": t.status,
            })

        st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
    else:
        st.info("No trades yet")


# === POSITIONS PAGE ===
elif page == "Positions":
    st.header("Open Positions")

    positions = load_open_positions()

    if positions:
        # Summary metrics
        total_delta = sum(p.delta * p.quantity for p in positions)
        total_theta = sum(p.theta * p.quantity for p in positions)
        total_vega = sum(p.vega * p.quantity for p in positions)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Delta", f"{total_delta:.2f}")
        with col2:
            st.metric("Portfolio Theta", f"{total_theta:.2f}")
        with col3:
            st.metric("Portfolio Vega", f"{total_vega:.2f}")

        # Positions table
        st.subheader("Position Details")
        pos_data = []
        for p in positions:
            pos_data.append({
                "Ticker": p.ticker,
                "Strategy": p.strategy,
                "Qty": p.quantity,
                "Entry": f"${p.entry_price:.2f}",
                "Days Open": (datetime.utcnow() - p.opened_at).days,
                "Delta": f"{p.delta:.3f}",
                "Theta": f"{p.theta:.2f}",
                "Vega": f"{p.vega:.2f}",
            })

        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

        # Strategy breakdown
        st.subheader("Strategy Breakdown")
        strategy_counts = {}
        for p in positions:
            strategy_counts[p.strategy] = strategy_counts.get(p.strategy, 0) + 1

        fig = px.pie(
            values=list(strategy_counts.values()),
            names=list(strategy_counts.keys()),
            title="Positions by Strategy"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions")


# === PLANNER PAGE ===
elif page == "Planner":
    st.header("Strategy Planner")

    st.info("This page would show proposed trades from the Strategy Planner agent")

    # Mock proposed trades
    st.subheader("Proposed Trades")

    mock_plans = [
        {
            "Ticker": "SPY",
            "Strategy": "Iron Condor",
            "Reasoning": "High IV rank (62), range-bound market",
            "Priority": 62,
            "Risk": "$500",
            "Status": "✅ Approved",
        },
        {
            "Ticker": "AAPL",
            "Strategy": "Wheel",
            "Reasoning": "Quality underlying, decent IV (45)",
            "Priority": 45,
            "Risk": "$750",
            "Status": "⚠️ Flagged - Human Review",
        },
    ]

    st.dataframe(pd.DataFrame(mock_plans), use_container_width=True)

    # Risk limits
    st.subheader("Risk Limits")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("VaR 95%", "2.1%", delta="0.9% below limit")
        st.metric("Margin Usage", "18%", delta="12% below limit")

    with col2:
        st.metric("Max Drawdown", "3.2%", delta="4.8% below alert")
        st.metric("Daily Loss", "-0.5%", delta="4.5% above limit")


# === ANALYTICS PAGE ===
elif page == "Analytics":
    st.header("Performance Analytics")

    # Load metrics
    metrics = load_performance_metrics(30)

    if metrics:
        # Create dataframe
        df = pd.DataFrame([{
            "Date": m.timestamp,
            "Total P&L": m.total_pnl,
            "Daily P&L": m.daily_pnl,
            "Open Positions": m.open_positions,
            "Win Rate": m.win_rate,
            "Sharpe": m.sharpe_ratio,
            "Max DD": m.max_drawdown,
        } for m in metrics])

        # P&L over time
        st.subheader("P&L Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Total P&L"],
            mode='lines+markers',
            name='Total P&L',
            line=dict(color='green', width=2),
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        st.subheader("Performance Metrics")
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No performance data yet")

    # Strategy comparison (mock)
    st.subheader("Strategy Attribution")

    strategy_pnl = {
        "Iron Condor": 2500,
        "Wheel": 1800,
        "Long Straddle": -500,
    }

    fig = px.bar(
        x=list(strategy_pnl.keys()),
        y=list(strategy_pnl.values()),
        title="P&L by Strategy",
        labels={"x": "Strategy", "y": "P&L ($)"},
        color=list(strategy_pnl.values()),
        color_continuous_scale=["red", "yellow", "green"],
    )
    st.plotly_chart(fig, use_container_width=True)


# === SETTINGS PAGE ===
elif page == "Settings":
    st.header("System Settings")

    st.subheader("Trading Configuration")

    environment = st.selectbox(
        "Environment",
        ["Paper Trading", "Live Trading (⚠️ DANGER)"],
        index=0,
    )

    if environment == "Live Trading (⚠️ DANGER)":
        st.error("⚠️ LIVE TRADING ENABLED - Real money at risk!")

    st.subheader("Risk Limits")

    col1, col2 = st.columns(2)

    with col1:
        max_var = st.slider("Max VaR (%)", 0.0, 10.0, 3.0, 0.1)
        max_dd = st.slider("Max Drawdown Alert (%)", 0.0, 20.0, 8.0, 1.0)

    with col2:
        max_margin = st.slider("Max Margin Usage (%)", 0.0, 100.0, 30.0, 5.0)
        max_loss = st.slider("Max Loss per Trade (%)", 0.0, 5.0, 1.0, 0.1)

    if st.button("Save Settings"):
        st.success("Settings saved (demo only - not implemented)")

    st.subheader("Database Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Initialize Database"):
            init_db()
            st.success("Database initialized")

    with col2:
        if st.button("Export Data"):
            st.info("Export feature not implemented")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Multi-Agent Trading System v0.1.0")
st.sidebar.caption("⚠️ Paper Trading Mode")

# Close session
session.close()
