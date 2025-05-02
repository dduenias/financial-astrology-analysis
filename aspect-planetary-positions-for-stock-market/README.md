# Financial Astrology Analysis

This repository contains a collection of Python scripts for analyzing correlations between astronomical planetary cycles and financial market movements. The project explores the potential relationship between celestial mechanics and price action in various markets.

## Overview

Financial astrology examines how planetary positions and aspects may coincide with market turning points. This project implements a systematic approach to test these relationships using astronomical calculations, statistical analysis, and visualization techniques.

## Scripts

### Main Analysis Scripts

#### `lunar_planetary_confluence.py`
* Analyzes the confluence of lunar cycle events and planetary aspects
* Identifies dates when multiple astrological factors align
* Tests correlations with market turning points
* Generates statistical significance analysis and success rates
* Creates visualizations of results

#### `financial_astrology.py`
* Calculates precise planetary positions for any given date
* Identifies aspects between planets (conjunctions, squares, etc.)
* Maps positions to zodiac signs and Human Design gates
* Fetches corresponding market prices
* Saves results to CSV for further analysis

#### `jupiter_saturn_aspects.py`
* Scans date ranges for specific aspects between Jupiter and Saturn
* Supports both geocentric and heliocentric calculations
* Analyzes market patterns around aspect dates
* Creates visualizations showing price action relative to aspects
* Calculates statistical significance of observed patterns

#### `major_planetary_aspects_analysis.py`
* Analyzes correlations between major planetary aspects and market movements
* Tracks positions of outer planets (Venus through Pluto)
* Detects major aspects with support for retrograde motion
* Categorizes market patterns (highs, lows, trend changes)
* Generates comprehensive statistical summaries

#### `angular_increment_analysis.py`
* Projects forward from significant market dates
* Finds when planets reach specific angular increments
* Analyzes price behavior at these projected dates
* Tests the predictive value of angular relationships
* Visualizes results with highlighting of key dates

### Enhanced Analysis & Utilities

#### `enhanced_lunar_analysis.py`
* Advanced framework extending the lunar cycle analysis
* Adds market regime analysis (bull/bear/sideways)
* Analyzes relationship between lunar factors and volatility
* Creates composite indicators with multiple factors
* Implements time frame optimization

#### `lunar_cycle_analysis.py`
* Focuses specifically on lunar factors (phases, declination, etc.)
* Identifies dates when multiple lunar factors converge
* Analyzes market behavior around these "lunar clusters"
* Tests statistical significance of lunar cycle effects

#### `planetary_sign_ingress_analysis.py`
* Analyzes market behavior around planetary sign changes
* Tests the significance of planets entering new zodiac signs
* Identifies potential correlation with market turning points

## Key Features

* **Comprehensive Astronomical Calculations**: Uses Swiss Ephemeris for precise planetary positions
* **Multi-Market Analysis**: Tests correlations across various markets (SPX, Gold, Silver, Oil, etc.)
* **Statistical Validation**: Compares results against random date samples
* **Visualization Tools**: Creates charts of price action with astrological events
* **Combination Testing**: Identifies which specific combinations have the highest predictive value
* **Regime Analysis**: Tests effectiveness across different market conditions
* **Time Frame Optimization**: Analyzes which planetary cycles work best for different time frames

## Results

The analysis reveals several statistically significant patterns:

* Inner planets (Mercury, Venus, Mars) show stronger correlation with SPX and Oil
* Outer planets (Jupiter through Pluto) show inverse correlation with precious metals
* Moon Void of Course start times consistently act as effective triggers across markets
* Specific combinations achieve 80-100% success rates in identifying turning points
* Different markets respond to different planetary combinations

## Usage

Each script can be run independently to analyze specific astrological factors and markets. The typical workflow involves:

1. Running analysis for a specific market and date range
2. Reviewing statistical significance and top-performing combinations
3. Creating visual confirmation of patterns
4. Testing results through backtesting

## Requirements

* Python 3.7+
* Swiss Ephemeris (pyswisseph)
* pandas
* numpy
* matplotlib
* yfinance
* scipy

## Contributors

David Roy Duenias