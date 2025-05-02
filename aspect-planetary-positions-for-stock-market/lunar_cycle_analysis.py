"""
Lunar Cycle Analysis Tool
------------------------
Analyzes market behavior around significant lunar events and clusters.

This script:
1. Identifies five key lunar factors:
   - Moon declination (max/zero)
   - True Node direction changes
   - Moon phases (New, Full, Quarter)
   - Apogee/Perigee points
   - Moon void of course periods
2. Detects dates when multiple lunar factors occur simultaneously
3. Analyzes market behavior around these dates
4. Visualizes results and identifies patterns

Based on financial astrology principles.
"""

import swisseph as swe
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from math import fmod
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import time

def initialize_ephemeris():
    """Initialize the Swiss Ephemeris with proper path."""
    # Let Swiss Ephemeris download files automatically
    swe.set_ephe_path(None)

def get_julian_day(date, hour=12.0):
    """Convert date to Julian day."""
    return swe.julday(date.year, date.month, date.day, hour)

def get_date_from_julian(jd):
    """Convert Julian day to datetime date."""
    y, m, d, h = swe.revjul(jd, swe.GREG_CAL)
    return datetime.date(y, m, d)

def find_declination_events(start_date, end_date, precision_hours=6):
    """
    Find dates when Moon reaches maximum declination or crosses equator (0°).
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        precision_hours: Time precision for finding exact points
        
    Returns:
        List of dictionaries with declination event details
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = get_julian_day(start_date)
    jd_end = get_julian_day(end_date)
    
    # Step size for scanning (1 day initially)
    initial_step = 1.0
    precision_step = precision_hours / 24.0
    
    # Store found events
    declination_events = []
    
    # Set up flags
    flag = swe.FLG_SWIEPH
    
    # Track when declination changes direction
    prev_decl = None
    prev_jd = None
    direction = None  # 1 for increasing, -1 for decreasing
    
    # Scan through the date range with initial step
    current_jd = jd_start
    while current_jd <= jd_end:
        # Calculate Moon's position
        moon_pos, status = swe.calc_ut(current_jd, swe.MOON, flag)
        
        # Extract declination (pos[1])
        declination = moon_pos[1]
        
        # If we have previous declination, check for changes
        if prev_decl is not None:
            # Determine direction of movement
            current_direction = 1 if declination > prev_decl else -1
            
            # Check for direction change (maximum/minimum declination)
            if direction is not None and current_direction != direction:
                # A direction change happened between prev_jd and current_jd
                # Refine to find the exact JD of the max/min
                exact_jd = find_exact_extremum(prev_jd, current_jd, precision_step)
                
                # Get the refined declination
                exact_pos, _ = swe.calc_ut(exact_jd, swe.MOON, flag)
                exact_decl = exact_pos[1]
                
                # Determine if this is a max or min
                is_max = direction > 0  # If direction was positive and changed to negative, it's a maximum
                
                # Add to events
                declination_events.append({
                    'type': 'declination_max' if is_max else 'declination_min',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': exact_decl,
                    'description': f"Maximum declination ({exact_decl:.2f}°)" if is_max else f"Minimum declination ({exact_decl:.2f}°)"
                })
            
            # Check for equator crossing (sign change)
            if (prev_decl < 0 and declination > 0) or (prev_decl > 0 and declination < 0):
                # Crossing happened, refine to find exact crossing
                exact_jd = find_zero_crossing(prev_jd, current_jd, precision_step)
                
                # Add to events
                declination_events.append({
                    'type': 'declination_zero',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 0.0,
                    'description': "Moon crosses equator (0° declination)"
                })
            
            # Update direction
            direction = current_direction
        else:
            # First iteration, just set direction
            direction = 1 if declination > 0 else -1
        
        # Update previous values
        prev_decl = declination
        prev_jd = current_jd
        
        # Move to next day
        current_jd += initial_step
    
    # Sort events by date
    declination_events.sort(key=lambda x: x['julian_day'])
    
    return declination_events

def find_exact_extremum(jd1, jd2, step):
    """
    Find the Julian day where a value reaches its extremum (max or min).
    Uses a simple iterative search with decreasing steps.
    
    Args:
        jd1: Starting Julian day
        jd2: Ending Julian day
        step: Initial step size
        
    Returns:
        Julian day of the extremum
    """
    # Initialize ephemeris
    flag = swe.FLG_SWIEPH
    
    # Start with the midpoint
    mid_jd = (jd1 + jd2) / 2
    
    # Initialize left and right points
    left_jd = mid_jd - step
    right_jd = mid_jd + step
    
    # Calculate initial values
    left_pos, _ = swe.calc_ut(left_jd, swe.MOON, flag)
    mid_pos, _ = swe.calc_ut(mid_jd, swe.MOON, flag)
    right_pos, _ = swe.calc_ut(right_jd, swe.MOON, flag)
    
    left_decl = left_pos[1]
    mid_decl = mid_pos[1]
    right_decl = right_pos[1]
    
    # Refine until step is very small
    while step > 0.0001:  # Approximately 8.6 seconds
        # If middle value is higher than both neighbors, it's a maximum
        if mid_decl > left_decl and mid_decl > right_decl:
            # Found a maximum, further refine
            step /= 2
            
            # Calculate new left and right values closer to the middle
            left_jd = mid_jd - step
            right_jd = mid_jd + step
            
            left_pos, _ = swe.calc_ut(left_jd, swe.MOON, flag)
            right_pos, _ = swe.calc_ut(right_jd, swe.MOON, flag)
            
            left_decl = left_pos[1]
            right_decl = right_pos[1]
            
        # If middle value is lower than both neighbors, it's a minimum
        elif mid_decl < left_decl and mid_decl < right_decl:
            # Found a minimum, further refine
            step /= 2
            
            # Calculate new left and right values closer to the middle
            left_jd = mid_jd - step
            right_jd = mid_jd + step
            
            left_pos, _ = swe.calc_ut(left_jd, swe.MOON, flag)
            right_pos, _ = swe.calc_ut(right_jd, swe.MOON, flag)
            
            left_decl = left_pos[1]
            right_decl = right_pos[1]
            
        # Otherwise, move in the direction of increasing/decreasing values
        else:
            if left_decl > mid_decl:
                # Maximum is to the left, move left
                right_jd = mid_jd
                right_decl = mid_decl
                
                mid_jd = left_jd
                mid_decl = left_decl
                
                left_jd = mid_jd - step
                left_pos, _ = swe.calc_ut(left_jd, swe.MOON, flag)
                left_decl = left_pos[1]
                
            else:
                # Maximum is to the right, move right
                left_jd = mid_jd
                left_decl = mid_decl
                
                mid_jd = right_jd
                mid_decl = right_decl
                
                right_jd = mid_jd + step
                right_pos, _ = swe.calc_ut(right_jd, swe.MOON, flag)
                right_decl = right_pos[1]
    
    # Return the Julian day of the extremum
    return mid_jd

def find_zero_crossing(jd1, jd2, step):
    """
    Find the Julian day where a value crosses zero.
    Uses bisection method.
    
    Args:
        jd1: Starting Julian day
        jd2: Ending Julian day
        step: Minimum step size for precision
        
    Returns:
        Julian day of the zero crossing
    """
    # Initialize ephemeris
    flag = swe.FLG_SWIEPH
    
    # Get the values at the endpoints
    pos1, _ = swe.calc_ut(jd1, swe.MOON, flag)
    pos2, _ = swe.calc_ut(jd2, swe.MOON, flag)
    
    val1 = pos1[1]  # Declination
    val2 = pos2[1]  # Declination
    
    # Ensure there is a zero crossing
    if (val1 < 0 and val2 < 0) or (val1 > 0 and val2 > 0):
        # No zero crossing, return midpoint as fallback
        return (jd1 + jd2) / 2
    
    # Binary search until we reach desired precision
    while jd2 - jd1 > step:
        mid_jd = (jd1 + jd2) / 2
        mid_pos, _ = swe.calc_ut(mid_jd, swe.MOON, flag)
        mid_val = mid_pos[1]
        
        if mid_val == 0:
            # Exact zero found (very unlikely)
            return mid_jd
        
        if (val1 < 0 and mid_val < 0) or (val1 > 0 and mid_val > 0):
            # Same sign as val1, replace jd1
            jd1 = mid_jd
            val1 = mid_val
        else:
            # Same sign as val2, replace jd2
            jd2 = mid_jd
            val2 = mid_val
    
    # Return midpoint of final interval
    return (jd1 + jd2) / 2

def find_true_node_direction_changes(start_date, end_date, precision_hours=2):
    """
    Find dates when the True Node changes direction (retrograde/direct).
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        precision_hours: Time precision for finding exact points
        
    Returns:
        List of dictionaries with direction change events
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = get_julian_day(start_date)
    jd_end = get_julian_day(end_date)
    
    # Step size for scanning (1 day initially)
    initial_step = 1.0
    precision_step = precision_hours / 24.0
    
    # Store found events
    direction_changes = []
    
    # Set up flags for calculating speed
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    
    # Track node motion
    prev_speed = None
    prev_jd = None
    
    # Scan through the date range with initial step
    current_jd = jd_start
    while current_jd <= jd_end:
        # Calculate True Node position and speed
        node_pos, status = swe.calc_ut(current_jd, swe.TRUE_NODE, flag)
        
        # Extract speed (daily motion in longitude)
        speed = node_pos[3]
        
        # If we have previous speed, check for direction change
        if prev_speed is not None:
            # Check for direction change (crossing zero)
            if (prev_speed < 0 and speed > 0) or (prev_speed > 0 and speed < 0):
                # Direction change happened, refine to find exact moment
                exact_jd = find_speed_zero_crossing(prev_jd, current_jd, precision_step)
                
                # Determine type of change
                if prev_speed < 0 and speed > 0:
                    change_type = "retrograde_to_direct"
                    description = "True Node changes from retrograde to direct motion"
                else:
                    change_type = "direct_to_retrograde"
                    description = "True Node changes from direct to retrograde motion"
                
                # Add to events
                direction_changes.append({
                    'type': f'true_node_{change_type}',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 0.0,  # Speed at change is zero
                    'description': description
                })
        
        # Update previous values
        prev_speed = speed
        prev_jd = current_jd
        
        # Move to next day
        current_jd += initial_step
    
    # Sort events by date
    direction_changes.sort(key=lambda x: x['julian_day'])
    
    return direction_changes

def find_speed_zero_crossing(jd1, jd2, step):
    """
    Find the Julian day where the speed crosses zero.
    Uses bisection method.
    
    Args:
        jd1: Starting Julian day
        jd2: Ending Julian day
        step: Minimum step size for precision
        
    Returns:
        Julian day of the zero crossing
    """
    # Initialize ephemeris
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    
    # Get the speeds at the endpoints
    pos1, _ = swe.calc_ut(jd1, swe.TRUE_NODE, flag)
    pos2, _ = swe.calc_ut(jd2, swe.TRUE_NODE, flag)
    
    speed1 = pos1[3]  # Daily speed
    speed2 = pos2[3]  # Daily speed
    
    # Ensure there is a zero crossing
    if (speed1 < 0 and speed2 < 0) or (speed1 > 0 and speed2 > 0):
        # No zero crossing, return midpoint as fallback
        return (jd1 + jd2) / 2
    
    # Binary search until we reach desired precision
    while jd2 - jd1 > step:
        mid_jd = (jd1 + jd2) / 2
        mid_pos, _ = swe.calc_ut(mid_jd, swe.TRUE_NODE, flag)
        mid_speed = mid_pos[3]
        
        if mid_speed == 0:
            # Exact zero found (very unlikely)
            return mid_jd
        
        if (speed1 < 0 and mid_speed < 0) or (speed1 > 0 and mid_speed > 0):
            # Same sign as speed1, replace jd1
            jd1 = mid_jd
            speed1 = mid_speed
        else:
            # Same sign as speed2, replace jd2
            jd2 = mid_jd
            speed2 = mid_speed
    
    # Return midpoint of final interval
    return (jd1 + jd2) / 2

def find_moon_phases(start_date, end_date):
    """
    Find dates of New Moon, Full Moon, and Quarter Moons.
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        
    Returns:
        List of dictionaries with moon phase events
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = get_julian_day(start_date)
    jd_end = get_julian_day(end_date)
    
    # Step size for scanning (0.5 day)
    step = 0.5
    
    # Store found events
    phase_events = []
    
    # Previous sun-moon elongation
    prev_elongation = None
    prev_jd = None
    
    # Scan through the date range
    current_jd = jd_start
    while current_jd <= jd_end:
        # Calculate Sun and Moon positions
        sun_pos, _ = swe.calc_ut(current_jd, swe.SUN, swe.FLG_SWIEPH)
        moon_pos, _ = swe.calc_ut(current_jd, swe.MOON, swe.FLG_SWIEPH)
        
        # Calculate elongation (difference in longitude)
        sun_lon = sun_pos[0]
        moon_lon = moon_pos[0]
        
        # Calculate angular difference (0 to 360)
        elongation = (moon_lon - sun_lon) % 360
        
        # If we have a previous elongation, check for phase crossings
        if prev_elongation is not None:
            # Check for New Moon (0° elongation, crossing from 350+ to 10- degrees)
            if (prev_elongation > 350 and elongation < 10):
                # Find exact crossing
                exact_jd = find_phase_crossing(prev_jd, current_jd, 0)
                
                phase_events.append({
                    'type': 'new_moon',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 0,
                    'description': "New Moon"
                })
            
            # Check for First Quarter (90° elongation)
            elif (prev_elongation < 90 and elongation >= 90):
                # Find exact crossing
                exact_jd = find_phase_crossing(prev_jd, current_jd, 90)
                
                phase_events.append({
                    'type': 'first_quarter',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 90,
                    'description': "First Quarter Moon"
                })
            
            # Check for Full Moon (180° elongation)
            elif (prev_elongation < 180 and elongation >= 180):
                # Find exact crossing
                exact_jd = find_phase_crossing(prev_jd, current_jd, 180)
                
                phase_events.append({
                    'type': 'full_moon',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 180,
                    'description': "Full Moon"
                })
            
            # Check for Last Quarter (270° elongation)
            elif (prev_elongation < 270 and elongation >= 270):
                # Find exact crossing
                exact_jd = find_phase_crossing(prev_jd, current_jd, 270)
                
                phase_events.append({
                    'type': 'last_quarter',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': 270,
                    'description': "Last Quarter Moon"
                })
        
        # Update previous values
        prev_elongation = elongation
        prev_jd = current_jd
        
        # Move to next step
        current_jd += step
    
    # Sort events by date
    phase_events.sort(key=lambda x: x['julian_day'])
    
    return phase_events

def find_phase_crossing(jd1, jd2, target_phase):
    """
    Find the exact Julian day when a moon phase occurs.
    
    Args:
        jd1: Start Julian day
        jd2: End Julian day
        target_phase: Target phase angle (0, 90, 180, 270)
        
    Returns:
        Julian day of the phase
    """
    # Use bisection method to find the exact crossing
    precision = 0.001  # About 1.5 minutes
    
    while jd2 - jd1 > precision:
        mid_jd = (jd1 + jd2) / 2
        
        # Calculate elongation at midpoint
        sun_pos, _ = swe.calc_ut(mid_jd, swe.SUN, swe.FLG_SWIEPH)
        moon_pos, _ = swe.calc_ut(mid_jd, swe.MOON, swe.FLG_SWIEPH)
        
        # Calculate angular difference
        sun_lon = sun_pos[0]
        moon_lon = moon_pos[0]
        elongation = (moon_lon - sun_lon) % 360
        
        # Special case for new moon (0°) crossing
        if target_phase == 0:
            # We're looking for the transition from 350+ to 10- degrees
            if elongation > 180:  # We're before the new moon
                jd1 = mid_jd
            else:  # We're after the new moon
                jd2 = mid_jd
        else:
            # For other phases, compare directly with target
            if elongation < target_phase:
                jd1 = mid_jd
            else:
                jd2 = mid_jd
    
    return (jd1 + jd2) / 2

def find_apogee_perigee(start_date, end_date):
    """
    Find dates when Moon reaches apogee (farthest) or perigee (closest).
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        
    Returns:
        List of dictionaries with apogee/perigee events
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = get_julian_day(start_date)
    jd_end = get_julian_day(end_date)
    
    # Step size for scanning (6 hours initially)
    initial_step = 0.25  # 6 hours
    precision_step = 0.1 / 24.0  # ~6 minutes
    
    # Store found events
    distance_events = []
    
    # Set up flags
    flag = swe.FLG_SWIEPH
    
    # Track moon distance
    distances = []
    jds = []
    
    # Scan through the date range with initial step
    current_jd = jd_start
    while current_jd <= jd_end:
        # Calculate Moon's position
        moon_pos, status = swe.calc_ut(current_jd, swe.MOON, flag)
        
        # Extract distance (pos[2] is distance in AU)
        distance = moon_pos[2]
        
        distances.append(distance)
        jds.append(current_jd)
        
        # Keep only enough points to find extrema (3 is minimum)
        if len(distances) > 5:
            distances.pop(0)
            jds.pop(0)
        
        # Need at least 3 points to find extrema
        if len(distances) >= 3:
            # Check if middle point is an extremum
            if distances[1] < distances[0] and distances[1] < distances[2]:
                # Middle point is a minimum (perigee)
                # Refine to find exact time
                exact_jd = find_exact_extremum_distance(jds[0], jds[2], precision_step, find_min=True)
                
                # Get exact distance
                exact_pos, _ = swe.calc_ut(exact_jd, swe.MOON, flag)
                exact_dist = exact_pos[2]
                
                # Convert AU to km
                dist_km = exact_dist * 149597870.7
                
                distance_events.append({
                    'type': 'perigee',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': dist_km,
                    'description': f"Moon at perigee (closest: {dist_km:.0f} km)"
                })
                
                # Reset tracking to avoid double-counting
                distances = []
                jds = []
                
            elif distances[1] > distances[0] and distances[1] > distances[2]:
                # Middle point is a maximum (apogee)
                # Refine to find exact time
                exact_jd = find_exact_extremum_distance(jds[0], jds[2], precision_step, find_min=False)
                
                # Get exact distance
                exact_pos, _ = swe.calc_ut(exact_jd, swe.MOON, flag)
                exact_dist = exact_pos[2]
                
                # Convert AU to km
                dist_km = exact_dist * 149597870.7
                
                distance_events.append({
                    'type': 'apogee',
                    'date': get_date_from_julian(exact_jd),
                    'julian_day': exact_jd,
                    'value': dist_km,
                    'description': f"Moon at apogee (farthest: {dist_km:.0f} km)"
                })
                
                # Reset tracking to avoid double-counting
                distances = []
                jds = []
        
        # Move to next step
        current_jd += initial_step
    
    # Sort events by date
    distance_events.sort(key=lambda x: x['julian_day'])
    
    return distance_events

def find_exact_extremum_distance(jd1, jd2, step, find_min=True):
    """
    Find the Julian day where the lunar distance reaches its extremum (min or max).
    Uses a simple iterative search.
    
    Args:
        jd1: Starting Julian day
        jd2: Ending Julian day
        step: Step size for precision
        find_min: True to find minimum (perigee), False to find maximum (apogee)
        
    Returns:
        Julian day of the extremum
    """
    # Initialize ephemeris
    flag = swe.FLG_SWIEPH
    
    # Initial boundaries
    a = jd1
    b = jd2
    
    # Golden ratio for golden section search
    golden_ratio = (3 - 5**0.5) / 2
    
    # Calculate function to minimize/maximize
    def f(jd):
        pos, _ = swe.calc_ut(jd, swe.MOON, flag)
        distance = pos[2]
        # Negate distance for maximization problem
        return distance if find_min else -distance
    
    # Initialize points
    c = a + golden_ratio * (b - a)
    d = b - golden_ratio * (b - a)
    fc = f(c)
    fd = f(d)
    
    # Iterative refinement
    while abs(b - a) > step:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + golden_ratio * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - golden_ratio * (b - a)
            fd = f(d)
    
    # Return midpoint of final interval
    return (a + b) / 2

def find_void_of_course(start_date, end_date, precision_hours=1):
    """
    Find periods when Moon is void of course (between signs with no major aspects).
    
    Args:
        start_date: Beginning of search range
        end_date: End of search range
        precision_hours: Time precision for calculations
        
    Returns:
        List of dictionaries with void of course periods
    """
    # Initialize ephemeris
    initialize_ephemeris()
    
    # Convert dates to Julian days
    jd_start = get_julian_day(start_date)
    jd_end = get_julian_day(end_date)
    
    # Step size 
    step = precision_hours / 24.0  # Convert hours to days
    
    # Store found events
    voc_events = []
    
    # Set up flags
    flag = swe.FLG_SWIEPH
    
    # Define major aspects (in degrees)
    major_aspects = [0, 60, 90, 120, 180]  # Conjunction, Sextile, Square, Trine, Opposition
    
    # Define orb (allowed deviation from exact aspect)
    orb = 1.0  # 1 degree orb
    
    # Define planets to check for aspects
    planets = [swe.SUN, swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO]
    
    # Track VOC status
    is_voc = False
    voc_start = None
    last_aspect_jd = None
    
    # Scan through the date range
    current_jd = jd_start
    while current_jd <= jd_end:
        # Get Moon's current sign
        moon_pos, _ = swe.calc_ut(current_jd, swe.MOON, flag)
        moon_lon = moon_pos[0]
        current_sign = int(moon_lon / 30)
        sign_position = moon_lon % 30
        
        # Will the Moon change signs soon?
        will_change_signs = sign_position > 29
        
        # Calculate remaining degrees until sign change
        degrees_to_sign_change = 30 - sign_position
        
        # Approximately how many days until sign change at current speed
        moon_speed = moon_pos[3] if len(moon_pos) > 3 else 13.0  # Default ~13°/day if speed not available
        
        # Protect against division by zero
        if abs(moon_speed) < 0.001:  # If speed is effectively zero
            days_to_sign_change = 0.5  # Default to half a day
        else:
            days_to_sign_change = degrees_to_sign_change / abs(moon_speed)
        
        # If Moon is about to change signs, check if it's void of course
        if will_change_signs or (days_to_sign_change < 0.5):  # Within ~12 hours of sign change
            # Check if Moon forms any major aspects before changing signs
            has_aspect = False
            
            # Create a test window from current time to sign change
            test_window = np.arange(current_jd, current_jd + days_to_sign_change * 1.1, step)
            
            for test_jd in test_window:
                # Check if we've already moved to next sign
                test_moon_pos, _ = swe.calc_ut(test_jd, swe.MOON, flag)
                test_moon_lon = test_moon_pos[0]
                test_sign = int(test_moon_lon / 30)
                
                if test_sign != current_sign:
                    # We've changed signs, stop checking
                    break
                
                # Check for aspects with each planet
                for planet in planets:
                    planet_pos, _ = swe.calc_ut(test_jd, planet, flag)
                    planet_lon = planet_pos[0]
                    
                    # Calculate angle difference
                    diff = abs(test_moon_lon - planet_lon) % 360
                    diff = min(diff, 360 - diff)
                    
                    # Check if difference is close to any major aspect
                    for aspect in major_aspects:
                        if abs(diff - aspect) <= orb:
                            has_aspect = True
                            last_aspect_jd = test_jd
                            break
                    
                    if has_aspect:
                        break
                
                if has_aspect:
                    break
            
            # If no aspects found before sign change, Moon is VOC
            if not has_aspect and not is_voc:
                # Start of VOC period
                is_voc = True
                if last_aspect_jd is not None:
                    voc_start = last_aspect_jd
                else:
                    # If we don't have a last aspect time, use current time
                    voc_start = current_jd
                
                # Get the date and positions for description
                start_date = get_date_from_julian(voc_start)
                moon_pos_start, _ = swe.calc_ut(voc_start, swe.MOON, flag)
                moon_sign_start = get_zodiac_sign(moon_pos_start[0])
                
                # Add event for VOC start
                voc_events.append({
                    'type': 'voc_start',
                    'date': start_date,
                    'julian_day': voc_start,
                    'value': moon_pos_start[0],  # Moon longitude
                    'description': f"Moon goes void of course in {moon_sign_start}"
                })
        
        # Check if Moon is changing signs now
        if is_voc:
            # Get Moon's next sign
            next_jd = current_jd + step
            next_moon_pos, _ = swe.calc_ut(next_jd, swe.MOON, flag)
            next_moon_lon = next_moon_pos[0]
            next_sign = int(next_moon_lon / 30)
            
            if next_sign != current_sign:
                # Moon is entering a new sign, VOC period ends
                is_voc = False
                
                # Get the date and positions for description
                end_date = get_date_from_julian(next_jd)
                moon_sign_end = get_zodiac_sign(next_moon_lon)
                
                # Add event for VOC end
                voc_events.append({
                    'type': 'voc_end',
                    'date': end_date,
                    'julian_day': next_jd,
                    'value': next_moon_lon,  # Moon longitude
                    'description': f"Moon enters {moon_sign_end}, void of course ends"
                })
        
        # Move to next step
        current_jd += step
    
    # If VOC is still active at the end of the range, add an ending event
    if is_voc and voc_start is not None:
        voc_events.append({
            'type': 'voc_end',
            'date': get_date_from_julian(jd_end),
            'julian_day': jd_end,
            'value': None,
            'description': "End of analysis period (Moon still void of course)"
        })
    
    # Sort events by date
    voc_events.sort(key=lambda x: x['julian_day'])
    
    return voc_events

def get_zodiac_sign(longitude):
    """Convert longitude to zodiac sign."""
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer",
        "Leo", "Virgo", "Libra", "Scorpio",
        "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    
    # Normalize longitude to 0-360 range
    norm_longitude = fmod(longitude, 360.0)
    if norm_longitude < 0:
        norm_longitude += 360.0
    
    # Determine sign index (each sign is 30 degrees)
    sign_index = int(norm_longitude / 30)
    
    return signs[sign_index]

def organize_output_directories(market_symbol, start_date, end_date):
    """
    Create an organized directory structure for analysis outputs.
    
    Args:
        market_symbol: Symbol for the market being analyzed
        start_date: Start date of analysis
        end_date: End date of analysis
        
    Returns:
        Dictionary with paths to different output directories
    """
    # Format dates for folder name
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create main directory structure
    base_dir = "LUNAR_CYCLE"  # Root folder in script directory
    analysis_dir = f"{base_dir}/lunar_analysis_{market_symbol}_{start_str}_{end_str}"
    csv_dir = f"{analysis_dir}/CSV"
    visualization_dir = f"{analysis_dir}/Visualizations"
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Return paths
    return {
        'base_dir': base_dir,
        'analysis_dir': analysis_dir,
        'csv_dir': csv_dir,
        'visualization_dir': visualization_dir
    }

def identify_lunar_clusters(lunar_events, threshold=2):
    """
    Identify dates when multiple lunar events occur together.
    
    Args:
        lunar_events: Dictionary of all lunar events by type
        threshold: Minimum number of events to consider a date a "cluster"
        
    Returns:
        List of dictionaries with cluster information
    """
    # First, convert events to a date-based dictionary
    events_by_date = {}
    
    # Combine all events from different categories
    all_events = []
    for event_type, events in lunar_events.items():
        all_events.extend(events)
    
    # Group by date
    for event in all_events:
        date = event['date']
        if date not in events_by_date:
            events_by_date[date] = []
        
        events_by_date[date].append(event)
    
    # Identify clusters based on threshold
    clusters = []
    
    for date, events in events_by_date.items():
        if len(events) >= threshold:
            # Create a cluster object
            event_types = [e['type'] for e in events]
            descriptions = [e['description'] for e in events]
            
            clusters.append({
                'date': date,
                'events': events,
                'event_count': len(events),
                'event_types': event_types,
                'descriptions': descriptions
            })
    
    # Sort clusters by date
    clusters.sort(key=lambda x: x['date'])
    
    return clusters

def get_market_data(symbol, event_dates, window_days=10):
    """
    Get market data around lunar event clusters.
    
    Args:
        symbol: Market symbol to analyze
        event_dates: List of dates when events occur
        window_days: Number of days to analyze before/after each event
        
    Returns:
        Dictionary mapping event dates to market data
    """
    # Format symbol for Yahoo Finance
    if symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
        yf_symbol = '^GSPC'
    elif symbol.upper() in ['DJI', 'DJIA', 'DOW']:
        yf_symbol = '^DJI'
    elif symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
        yf_symbol = '^IXIC'
    elif symbol.upper() in ['RUT', 'RUSSELL', 'RUSSELL2000']:
        yf_symbol = '^RUT'   # Russell 2000
    elif symbol.upper() in ['GOLD', 'XAU']:
        yf_symbol = 'GC=F'  # Gold Futures
    elif symbol.upper() in ['SILVER', 'XAG']:
        yf_symbol = 'SI=F'  # Silver Futures
    elif symbol.upper() in ['WHEAT', 'ZW', 'W']:
        yf_symbol = 'ZW=F'   # Wheat Futures
    elif symbol.upper() in ['CORN', 'ZC', 'C']:
        yf_symbol = 'ZC=F'   # Corn Futures
    elif symbol.upper() in ['SOYBEANS', 'ZS', 'S']:
        yf_symbol = 'ZS=F'   # Soybean Futures
    elif symbol.upper() in ['COTTON', 'CT']:
        yf_symbol = 'CT=F'   # Cotton Futures
    elif symbol.upper() in ['SUGAR', 'SB']:
        yf_symbol = 'SB=F'   # Sugar Futures
    elif symbol.upper() in ['COFFEE', 'KC']:
        yf_symbol = 'KC=F'   # Coffee Futures
    elif symbol.upper() in ['COCOA', 'CC']:
        yf_symbol = 'CC=F'   # Cocoa Futures
    else:
        yf_symbol = symbol
    
    # Find the earliest and latest dates we need data for
    earliest_date = min(event_dates) - datetime.timedelta(days=window_days)
    latest_date = max(event_dates) + datetime.timedelta(days=window_days)
    
    # Get all market data for the full range
    try:
        all_data = yf.download(yf_symbol, start=earliest_date, end=latest_date, progress=False)
        
        if all_data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return {}
        
        # Prepare result dictionary
        result = {}
        
        # For each event date, extract the relevant window of data
        for event_date in event_dates:
            start_date = event_date - datetime.timedelta(days=window_days)
            end_date = event_date + datetime.timedelta(days=window_days)
            
            # Get data within this window
            mask = (all_data.index >= pd.Timestamp(start_date)) & (all_data.index <= pd.Timestamp(end_date))
            window_data = all_data.loc[mask].copy()
            
            # Calculate the daily percent change
            window_data['Pct_Change'] = window_data['Close'].pct_change() * 100
            
            # Calculate days relative to event date
            window_data['Days_From_Event'] = [(d.date() - event_date).days for d in window_data.index]
            
            # Store in result
            result[event_date] = window_data
        
        return result
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return {}

def analyze_market_behavior(market_data, event_date):
    """
    Analyze market behavior around a lunar event.
    
    Args:
        market_data: Market data around event date
        event_date: Date of the event
        
    Returns:
        Dictionary with analysis results
    """
    if event_date not in market_data or market_data[event_date].empty:
        return {
            'has_data': False,
            'price_at_event': None,
            'is_turning_point': False,
            'pattern': None,
            'before_change': None,
            'after_change': None,
            'volatility_before': None,
            'volatility_after': None
        }
    
    data = market_data[event_date]
    
    # Find the closest trading day to the event date
    closest_idx = data['Days_From_Event'].abs().idxmin()
    event_day_offset = data.loc[closest_idx, 'Days_From_Event']
    
    # Get price at event date
    price_at_event = float(data.loc[closest_idx, 'Close'])
    
    # Define window sizes for analysis
    pre_window = 5  # Days before
    post_window = 5  # Days after
    
    # Extract pre and post windows
    pre_data = data[data['Days_From_Event'] < 0].tail(pre_window)
    post_data = data[data['Days_From_Event'] > 0].head(post_window)
    
    # Check if we have enough data for analysis
    if len(pre_data) < 3 or len(post_data) < 3:
        return {
            'has_data': True,
            'price_at_event': price_at_event,
            'is_turning_point': False,
            'pattern': 'Insufficient data',
            'before_change': None,
            'after_change': None,
            'volatility_before': None,
            'volatility_after': None
        }
    
    # Check if it's a local maximum (high)
    pre_max = float(pre_data['Close'].max())
    post_max = float(post_data['Close'].max())
    
    # Check if it's a local minimum (low)
    pre_min = float(pre_data['Close'].min())
    post_min = float(post_data['Close'].min())
    
    # Calculate percentage changes before and after
    if len(pre_data) > 0:
        pre_first = float(pre_data['Close'].iloc[0])
        pre_last = float(pre_data['Close'].iloc[-1])
        before_change = ((pre_last - pre_first) / pre_first) * 100
    else:
        before_change = None
    
    if len(post_data) > 0:
        post_first = float(post_data['Close'].iloc[0])
        post_last = float(post_data['Close'].iloc[-1])
        after_change = ((post_last - post_first) / post_first) * 100
    else:
        after_change = None
    
    # Calculate volatility (standard deviation of percent changes)
    volatility_before = float(pre_data['Pct_Change'].std()) if len(pre_data) > 0 else None
    volatility_after = float(post_data['Pct_Change'].std()) if len(post_data) > 0 else None
    
    # Determine if it's a turning point
    is_high = price_at_event >= pre_max * 0.99 and price_at_event >= post_max * 0.99
    is_low = price_at_event <= pre_min * 1.01 and price_at_event <= post_min * 1.01
    
    # Determine pattern
    if is_high:
        pattern = 'High'
    elif is_low:
        pattern = 'Low'
    else:
        # Calculate trend before and after
        pre_trend = (pre_last - pre_first) / pre_first if len(pre_data) > 0 else 0
        post_trend = (post_last - post_first) / post_first if len(post_data) > 0 else 0
        
        # Check for trend changes
        if pre_trend > 0.01 and post_trend < -0.01:
            pattern = 'Uptrend to Downtrend'
        elif pre_trend < -0.01 and post_trend > 0.01:
            pattern = 'Downtrend to Uptrend'
        elif pre_trend > 0.01 and post_trend > 0.01:
            pattern = 'Continued Uptrend'
        elif pre_trend < -0.01 and post_trend < -0.01:
            pattern = 'Continued Downtrend'
        else:
            pattern = 'Sideways'
    
    return {
        'has_data': True,
        'price_at_event': price_at_event,
        'is_turning_point': is_high or is_low,
        'pattern': pattern,
        'before_change': before_change,
        'after_change': after_change,
        'volatility_before': volatility_before,
        'volatility_after': volatility_after
    }

def visualize_results(lunar_clusters, market_data, market_symbol, visualization_dir):
    """
    Create visualizations of market behavior around lunar clusters.
    
    Args:
        lunar_clusters: List of lunar event clusters
        market_data: Dictionary of market data around cluster dates
        market_symbol: Symbol for the market being analyzed
        visualization_dir: Directory where visualizations will be saved
    """
    # DO NOT create a new directory here - use the one passed in
    # Ensure it exists
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Event type colors for visualization
    event_colors = {
        'declination_max': '#FF0000',      # Bright red
        'declination_min': '#FF6600',      # Bright orange
        'declination_zero': '#CC6600',     # Darker orange/brown
        'true_node_retrograde_to_direct': '#006600',  # Dark green
        'true_node_direct_to_retrograde': '#0000CC',  # Dark blue
        'new_moon': '#660066',             # Dark purple
        'full_moon': '#000000',            # Black (for "white")
        'first_quarter': '#333333',        # Dark gray
        'last_quarter': '#663300',         # Dark brown
        'apogee': '#006666',               # Dark teal
        'perigee': '#990099',              # Dark magenta
        'voc_start': '#009999',            # Dark cyan
        'voc_end': '#CC0066'               # Dark pink
    }
    
    # Create individual charts for each cluster
    for i, cluster in enumerate(lunar_clusters):
        cluster_date = cluster['date']
        
        if cluster_date not in market_data or market_data[cluster_date].empty:
            print(f"No market data available for cluster on {cluster_date}")
            continue
        
        data = market_data[cluster_date]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price data
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'])
        
        # Mark cluster date
        closest_idx = data['Days_From_Event'].abs().idxmin()
        event_price = data.loc[closest_idx, 'Close']
        
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.plot(closest_idx, event_price, 'ro', markersize=8)
        
        # Add annotation for each event in the cluster
        y_offset = 0
        # Keep track of descriptions we've already added to avoid duplicates
        added_descriptions = set()
        for event in cluster['events']:
            event_type = event['type']
            description = event['description']

            # Skip empty or None descriptions
            if not description or description.strip() == "":
                continue
            
            # Skip if we've already added this exact description
            if description in added_descriptions:
                continue
            
            added_descriptions.add(description)
            color = event_colors.get(event_type, 'black')
            
            plt.annotate(description, 
                        xy=(closest_idx, event_price), 
                        xytext=(10, 10 + y_offset),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            
            y_offset += 15
        
        # Add title
        plt.title(f"Lunar Cluster on {cluster_date.strftime('%Y-%m-%d')} - {cluster['event_count']} Events")
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Plot percent change
        plt.subplot(2, 1, 2)
        plt.bar(data.index, data['Pct_Change'])
        plt.axvline(x=closest_idx, color='r', linestyle='--')
        plt.ylabel('Daily % Change')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart - use the provided visualization_dir
        chart_filename = f"{visualization_dir}/cluster_{i+1}_{cluster_date.strftime('%Y-%m-%d')}.png"
        plt.savefig(chart_filename)
        print(f"Chart saved as {chart_filename}")
        
        plt.close()
    
    # Create a summary chart showing all clusters on a timeline
    if lunar_clusters:
        # Get min and max dates from all clusters
        all_dates = [cluster['date'] for cluster in lunar_clusters]
        min_date = min(all_dates) - datetime.timedelta(days=30)
        max_date = max(all_dates) + datetime.timedelta(days=30)
        
        # Get market data for entire period
        try:
            if market_symbol.upper() in ['SPX', 'S&P500', 'S&P 500']:
                yf_symbol = '^GSPC'
            elif market_symbol.upper() in ['DJI', 'DJIA', 'DOW']:
                yf_symbol = '^DJI'
            elif market_symbol.upper() in ['IXIC', 'NASDAQ', 'NDX']:
                yf_symbol = '^IXIC'
            else:
                yf_symbol = market_symbol
                
            full_data = yf.download(yf_symbol, start=min_date, end=max_date, progress=False)
            
            if not full_data.empty:
                plt.figure(figsize=(16, 10))
                
                # Plot the market for the entire period
                plt.plot(full_data.index, full_data['Close'], 'k-', alpha=0.7)
                
                # Mark each cluster date
                for i, cluster in enumerate(lunar_clusters):
                    cluster_date = cluster['date']
                    
                    # Find the closest market date
                    closest_date = None
                    min_diff = float('inf')
                    
                    for date_idx in full_data.index:
                        date_diff = abs((date_idx.date() - cluster_date).days)
                        if date_diff < min_diff:
                            min_diff = date_diff
                            closest_date = date_idx
                    
                    if closest_date is not None:
                        # Get price on that date
                        price = full_data.loc[closest_date, 'Close']
                        
                        # Mark on chart
                        plt.axvline(x=closest_date, color='r', linestyle='--', alpha=0.5)
                        plt.plot(closest_date, price, 'ro', markersize=8)
                        
                        # Add annotation
                        plt.annotate(f"Cluster {i+1}: {cluster['event_count']} events\n{cluster_date.strftime('%Y-%m-%d')}\n${float(price):.2f}", 
                            xy=(closest_date, price), 
                            xytext=(0, 20),
                            textcoords="offset points",
                            fontsize=8,
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                plt.title(f"All Lunar Clusters - {market_symbol}")
                plt.ylabel('Price')
                plt.grid(True, alpha=0.3)
                
                # Format x-axis to show dates nicely
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.gcf().autofmt_xdate()
                
                # Save the summary chart - use the provided visualization_dir
                summary_filename = f"{visualization_dir}/summary_chart.png"
                plt.savefig(summary_filename)
                print(f"Summary chart saved as {summary_filename}")
                
                plt.close()
        except Exception as e:
            print(f"Error creating summary chart: {e}")

def calculate_statistics(lunar_clusters, market_analyses):
    """
    Calculate statistics on lunar cluster effectiveness.
    
    Args:
        lunar_clusters: List of lunar clusters
        market_analyses: Dictionary of market analyses for each cluster
        
    Returns:
        Dictionary with statistical results
    """
    stats = {
        'total_clusters': len(lunar_clusters),
        'clusters_with_data': 0,
        'turning_points': 0,
        'patterns': {
            'High': 0,
            'Low': 0,
            'Uptrend to Downtrend': 0,
            'Downtrend to Uptrend': 0,
            'Continued Uptrend': 0,
            'Continued Downtrend': 0,
            'Sideways': 0,
            'Insufficient data': 0
        },
        'avg_before_change': None,
        'avg_after_change': None,
        'avg_volatility_before': None,
        'avg_volatility_after': None,
        'event_type_effectiveness': {}
    }
    
    # Collect data for calculations
    before_changes = []
    after_changes = []
    volatility_before = []
    volatility_after = []
    
    # Track effectiveness by event types
    event_type_counts = {}
    event_type_turning_points = {}
    
    # Analyze each cluster
    for cluster in lunar_clusters:
        cluster_date = cluster['date']
        
        if cluster_date in market_analyses:
            analysis = market_analyses[cluster_date]
            
            if analysis['has_data']:
                stats['clusters_with_data'] += 1
                
                if analysis['is_turning_point']:
                    stats['turning_points'] += 1
                
                pattern = analysis['pattern']
                if pattern in stats['patterns']:
                    stats['patterns'][pattern] += 1
                
                # Collect percentage changes and volatility
                if analysis['before_change'] is not None:
                    before_changes.append(analysis['before_change'])
                
                if analysis['after_change'] is not None:
                    after_changes.append(analysis['after_change'])
                
                if analysis['volatility_before'] is not None:
                    volatility_before.append(analysis['volatility_before'])
                
                if analysis['volatility_after'] is not None:
                    volatility_after.append(analysis['volatility_after'])
                
                # Track by event types
                for event_type in cluster['event_types']:
                    if event_type not in event_type_counts:
                        event_type_counts[event_type] = 0
                        event_type_turning_points[event_type] = 0
                    
                    event_type_counts[event_type] += 1
                    
                    if analysis['is_turning_point']:
                        event_type_turning_points[event_type] += 1
    
    # Calculate averages
    if before_changes:
        stats['avg_before_change'] = sum(before_changes) / len(before_changes)
    
    if after_changes:
        stats['avg_after_change'] = sum(after_changes) / len(after_changes)
    
    if volatility_before:
        stats['avg_volatility_before'] = sum(volatility_before) / len(volatility_before)
    
    if volatility_after:
        stats['avg_volatility_after'] = sum(volatility_after) / len(volatility_after)
    
    # Calculate effectiveness by event type
    for event_type, count in event_type_counts.items():
        turning_points = event_type_turning_points[event_type]
        effectiveness = (turning_points / count) * 100 if count > 0 else 0
        
        stats['event_type_effectiveness'][event_type] = {
            'count': count,
            'turning_points': turning_points,
            'effectiveness': effectiveness
        }
    
    return stats

def display_results(lunar_clusters, market_analyses, statistics, market_symbol):
    """Display the analysis results in a formatted table."""
    print("\n" + "="*100)
    print(f"LUNAR CYCLE ANALYSIS FOR {market_symbol}")
    print("="*100)
    
    if not lunar_clusters:
        print("No lunar clusters found matching the specified criteria.")
        return
    
    print("\nLUNAR CLUSTERS:")
    print("-"*100)
    print(f"{'Date':12} | {'Events':5} | {'Event Types':40} | {'Price':10} | {'Pattern':20} | {'Turning Point':12}")
    print("-"*100)
    
    for cluster in lunar_clusters:
        cluster_date = cluster['date']
        
        # Format event types for display
        event_types = ", ".join([e.split('_')[0] for e in cluster['event_types']])
        if len(event_types) > 38:
            event_types = event_types[:35] + "..."
        
        # Get analysis if available
        if cluster_date in market_analyses:
            analysis = market_analyses[cluster_date]
        else:
            analysis = {'has_data': False, 'price_at_event': None, 'is_turning_point': False, 'pattern': None}
        
        # Format fields
        date_str = cluster_date.strftime('%Y-%m-%d')
        event_count = str(cluster['event_count'])
        
        if analysis['has_data']:
            price = f"${analysis['price_at_event']:.2f}" if analysis['price_at_event'] is not None else "N/A"
            pattern = analysis['pattern'] or "N/A"
            turning_point = "Yes" if analysis['is_turning_point'] else "No"
        else:
            price = "N/A"
            pattern = "No data"
            turning_point = "N/A"
        
        # Print row
        print(f"{date_str:12} | {event_count:5} | {event_types:40} | {price:10} | {pattern:20} | {turning_point:12}")
    
    print("\nSTATISTICAL SUMMARY:")
    print("-"*100)
    print(f"Total Clusters: {statistics['total_clusters']}")
    print(f"Clusters with Market Data: {statistics['clusters_with_data']}")
    
    if statistics['clusters_with_data'] > 0:
        turning_point_pct = (statistics['turning_points'] / statistics['clusters_with_data']) * 100
        print(f"Turning Points: {statistics['turning_points']} ({turning_point_pct:.1f}%)")
        
        print("\nPattern Distribution:")
        for pattern, count in statistics['patterns'].items():
            if count > 0:
                pct = (count / statistics['clusters_with_data']) * 100
                print(f"  {pattern}: {count} ({pct:.1f}%)")
        
        print("\nMarket Behavior:")
        if statistics['avg_before_change'] is not None:
            print(f"  Avg % Change Before: {statistics['avg_before_change']:.2f}%")
        if statistics['avg_after_change'] is not None:
            print(f"  Avg % Change After: {statistics['avg_after_change']:.2f}%")
        if statistics['avg_volatility_before'] is not None and statistics['avg_volatility_after'] is not None:
            volatility_change = ((statistics['avg_volatility_after'] / statistics['avg_volatility_before']) - 1) * 100
            print(f"  Volatility Change: {volatility_change:.2f}%")
        
        print("\nEvent Type Effectiveness:")
        for event_type, data in sorted(statistics['event_type_effectiveness'].items(), 
                                     key=lambda x: x[1]['effectiveness'], reverse=True):
            print(f"  {event_type}: {data['effectiveness']:.1f}% ({data['turning_points']}/{data['count']})")
    
    print("="*100)

def save_to_csv(lunar_clusters, market_analyses, statistics, market_symbol, csv_dir):
    """
    Save analysis results to CSV files.
    
    Args:
        lunar_clusters: List of lunar clusters
        market_analyses: Dictionary of market analyses for each cluster
        statistics: Statistics calculated from the analysis
        market_symbol: Symbol for the market being analyzed
        csv_dir: Directory to save CSV files
    """
    # Create clusters_data globally for access by save_console_output
    global clusters_data
    clusters_data = []
    
    for cluster in lunar_clusters:
        cluster_date = cluster['date']
        
        # Get analysis if available
        if cluster_date in market_analyses:
            analysis = market_analyses[cluster_date]
        else:
            analysis = {'has_data': False, 'price_at_event': None, 'is_turning_point': False, 
                       'pattern': None, 'before_change': None, 'after_change': None,
                       'volatility_before': None, 'volatility_after': None}
        
        # Create row
        row = {
            'Date': cluster_date,
            'Event_Count': cluster['event_count'],
            'Event_Types': ", ".join(cluster['event_types']),
            'Descriptions': " | ".join(cluster['descriptions']),
            'Has_Market_Data': analysis['has_data'],
            'Price': analysis['price_at_event'],
            'Is_Turning_Point': analysis['is_turning_point'],
            'Pattern': analysis['pattern'],
            'Before_Change_Pct': analysis['before_change'],
            'After_Change_Pct': analysis['after_change'],
            'Volatility_Before': analysis['volatility_before'],
            'Volatility_After': analysis['volatility_after']
        }
        
        clusters_data.append(row)
    
    # Create DataFrame and save
    if clusters_data:
        df = pd.DataFrame(clusters_data)
        
        # Save clusters file
        clusters_filename = f"{csv_dir}/{market_symbol}_lunar_clusters.csv"
        df.to_csv(clusters_filename, index=False)
        print(f"Clusters data saved to {clusters_filename}")
    
    # Save statistics to CSV
    stats_data = {
        'Metric': [
            'Total_Clusters',
            'Clusters_With_Data',
            'Turning_Points',
            'Turning_Points_Pct',
            'Pattern_High',
            'Pattern_Low',
            'Pattern_Uptrend_to_Downtrend',
            'Pattern_Downtrend_to_Uptrend',
            'Pattern_Continued_Uptrend',
            'Pattern_Continued_Downtrend',
            'Pattern_Sideways',
            'Pattern_Insufficient_Data',
            'Avg_Before_Change_Pct',
            'Avg_After_Change_Pct',
            'Avg_Volatility_Before',
            'Avg_Volatility_After',
            'Volatility_Change_Pct'
        ],
        'Value': [
            statistics['total_clusters'],
            statistics['clusters_with_data'],
            statistics['turning_points'],
            (statistics['turning_points'] / statistics['clusters_with_data'] * 100) if statistics['clusters_with_data'] > 0 else 0,
            statistics['patterns'].get('High', 0),
            statistics['patterns'].get('Low', 0),
            statistics['patterns'].get('Uptrend to Downtrend', 0),
            statistics['patterns'].get('Downtrend to Uptrend', 0),
            statistics['patterns'].get('Continued Uptrend', 0),
            statistics['patterns'].get('Continued Downtrend', 0),
            statistics['patterns'].get('Sideways', 0),
            statistics['patterns'].get('Insufficient data', 0),
            statistics['avg_before_change'],
            statistics['avg_after_change'],
            statistics['avg_volatility_before'],
            statistics['avg_volatility_after'],
            ((statistics['avg_volatility_after'] / statistics['avg_volatility_before']) - 1) * 100 
                if (statistics['avg_volatility_before'] is not None and 
                    statistics['avg_volatility_after'] is not None and 
                    statistics['avg_volatility_before'] != 0) else None
        ]
    }
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    stats_filename = f"{csv_dir}/{market_symbol}_statistics.csv"
    stats_df.to_csv(stats_filename, index=False)
    print(f"Statistics saved to {stats_filename}")
    
    # Save event type effectiveness
    effectiveness_data = []
    
    for event_type, data in statistics['event_type_effectiveness'].items():
        effectiveness_data.append({
            'Event_Type': event_type,
            'Count': data['count'],
            'Turning_Points': data['turning_points'],
            'Effectiveness_Pct': data['effectiveness']
        })
    
    if effectiveness_data:
        eff_df = pd.DataFrame(effectiveness_data)
        eff_filename = f"{csv_dir}/{market_symbol}_event_effectiveness.csv"
        eff_df.to_csv(eff_filename, index=False)
        print(f"Event effectiveness data saved to {eff_filename}")
    
    # Save console output
    from datetime import datetime
    current_date = datetime.now().date()
    save_console_output(statistics, market_symbol, current_date, current_date, csv_dir)

def save_console_output(statistics, market_symbol, start_date, end_date, csv_dir):
    """
    Save the console output to a text file.
    
    Args:
        statistics: Statistics calculated from the analysis
        market_symbol: Symbol for the market being analyzed
        start_date: Start date of analysis period (or current date if not available)
        end_date: End date of analysis period (or current date if not available)
        csv_dir: Directory to save the text file
    """
    # Create console output similar to what's printed to the console
    output = []
    output.append("=" * 100)
    output.append(f"LUNAR CYCLE ANALYSIS FOR {market_symbol}")
    output.append("=" * 100)
    output.append("\nLUNAR CLUSTERS:")
    output.append("-" * 100)
    output.append(f"{'Date':12} | {'Events':5} | {'Event Types':40} | {'Price':10} | {'Pattern':20} | {'Turning Point':12}")
    output.append("-" * 100)
    
    # Add cluster data using the global clusters_data
    if 'clusters_data' in globals():
        for row in clusters_data:
            date_str = row['Date'].strftime('%Y-%m-%d')
            event_count = str(row['Event_Count'])
            event_types = row['Event_Types']
            if len(event_types) > 38:
                event_types = event_types[:35] + "..."
            
            price = f"${row['Price']:.2f}" if row['Price'] is not None else "N/A"
            pattern = row['Pattern'] or "N/A"
            turning_point = "Yes" if row['Is_Turning_Point'] else "No"
            
            output.append(f"{date_str:12} | {event_count:5} | {event_types:40} | {price:10} | {pattern:20} | {turning_point:12}")
    
    output.append("\nSTATISTICAL SUMMARY:")
    output.append("-" * 100)
    output.append(f"Total Clusters: {statistics['total_clusters']}")
    output.append(f"Clusters with Market Data: {statistics['clusters_with_data']}")
    
    if statistics['clusters_with_data'] > 0:
        turning_point_pct = (statistics['turning_points'] / statistics['clusters_with_data']) * 100
        output.append(f"Turning Points: {statistics['turning_points']} ({turning_point_pct:.1f}%)")
        
        output.append("\nPattern Distribution:")
        for pattern, count in statistics['patterns'].items():
            if count > 0:
                pct = (count / statistics['clusters_with_data']) * 100
                output.append(f"  {pattern}: {count} ({pct:.1f}%)")
        
        output.append("\nMarket Behavior:")
        if statistics['avg_before_change'] is not None:
            output.append(f"  Avg % Change Before: {statistics['avg_before_change']:.2f}%")
        if statistics['avg_after_change'] is not None:
            output.append(f"  Avg % Change After: {statistics['avg_after_change']:.2f}%")
        if statistics['avg_volatility_before'] is not None and statistics['avg_volatility_after'] is not None:
            volatility_change = ((statistics['avg_volatility_after'] / statistics['avg_volatility_before']) - 1) * 100
            output.append(f"  Volatility Change: {volatility_change:.2f}%")
        
        output.append("\nEvent Type Effectiveness:")
        for event_type, data in sorted(statistics['event_type_effectiveness'].items(), 
                                     key=lambda x: x[1]['effectiveness'], reverse=True):
            output.append(f"  {event_type}: {data['effectiveness']:.1f}% ({data['turning_points']}/{data['count']})")
    
    output.append("=" * 100)
    
    # Create filename and save
    filename = f"{csv_dir}/{market_symbol}_console_log.txt"
    with open(filename, 'w') as f:
        f.write('\n'.join(output))
    print(f"Console output saved to {filename}")

def main():
    """Main function to run the tool."""
    print("Lunar Cycle Analysis Tool")
    print("------------------------")
    
    # Get date range
    print("Enter start date (YYYY-MM-DD):")
    start_date_str = input()
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Using default of 1 year ago.")
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=365)
    
    print("Enter end date (YYYY-MM-DD):")
    end_date_str = input()
    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Using default of today.")
        end_date = datetime.datetime.now().date()
    
    # Get market symbol
    print("Enter symbol (e.g., SPX, NASDAQ, DJI, GOLD, SILVER, WHEAT, CORN, SOYBEANS, COTTON, SUGAR, COFFEE, COCOA):")
    market_symbol = input()
    
    # Get lunar factors to analyze
    print("Select lunar factors to analyze (comma-separated, e.g., 1,2,3,4,5):")
    print("1. Moon declination (max/zero)")
    print("2. True Node direction changes")
    print("3. Moon phases (New, Full, Quarter)")
    print("4. Apogee/Perigee")
    print("5. Moon void of course")
    print("Or enter 'all' to include all factors:")
    
    factors_input = input().strip()
    if factors_input.lower() == 'all':
        lunar_factors = [1, 2, 3, 4, 5]
    else:
        try:
            lunar_factors = [int(f.strip()) for f in factors_input.split(',')]
        except ValueError:
            print("Invalid input. Using all factors as default.")
            lunar_factors = [1, 2, 3, 4, 5]
    
    # Get cluster threshold
    print("Minimum number of lunar factors required to consider a date as a 'cluster' (2-5):")
    try:
        cluster_threshold = int(input())
        if cluster_threshold < 2 or cluster_threshold > 5:
            print("Value out of range. Using default of 2.")
            cluster_threshold = 2
    except ValueError:
        print("Invalid input. Using default of 2.")
        cluster_threshold = 2
    
    # Get analysis window
    print("Number of days to analyze before/after each lunar cluster (default 10):")
    try:
        window_days = int(input())
        if window_days < 1:
            print("Value too small. Using default of 10.")
            window_days = 10
    except ValueError:
        print("Invalid input. Using default of 10.")
        window_days = 10

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create organized directory structure
    directories = organize_output_directories(market_symbol, start_date, end_date)
    
    # Start calculating lunar events
    print("\nCalculating lunar events... (this may take a few minutes)")
    start_time = time.time()
    
    lunar_events = {}
    
    # Calculate each selected lunar factor
    if 1 in lunar_factors:
        print("Finding Moon declination events...")
        lunar_events['declination'] = find_declination_events(start_date, end_date)
        print(f"Found {len(lunar_events['declination'])} declination events.")
    
    if 2 in lunar_factors:
        print("Finding True Node direction changes...")
        lunar_events['true_node'] = find_true_node_direction_changes(start_date, end_date)
        print(f"Found {len(lunar_events['true_node'])} True Node direction changes.")
    
    if 3 in lunar_factors:
        print("Finding Moon phases...")
        lunar_events['moon_phase'] = find_moon_phases(start_date, end_date)
        print(f"Found {len(lunar_events['moon_phase'])} Moon phases.")
    
    if 4 in lunar_factors:
        print("Finding Apogee/Perigee points...")
        lunar_events['distance'] = find_apogee_perigee(start_date, end_date)
        print(f"Found {len(lunar_events['distance'])} distance extremes.")
    
    if 5 in lunar_factors:
        print("Finding Moon void of course periods...")
        lunar_events['voc'] = find_void_of_course(start_date, end_date)
        print(f"Found {len(lunar_events['voc'])} void of course events.")
    
    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    print(f"Calculations completed in {elapsed_time:.1f} seconds.")
    
    # Identify clusters
    print("\nIdentifying lunar clusters...")
    lunar_clusters = identify_lunar_clusters(lunar_events, threshold=cluster_threshold)
    print(f"Found {len(lunar_clusters)} lunar clusters.")
    
    if not lunar_clusters:
        print("No clusters found matching the criteria. Try reducing the threshold or including more factors.")
        return
    
    # Get market data
    print("\nRetrieving market data...")
    cluster_dates = [cluster['date'] for cluster in lunar_clusters]
    market_data = get_market_data(market_symbol, cluster_dates, window_days)
    
    # Analyze market behavior
    print("\nAnalyzing market behavior...")
    market_analyses = {}
    for cluster_date in cluster_dates:
        market_analyses[cluster_date] = analyze_market_behavior(market_data, cluster_date)
    
    # Calculate statistics
    statistics = calculate_statistics(lunar_clusters, market_analyses)
    
    # Display results
    display_results(lunar_clusters, market_analyses, statistics, market_symbol)
    
    # Option to visualize results
    print("\nWould you like to visualize the results? (y/n)")
    if input().lower() == 'y':
        print("\nCreating visualizations...")
        visualize_results(lunar_clusters, market_data, market_symbol, directories['visualization_dir'])

    # Option to save results (around line 1705)
    print("\nWould you like to save the results to CSV files? (y/n)")
    if input().lower() == 'y':
        print("\nSaving results...")
        save_to_csv(lunar_clusters, market_analyses, statistics, market_symbol, directories['csv_dir'])

if __name__ == "__main__":
    main()