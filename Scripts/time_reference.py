from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy import text
import json
import os


def get_business_date() -> str:
    """
    Fetch the latest business date from the database.
    Falls back to a default date if the database is unavailable.
    
    Returns:
        str: Business date in YYYY-MM-DD format
    """
    try:
        from sql_generator import get_db_engine
        
        with get_db_engine().connect() as conn:
            query = text("""
                SELECT MAX(BUSINESS_DATE)
                FROM DM_MIS_DETAILS_VW1
            """)
            result = conn.execute(query)
            date_value = result.scalar()
               
            # Convert to datetime object and format as YYYY-MM-DD
            if isinstance(date_value, str):
                # If it's a string, parse it first
                date_obj = datetime.strptime(date_value.split('.')[0], '%Y-%m-%d %H:%M:%S')
            else:
                # If it's already a datetime object
                date_obj = date_value
               
            return date_obj.strftime('%Y-%m-%d')
           
    except Exception as e:
        print(f"Error fetching business date: {e}")
        # Return a default date if database is unavailable
        return '2021-01-10'


def get_month_name(month: int) -> str:
    """Convert month number to month name abbreviation."""
    return datetime(2020, month, 1).strftime('%b')


def format_date(day: str, month: int, year: int) -> str:
    """Format date as YYYY-MM-DD or MonthName YYYY for LastDay."""
    if day == 'LastDay':
        return f"{get_month_name(month)} {year}"
    return f"{year}-{month:02d}-{int(day):02d}"


def get_quarter(month: int) -> int:
    """
    Get calendar quarter (1-4) from month number.
    Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
    """
    return (month - 1) // 3 + 1


def format_quarter_with_months(quarter: int) -> str:
    """
    Format quarter as 'Q (MON - MON)'
    Example: 1 (JAN - MAR), 2 (APR - JUN), etc.
    """
    quarter_ranges = {
        1: "JAN - MAR",
        2: "APR - JUN",
        3: "JUL - SEP",
        4: "OCT - DEC"
    }
    return f"{quarter} ({quarter_ranges[quarter]})"


def get_fiscal_year(month: int, year: int) -> int:
    """
    Get fiscal year based on calendar month and year.
    Fiscal year starts in January (same as calendar year)
    """
    return year


def get_fiscal_quarter(month: int) -> int:
    """
    Get fiscal quarter (1-4) from calendar month number.
    Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
    """
    return (month - 1) // 3 + 1


def create_time_reference_table(business_date: str = None) -> Dict[str, Any]:
    """
    Generate a time reference dictionary based on the given business date.
    
    Args:
        business_date: Business date in YYYY-MM-DD format. If None, fetches from database.
        
    Returns:
        Dictionary containing all time reference entries
    """
    # Helper functions
    def get_previous_month(m: int, y: int, months_ago: int = 1) -> tuple[int, int]:
        """Helper to get previous month and year, handling year transitions"""
        m = m - months_ago
        y = y
        while m < 1:
            m += 12
            y -= 1
        return m, y
    
    def format_month_year(m: int, y: int) -> str:
        return f"{get_month_name(m)} {y}"
    
    def get_previous_day(d: int, m: int, y: int, days_ago: int = 1) -> tuple[int, int, int]:
        """Get date that is N business days ago"""
        date = datetime(y, m, d) - timedelta(days=days_ago)
        return date.day, date.month, date.year
    
    def get_week_start(d: int, m: int, y: int) -> tuple[int, int, int]:
        """Get start of week (Monday)"""
        date = datetime(y, m, d)
        monday = date - timedelta(days=date.weekday())
        return monday.day, monday.month, monday.year
    
    def get_month_label(months_ago: int) -> str:
        """Get the correct database label for a month N months ago."""
        if months_ago == 0:
            return "MTD"
        elif months_ago == 1:
            return "Current_Month"
        elif months_ago == 2:
            return "Prior_Month"
        elif months_ago == 3:
            return "Prior_2_Month"
        elif months_ago == 4:
            return "Prior_3_Month"
        elif months_ago == 5:
            return "Prior_4_Month"
        elif months_ago == 6:
            return "Prior_5_Month"
        elif months_ago == 7:
            return "Prior_6_Month"
        elif months_ago == 8:
            return "Prior_7_Month"
        elif months_ago == 9:
            return "Prior_8_Month"
        elif months_ago == 10:
            return "Prior_9_Month"
        elif months_ago == 11:
            return "Prior_10_Month"
        elif months_ago == 12:
            return "Prior_11_Month"
        elif months_ago == 13:
            return "PY_Current_Month"
        elif months_ago == 14:
            return "PY_Prior_Month"
        elif months_ago == 15:
            return "PY_Prior_2_Month"
        elif months_ago == 16:
            return "PY_Prior_3_Month"
        elif months_ago == 17:
            return "PY_Prior_4_Month"
        elif months_ago == 18:
            return "PY_Prior_5_Month"
        return "NOT_AVAILABLE"
    
    def is_valid_month(m: int, y: int) -> bool:
        """Check if month and year are valid for data availability."""
        # Check if month is valid (1-12)
        if m < 1 or m > 12:
            return False
        
        # Check if year is within reasonable range (adjust as needed)
        if y < 2000:
            return False
            
        return True
    
    def get_month_ref(months_ago: int) -> Dict:
        """Get a month reference with proper validation."""
        # Calculate the month and year
        m, y = get_previous_month(original_month, original_year, months_ago)
        
        # Check if month is valid
        if not is_valid_month(m, y):
            return {
                "MONTH": "NOT_AVAILABLE",
                "quarter": None,
                "definition": f"Month {get_month_name(m) if 1 <= m <= 12 else 'Invalid'} {y} is not available"
            }
        
        # Get the label for this month
        label = get_month_label(months_ago)
        
        return {
            "MONTH": format_month_year(m, y),
            "quarter": get_quarter(m),
            "definition": f"Closing balance {label.replace('_', ' ').replace('MTD', 'month-to-date')}"
        }
    
    # Fetch business date from database if not provided
    if business_date is None:
        business_date = get_business_date()
    
    # Parse the business date
    original_year, original_month, original_day = map(int, business_date.split('-'))
    
    # Calculate fiscal year and quarter
    current_quarter = get_fiscal_quarter(original_month)
    
    # Calculate week start
    week_start = get_week_start(original_day, original_month, original_year)
    
    # Create the time reference dictionary
    time_ref = {
        "Business_Date": f"{original_year}-{original_month:02d}-{original_day:02d}",
        # Daily references
        "Month": format_month_year(original_month, original_year),
        "Current_Quarter": format_quarter_with_months(current_quarter),
        "Business_Day": {
            "DATE": format_date(str(original_day), original_month, original_year),
            "quarter": get_quarter(original_month),
            "definition": "Actual closing balance as of business date"
        },
        "Previous_Day": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 1)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual closing balance as of previous business day (Business Date -1)"
        },
        "Prev_Day_3": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 2)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual Closing Bal as of Business Date -2"
        },
        "Prev_Day_4": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 3)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual Closing Bal as of Business Date -3"
        },
        "Prev_Day_5": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 4)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual Closing Bal as of Business Date -4"
        },
        "Prev_Day_6": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 5)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual Closing Bal as of Business Date -5"
        },
        "Prev_Day_7": {
            "DATE": format_date(*get_previous_day(original_day, original_month, original_year, 6)),
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": "Actual Closing Bal as of Business Date -6"
        },
        
        # Month-to-date references
        "MTD": {
            "TIME_PERIOD": f"{get_month_name(original_month)} {original_year} MTD",
            "quarter": get_quarter(original_month),
            "definition": f"Month-to-date up to {get_month_name(original_month)} {original_day}, {original_year}"
        },
        "PM_SP_MTD": {
            "TIME_PERIOD": f"{get_month_name(original_month-1 or 12)} {original_year-1 if original_month == 1 else original_year} MTD",
            "quarter": get_quarter((original_month - 1) % 12 or 12),
            "definition": f"Previous month's month-to-date for same period"
        },
        "MTD_Target": {
            "name": f"{get_month_name(original_month)} {original_year} Target",
            "quarter": get_quarter(original_month),
            "definition": "Month-to-date target for current month"
        },
        
        # Monthly references (using the correct database mapping)
        "Current_Month": get_month_ref(1),
        "Prior_Month": get_month_ref(2),
        "Prior_2_Month": get_month_ref(3),
        "Prior_3_Month": get_month_ref(4),
        "Prior_4_Month": get_month_ref(5),
        "Prior_5_Month": get_month_ref(6),
        "Prior_6_Month": get_month_ref(7),
        "Prior_7_Month": get_month_ref(8),
        "Prior_8_Month": get_month_ref(9),
        "Prior_9_Month": get_month_ref(10),
        "Prior_10_Month": get_month_ref(11),
        "Prior_11_Month": get_month_ref(12),
        "PY_Current_Month": get_month_ref(13),
        "PY_Prior_Month": get_month_ref(14),
        "PY_Prior_2_Month": get_month_ref(15),
        "PY_Prior_3_Month": get_month_ref(16),
        "PY_Prior_4_Month": get_month_ref(17),
        "PY_Prior_5_Month": get_month_ref(18),
        
        # Year-to-date references
        "YTD": {
            "TIME_PERIOD": f"{original_year} YTD",
            "quarter": get_fiscal_quarter(original_month),
            "definition": f"Year-to-date up to {get_month_name(original_month)} {original_day}, {original_year}"
        },
        "YTD_Target": {
            "TIME_PERIOD": f"{original_year} YTD Target",
            "quarter": current_quarter,
            "definition": "Year-to-date target for current year"
        },
        "PY_MTD": {
            "TIME_PERIOD": f"{original_year-1} {get_month_name(original_month)} MTD",
            "quarter": current_quarter,
            "definition": f"Month-to-date for {get_month_name(original_month)} {original_year-1}"
        },
        "Prev_Year_YTD": {
            "TIME_PERIOD": f"{original_year-1} YTD",
            "quarter": current_quarter,
            "definition": f"Year-to-date for previous year up to {get_month_name(original_month)} {original_day}, {original_year-1}"
        },
        "PY_SP_MTD": {
            "TIME_PERIOD": f"{original_year-1} {get_month_name(original_month)} MTD",
            "quarter": current_quarter,
            "definition": f"Same period month-to-date from previous year"
        },
        "PY_SP_YTD": {
            "TIME_PERIOD": f"{original_year-1} YTD",
            "quarter": current_quarter,
            "definition": f"Same period year-to-date from previous year"
        },
        
        # Week-to-date reference
        "WTD": {
            "TIME_PERIOD": f"{get_month_name(week_start[1])} {week_start[0]}-{original_day}, {original_year}",
            "quarter": get_fiscal_quarter(original_month),
            "definition": f"Week-to-date from {get_month_name(week_start[1])} {week_start[0]} to {get_month_name(original_month)} {original_day}, {original_year}"
        }
    }

    # ========================================================================
    # QUARTER REFERENCES LOGIC
    # ========================================================================

    # Last quarter reference
    if original_month in [1,4,7,10]:
        time_ref["Last_Quarter_Months"] = ["Prior_2_Month", "Prior_Month", "Current_Month"]
    elif original_month in [2,5,8,11]:
        time_ref["Last_Quarter_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_Month"]
    elif original_month in [3,6,9,12]:
        time_ref["Last_Quarter_Months"] = ["Prior_4_Month", "Prior_3_Month", "Prior_2_Month"]
    
    # Current Quarter Reference
    if original_month in [1,4,7,10]:
        time_ref["Current_Quarter_Months"] = ["MTD"]
    elif original_month in [2,5,8,11]:
        time_ref["Current_Quarter_Months"] = ["Current_Month", "MTD"]
    elif original_month in [3,6,9,12]:
        time_ref["Current_Quarter_Months"] = ["Prior_Month", "Current_Month", "MTD"]

    # ========================================================================
    # SPECIFIC QUARTER REFERENCES (Q1, Q2, Q3, Q4)
    # ========================================================================
    # Q1 (Jan-Mar) Reference
    if original_month <= 3:  # January-March (Q1)
        if original_month == 1:
            time_ref["Q1_Months"] = ["MTD"]
        elif original_month == 2:
            time_ref["Q1_Months"] = ["Current_Month", "MTD"]
        elif original_month == 3:
            time_ref["Q1_Months"] = ["Prior_Month", "Current_Month", "MTD"]
        time_ref["Q1_Status"] = "in_progress"
    else:
        # Q1 is completed - determine which months to reference
        months_after_q1 = original_month - 3
        if months_after_q1 == 1:  # April
            time_ref["Q1_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_Month"]
        elif months_after_q1 == 2:  # May
            time_ref["Q1_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_1_Month"]
        elif months_after_q1 == 3:  # June
            time_ref["Q1_Months"] = ["Prior_4_Month", "Prior_3_Month", "Prior_2_Month"]
        elif months_after_q1 == 4:  # July
            time_ref["Q1_Months"] = ["Prior_5_Month", "Prior_4_Month", "Prior_3_Month"]
        elif months_after_q1 == 5:  # August
            time_ref["Q1_Months"] = ["Prior_6_Month", "Prior_5_Month", "Prior_4_Month"]
        elif months_after_q1 == 6:  # September
            time_ref["Q1_Months"] = ["Prior_7_Month", "Prior_6_Month", "Prior_5_Month"]
        elif months_after_q1 == 7:  # October
            time_ref["Q1_Months"] = ["Prior_8_Month", "Prior_7_Month", "Prior_6_Month"]
        elif months_after_q1 == 8:  # November
            time_ref["Q1_Months"] = ["Prior_9_Month", "Prior_8_Month", "Prior_7_Month"]
        elif months_after_q1 == 9:  # December
            time_ref["Q1_Months"] = ["Prior_10_Month", "Prior_9_Month", "Prior_8_Month"]
        time_ref["Q1_Status"] = "completed"
    
    # Q2 (Apr-Jun) Reference
    if original_month < 4:  # Before Q2 starts
        time_ref["Q2_Status"] = "not_available"
        time_ref["Q2_Error"] = f"Q2 is not available yet. Current month is {get_month_name(original_month)} {original_year}"
    elif original_month <= 6:  # April-June
        if original_month == 4:
            time_ref["Q2_Months"] = ["MTD"]
        elif original_month == 5:
            time_ref["Q2_Months"] = ["Current_Month", "MTD"]
        elif original_month == 6:
            time_ref["Q2_Months"] = ["Prior_Month", "Current_Month", "MTD"]
        time_ref["Q2_Status"] = "in_progress" if original_month < 6 else "completed"
    else:
        # Q2 is completed - determine which months to reference
        months_after_q2 = original_month - 6
        if months_after_q2 == 1:  # July
            time_ref["Q2_Months"] = ["Prior_2_Month", "Prior_Month", "Current_Month"]
        elif months_after_q2 == 2:  # August
            time_ref["Q2_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_Month"]
        elif months_after_q2 == 3:  # September
            time_ref["Q2_Months"] = ["Prior_4_Month", "Prior_3_Month", "Prior_2_Month"]
        elif months_after_q2 == 4:  # October
            time_ref["Q2_Months"] = ["Prior_5_Month", "Prior_4_Month", "Prior_3_Month"]
        elif months_after_q2 == 5:  # November
            time_ref["Q2_Months"] = ["Prior_6_Month", "Prior_5_Month", "Prior_4_Month"]
        elif months_after_q2 == 6:  # December
            time_ref["Q2_Months"] = ["Prior_7_Month", "Prior_6_Month", "Prior_5_Month"]
        time_ref["Q2_Status"] = "completed"
    
    # Q3 (Jul-Sep) Reference
    if original_month < 7:  # Before Q3 starts
        time_ref["Q3_Status"] = "not_available"
        time_ref["Q3_Error"] = f"Q3 is not available yet. Current month is {get_month_name(original_month)} {original_year}"
    elif original_month <= 9:  # July-September
        if original_month == 7:
            time_ref["Q3_Months"] = ["MTD"]
        elif original_month == 8:
            time_ref["Q3_Months"] = ["Current_Month", "MTD"]
        elif original_month == 9:
            time_ref["Q3_Months"] = ["Prior_Month", "Current_Month", "MTD"]
        time_ref["Q3_Status"] = "in_progress" if original_month < 9 else "completed"
    else:
        # Q3 is completed - determine which months to reference
        months_after_q3 = original_month - 9
        if months_after_q3 == 1:  # October
            time_ref["Q3_Months"] = ["Prior_2_Month", "Prior_Month", "Current_Month"]
        elif months_after_q3 == 2:  # November
            time_ref["Q3_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_Month"]
        elif months_after_q3 == 3:  # December
            time_ref["Q3_Months"] = ["Prior_4_Month", "Prior_3_Month", "Prior_2_Month"]
        time_ref["Q3_Status"] = "completed"
    
    # Q4 (Oct-Dec) Reference
    if original_month < 10:  # Before Q4 starts
        time_ref["Q4_Status"] = "not_available"
        time_ref["Q4_Error"] = f"Q4 is not available yet. Current month is {get_month_name(original_month)} {original_year}"
    elif original_month <= 12:  # October-December
        if original_month == 10:
            time_ref["Q4_Months"] = ["MTD"]
        elif original_month == 11:
            time_ref["Q4_Months"] = ["Current_Month", "MTD"]
        elif original_month == 12:
            time_ref["Q4_Months"] = ["Prior_Month", "Current_Month", "MTD"]
        time_ref["Q4_Status"] = "in_progress" if original_month < 12 else "completed"

     # ========================================================================
    # PREVIOUS YEAR QUARTER REFERENCES (PY_Q1, PY_Q2, PY_Q3, PY_Q4)
    # ========================================================================
    
    # PY Q1 (Previous Year Jan-Mar)
    months_after_py_q1 = original_month - 1
    if months_after_py_q1 == 0:  # January
        time_ref["PY_Q1_Months"] = ["Prior_11_Month", "Prior_10_Month", "Prior_9_Month"]
    elif months_after_py_q1 == 1:  # February
        time_ref["PY_Q1_Months"] = ["PY_Current_Month", "Prior_11_Month", "Prior_10_Month"]
    elif months_after_py_q1 == 2:  # March
        time_ref["PY_Q1_Months"] = ["PY_Prior_Month", "PY_Current_Month", "Prior_11_Month"]
    elif months_after_py_q1 == 3:  # April
        time_ref["PY_Q1_Months"] = ["PY_Prior_2_Month", "PY_Prior_Month", "PY_Current_Month"]
    elif months_after_py_q1 == 4:  # May
        time_ref["PY_Q1_Months"] = ["PY_Prior_3_Month", "PY_Prior_2_Month", "PY_Prior_Month"]
    elif months_after_py_q1 == 5:  # June
        time_ref["PY_Q1_Months"] = ["PY_Prior_4_Month", "PY_Prior_3_Month", "PY_Prior_2_Month"]
    elif months_after_py_q1 == 6:  # July
        time_ref["PY_Q1_Months"] = ["PY_Prior_5_Month", "PY_Prior_4_Month", "PY_Prior_3_Month"]
    elif months_after_py_q1 == 7:  # August
        time_ref["PY_Q1_Months"] = ["PY_Prior_5_Month", "PY_Prior_4_Month"]
    elif months_after_py_q1 == 8:  # September
        time_ref["PY_Q1_Months"] = ["PY_Prior_5_Month"]
    time_ref["PY_Q1_Status"] = "completed"
    if months_after_py_q1 == 9:  # October
        time_ref["PY_Q1_Status"] = "Not_Available"
    elif months_after_py_q1 == 10:  # November
        time_ref["PY_Q1_Status"] = "Not_Available"
    elif months_after_py_q1 == 11:  # December
        time_ref["PY_Q1_Status"] = "Not_Available"
    time_ref["PY_Q1_Status"] = "Not_Available"

    # PY Q2 (Previous Year Apr-Jun)
    months_after_py_q2 = original_month - 0
    if months_after_py_q2 == 1:  # May
        time_ref["PY_Q2_Months"] = ["Prior_8_Month", "Prior_7_Month", "Prior_6_Month"]
    elif months_after_py_q2 == 2:  # June
        time_ref["PY_Q2_Months"] = ["Prior_9_Month", "Prior_8_Month", "Prior_7_Month"]
    elif months_after_py_q2 == 3:  # July
        time_ref["PY_Q2_Months"] = ["Prior_10_Month", "Prior_9_Month", "Prior_8_Month"]
    elif months_after_py_q2 == 4:  # August
        time_ref["PY_Q2_Months"] = ["Prior_11_Month", "Prior_10_Month", "Prior_9_Month"]
    elif months_after_py_q2 == 5:  # September
        time_ref["PY_Q2_Months"] = ["PY_Current_Month", "Prior_11_Month", "Prior_10_Month"]
    elif months_after_py_q2 == 6:  # October
        time_ref["PY_Q2_Months"] = ["PY_Prior_Month", "PY_Current_Month", "Prior_11_Month"]
    elif months_after_py_q2 == 7:  # November
        time_ref["PY_Q2_Months"] = ["PY_Prior_2_Month", "PY_Prior_Month", "PY_Current_Month"]
    elif months_after_py_q2 == 8:  # December
        time_ref["PY_Q2_Months"] = ["PY_Prior_3_Month", "PY_Prior_2_Month", "PY_Prior_Month"]
    elif months_after_py_q2 == 9:  # January
        time_ref["PY_Q2_Months"] = ["PY_Prior_4_Month", "PY_Prior_3_Month", "PY_Prior_2_Month"]
    elif months_after_py_q2 == 10:  # February
        time_ref["PY_Q2_Months"] = ["PY_Prior_5_Month", "PY_Prior_4_Month", "PY_Prior_3_Month"]
    elif months_after_py_q2 == 11:  # March
        time_ref["PY_Q2_Months"] = ["PY_Prior_5_Month", "PY_Prior_4_Month"]
    elif months_after_py_q2 == 11:  # March
        time_ref["PY_Q2_Months"] = ["PY_Prior_5_Month"]
    time_ref["PY_Q2_Status"] = "completed"

    # PY Q3 (Previous Year Jul-Sep)
    months_after_py_q3 = original_month - 0
    if months_after_py_q3 == 1:  # August
        time_ref["PY_Q3_Months"] = ["Prior_5_Month", "Prior_4_Month", "Prior_3_Month"]
    elif months_after_py_q3 == 2:  # September
        time_ref["PY_Q3_Months"] = ["Prior_6_Month", "Prior_5_Month", "Prior_4_Month"]
    elif months_after_py_q3 == 3:  # October
        time_ref["PY_Q3_Months"] = ["Prior_7_Month", "Prior_6_Month", "Prior_5_Month"]
    elif months_after_py_q3 == 4:  # November
        time_ref["PY_Q3_Months"] = ["Prior_8_Month", "Prior_7_Month", "Prior_6_Month"]
    elif months_after_py_q3 == 5:  # December
        time_ref["PY_Q3_Months"] = ["Prior_9_Month", "Prior_8_Month", "Prior_7_Month"]
    elif months_after_py_q3 == 6:  # January
        time_ref["PY_Q3_Months"] = ["Prior_10_Month", "Prior_9_Month", "Prior_8_Month"]
    elif months_after_py_q3 == 7:  # February
        time_ref["PY_Q3_Months"] = ["Prior_11_Month", "Prior_10_Month", "Prior_9_Month"]
    elif months_after_py_q3 == 8:  # March
        time_ref["PY_Q3_Months"] = ["PY_Current_Month", "Prior_11_Month", "Prior_10_Month"]
    elif months_after_py_q3 == 9:  # April
        time_ref["PY_Q3_Months"] = ["PY_Prior_Month", "PY_Current_Month", "Prior_11_Month"]
    elif months_after_py_q3 == 10:  # May
        time_ref["PY_Q3_Months"] = ["PY_Prior_2_Month", "PY_Prior_Month", "PY_Current_Month"]
    elif months_after_py_q3 == 11:  # June
        time_ref["PY_Q3_Months"] = ["PY_Prior_3_Month", "PY_Prior_2_Month", "PY_Prior_Month"]
    elif months_after_py_q3 == 12:  # September
        time_ref["PY_Q3_Months"] = ["PY_Prior_4_Month", "PY_Prior_3_Month", "PY_Prior_2_Month"]
    time_ref["PY_Q3_Status"] = "completed"

    # PY Q4 (Previous Year Oct-Dec)
    months_after_py_q4 = original_month - 0
    if months_after_py_q4 == 1:  # November
        time_ref["PY_Q4_Months"] = ["Prior_2_Month", "Prior_Month", "Current_Month"]
    elif months_after_py_q4 == 2:  # December
        time_ref["PY_Q4_Months"] = ["Prior_3_Month", "Prior_2_Month", "Prior_Month"]
    elif months_after_py_q4 == 3:  # January
        time_ref["PY_Q4_Months"] = ["Prior_4_Month", "Prior_3_Month", "Prior_2_Month"]
    elif months_after_py_q4 == 4:  # February
        time_ref["PY_Q4_Months"] = ["Prior_5_Month", "Prior_4_Month", "Prior_3_Month"]
    elif months_after_py_q4 == 5:  # March
        time_ref["PY_Q4_Months"] = ["Prior_6_Month", "Prior_5_Month", "Prior_4_Month"]
    elif months_after_py_q4 == 6:  # April
        time_ref["PY_Q4_Months"] = ["Prior_7_Month", "Prior_6_Month", "Prior_5_Month"]
    elif months_after_py_q4 == 7:  # May
        time_ref["PY_Q4_Months"] = ["Prior_8_Month", "Prior_7_Month", "Prior_6_Month"]
    elif months_after_py_q4 == 8:  # June
        time_ref["PY_Q4_Months"] = ["Prior_9_Month", "Prior_8_Month", "Prior_7_Month"]
    elif months_after_py_q4 == 9:  # July
        time_ref["PY_Q4_Months"] = ["Prior_10_Month", "Prior_9_Month", "Prior_8_Month"]
    elif months_after_py_q4 == 10:  # August
        time_ref["PY_Q4_Months"] = ["Prior_11_Month", "Prior_10_Month", "Prior_9_Month"]
    elif months_after_py_q4 == 11:  # September
        time_ref["PY_Q4_Months"] = ["PY_Current_Month", "Prior_11_Month", "Prior_10_Month"]
    elif months_after_py_q4 == 12:  # September
        time_ref["PY_Q4_Months"] = ["PY_Prior_Month", "PY_Current_Month", "Prior_11_Month"]
    time_ref["PY_Q4_Status"] = "completed"

    # --- Normalize PY quarter month labels to always have 3 entries ---
    def _ensure_three_months(labels: list[str]) -> list[str]:
        if not labels:
            return []
        if len(labels) >= 3:
            return labels[:3]
        return labels + [labels[-1]] * (3 - len(labels))

    # If any PY list is empty (due to 18-month window), backfill with boundary label
    for py_code in ("PY_Q1", "PY_Q2", "PY_Q3", "PY_Q4"):
        key = f"{py_code}_Months"
        months_list = time_ref.get(key, [])
        if not months_list:
            # choose safest fallback
            fallback = ["PY_Prior_5_Month", "PY_Prior_5_Month", "PY_Prior_5_Month"]
            time_ref[key] = fallback
        else:
            time_ref[key] = _ensure_three_months(months_list)
        time_ref[f"{py_code}_Status"] = "completed"

    return time_ref



# ============================================================================
# Quarter Rules Functions
# ============================================================================

def get_quarter_keywords() -> list:
    """Return list of keywords that trigger quarter rules inclusion."""
    return [
        "quater",
        "quarter",
        "quarterly",
        "q1", "q2", "q3", "q4",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "quarter 1", "quarter 2", "quarter 3", "quarter 4",
        "quarter-end",
        "quarterend",
        "last quarter",
        "current quarter",
        "this quarter",
        "previous quarter",
        "prior quarter",
        "previous year q1", "previous year q2", "previous year q3", "previous year q4",
        "last year q1", "last year q2", "last year q3", "last year q4",
        "py q1", "py q2", "py q3", "py q4"
    ]


def should_include_quarter_rules(query: str) -> bool:
    """
    Check if the query contains quarter-related keywords.
    
    Args:
        query: User's natural language query
        
    Returns:
        True if quarter rules should be included, False otherwise
    """
    query_lower = query.lower()
    # Base quarter keywords (safe list) – these explicitly mention quarters
    base_keywords = [k.lower() for k in get_quarter_keywords()]
    # Configured aliases – but filter to only tokens that clearly indicate a quarter
    aliases_map = get_configured_quarter_aliases()

    def is_quarter_token(tok: str) -> bool:
        t = tok.lower()
        if not t:
            return False
        # tokens that include the word 'quarter' are valid
        if 'quarter' in t:
            return True
        # q1/q 1/q-1 style
        import re
        if re.fullmatch(r"q\s*-?\s*[1-4]", t):
            return True
        # Avoid pure ordinals (e.g., '1st') here to reduce false positives
        return False

    alias_quarter_tokens = [tok for lst in aliases_map.values() for tok in lst if is_quarter_token(tok)]

    # Do NOT let PY hints alone trigger quarter rules
    triggers = set(base_keywords + alias_quarter_tokens + ["quarterly"])  # allow 'quarterly'
    return any(k in query_lower for k in triggers)


"""
Quarter alias loading: defaults are defined here but can be extended/overridden
by values in the repo-level keywords_and_rules.json under quarter_detection.
"""

# Defaults used when config is missing or partially defined
DEFAULT_QUARTER_ALIASES: Dict[str, list[str]] = {
    "CURRENT": ["current quarter", "this quarter", "present quarter"],
    "LAST": ["last quarter", "previous quarter", "prior quarter", "quarter-end", "quarterend", "prev quarter"],
    "Q1": ["q1", "q 1", "q-1", "1st", "first", "quarter 1", "1st quarter", "first quarter", "quarter one"],
    "Q2": ["q2", "q 2", "q-2", "2nd", "second", "quarter 2", "2nd quarter", "second quarter", "quarter two"],
    "Q3": ["q3", "q 3", "q-3", "3rd", "third", "quarter 3", "3rd quarter", "third quarter", "quarter three"],
    "Q4": ["q4", "q 4", "q-4", "4th", "fourth", "quarter 4", "4th quarter", "fourth quarter", "quarter four"],
}

DEFAULT_PY_HINTS = [
    "previous year", "last year", "prior year",
    "py ", "py-", "py_", "py",
    "ly", "last yr", "prior yr"
]

_QUARTER_CONFIG_CACHE: Dict[str, Any] | None = None


def _get_keywords_json_path() -> str:
    """Resolve absolute path to keywords_and_rules.json in project root."""
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    return os.path.join(root, "keywords_and_rules.json")


def _load_quarter_config() -> Dict[str, Any]:
    """Load quarter detection config from JSON with caching and safe fallback."""
    global _QUARTER_CONFIG_CACHE
    if _QUARTER_CONFIG_CACHE is not None:
        return _QUARTER_CONFIG_CACHE
    path = _get_keywords_json_path()
    data: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    _QUARTER_CONFIG_CACHE = data
    return data


def _merge_aliases(defaults: Dict[str, list[str]], overrides: Dict[str, list[str]] | None) -> Dict[str, list[str]]:
    """Merge default alias mapping with overrides (union, lowercase, dedup)."""
    result: Dict[str, list[str]] = {}
    overrides = overrides or {}
    for key, def_list in defaults.items():
        ovr_list = overrides.get(key, []) or []
        # normalize and merge
        merged = []
        seen = set()
        for token in [*(t.lower() for t in def_list), *(t.lower() for t in ovr_list)]:
            if token and token not in seen:
                seen.add(token)
                merged.append(token)
        result[key] = merged
    # Include any extra keys present only in overrides
    for key, ovr_list in overrides.items():
        if key not in result:
            norm = []
            seen = set()
            for t in (ovr_list or []):
                t = t.lower()
                if t and t not in seen:
                    seen.add(t)
                    norm.append(t)
            result[key] = norm
    return result


def get_configured_quarter_aliases() -> Dict[str, list[str]]:
    cfg = _load_quarter_config()
    qdet = cfg.get("quarter_detection", {}) if isinstance(cfg, dict) else {}
    overrides = qdet.get("quarter_aliases") if isinstance(qdet, dict) else None
    return _merge_aliases(DEFAULT_QUARTER_ALIASES, overrides if isinstance(overrides, dict) else None)


def get_configured_py_hints() -> list[str]:
    cfg = _load_quarter_config()
    qdet = cfg.get("quarter_detection", {}) if isinstance(cfg, dict) else {}
    hints = qdet.get("py_hints") if isinstance(qdet, dict) else None
    base = [t.lower() for t in DEFAULT_PY_HINTS]
    extra = [t.lower() for t in (hints or [])]
    # dedup preserve order (defaults first)
    out = []
    seen = set()
    for t in base + extra:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _detect_quarter_from_query(query: str, time_reference: Dict) -> tuple[str | None, int | None, str | None]:
    """
    Detect specific quarter intent from the query.

    Returns a tuple of (quarter_code, display_year, quarter_name)
    - quarter_code in {"Q1","Q2","Q3","Q4","PY_Q1","PY_Q2","PY_Q3","PY_Q4","CURRENT","LAST"}
    - display_year is the natural language year to show (current year or previous year)
    - quarter_name is a friendly name like "First quarter"
    """
    import re

    # Parse current year from time_reference
    try:
        y_str = str(time_reference.get('Business_Date', '0000-01-01')).split('-')[0]
        current_year = int(y_str)
    except Exception:
        current_year = None

    quarter_names = {
        'Q1': 'First quarter',
        'Q2': 'Second quarter',
        'Q3': 'Third quarter',
        'Q4': 'Fourth quarter',
        'CURRENT': 'Current quarter',
        'LAST': 'Last quarter'
    }

    

    q = query.lower()

    # Explicit year in query (support current or previous year only)
    year_match = re.search(r"\b(20\d{2})\b", q)
    explicit_year = int(year_match.group(1)) if year_match else None

    # Load configured aliases/hints
    QUARTER_ALIASES = get_configured_quarter_aliases()
    PY_HINTS = get_configured_py_hints()

    # Special cases: current/last quarter
    if any(alias in q for alias in QUARTER_ALIASES.get("CURRENT", [])):
        return ("CURRENT", current_year, quarter_names['CURRENT'])
    if any(alias in q for alias in QUARTER_ALIASES.get("LAST", [])):
        return ("LAST", current_year, quarter_names['LAST'])

    # Determine basic quarter using aliases
    base = None
    for code in ("Q1", "Q2", "Q3", "Q4"):
        if any(alias in q for alias in QUARTER_ALIASES.get(code, [])):
            base = code
            break

    if not base:
        return (None, None, None)

    # Previous year intent
    if any(h in q for h in PY_HINTS):
        return (f"PY_{base}", current_year - 1 if current_year else None, quarter_names[base])

    # Explicit year mapping (limit to current or previous year)
    if explicit_year and current_year:
        if explicit_year == current_year:
            return (base, explicit_year, quarter_names[base])
        if explicit_year == current_year - 1:
            return (f"PY_{base}", explicit_year, quarter_names[base])
        # Unsupported year for available labels; fall back to base
        return (base, current_year, quarter_names[base])

    # Default: current year
    return (base, current_year, quarter_names[base])


def _detect_bal_type_and_metric(query: str) -> tuple[int, str]:
    """Heuristic to choose BAL_TYPE and a readable metric phrase from the query."""
    q = query.lower()
    # Income/P&L style
    if any(k in q for k in ["income", "revenue", "expense", "profit", "fee", "commission", "interest income", "net income", "p&l", "pnL", "earning"]):
        return 3, "interest income"
    # Balance-sheet style
    if any(k in q for k in ["deposit", "loan", "advance", "balance", "asset", "liabilit", "outstanding", "eop", "closing"]):
        return 1, "total deposits"
    # Default to balance-sheet deposits
    return 1, "total deposits"


def _detect_quarters_from_query(query: str, time_reference: Dict) -> list[tuple[str, int | None, str]]:
    """
    Detect multiple quarter intents in order, supporting mixed-year requests like:
      - "q3 and last year q3"
      - "last year q3 and q4" (propagates last-year to the following unqualified quarter)

    Returns [(quarter_code, display_year, quarter_name), ...]
    quarter_code ∈ {Q1..Q4, PY_Q1..PY_Q4, CURRENT, LAST}
    """
    import re

    q = query.lower()
    QUARTER_ALIASES = get_configured_quarter_aliases()
    PY_HINTS = get_configured_py_hints()

    # Determine current year
    try:
        y_str = str(time_reference.get('Business_Date', '0000-01-01')).split('-')[0]
        current_year = int(y_str)
    except Exception:
        current_year = None

    quarter_names = {
        'Q1': 'First quarter',
        'Q2': 'Second quarter',
        'Q3': 'Third quarter',
        'Q4': 'Fourth quarter',
        'CURRENT': 'Current quarter',
        'LAST': 'Last quarter'
    }

    # Split into segments by simple conjunctions to localize qualifiers
    segments = [s.strip() for s in re.split(r"\s+(?:and|&)\s+|,", q) if s.strip()]

    results: list[tuple[str, int | None, str]] = []
    last_year_mode: dict | None = None  # {'type': 'py'|'year', 'year': int|None}

    # Tail qualifier applies to all earlier mentions, e.g., "... for last year"
    tail_mode: dict | None = None
    # Try explicit year at tail
    m_tail_year = re.search(r"(?:for\s+)?(20\d{2})\s*$", q)
    if m_tail_year:
        ty = int(m_tail_year.group(1))
        tail_mode = {'type': 'year', 'year': ty}
    else:
        # Try previous-year phrases at tail
        if re.search(r"(?:for\s+)?(last year|previous year|prior year)\s*$", q):
            tail_mode = {'type': 'py', 'year': (current_year - 1) if current_year else None}

    # Leading qualifier applies forward to following segments
    m_head_year = re.match(r"^\s*(?:for\s+)?(20\d{2})\b", q)
    if m_head_year:
        last_year_mode = {'type': 'year', 'year': int(m_head_year.group(1))}
    elif re.match(r"^\s*(?:for\s+)?(last year|previous year|prior year)\b", q):
        last_year_mode = {'type': 'py', 'year': (current_year - 1) if current_year else None}

    def has_py(seg: str) -> bool:
        return any(h in seg for h in PY_HINTS)

    def get_explicit_year(seg: str) -> int | None:
        m = re.search(r"\b(20\d{2})\b", seg)
        return int(m.group(1)) if m else None

    for seg in segments:
        seg_year = get_explicit_year(seg)
        seg_py = has_py(seg)

        # Decide segment year mode
        seg_mode: dict | None = None
        if seg_year is not None:
            seg_mode = {'type': 'year', 'year': seg_year}
            last_year_mode = seg_mode  # propagate forward
        elif seg_py:
            seg_mode = {'type': 'py', 'year': (current_year - 1) if current_year else None}
            last_year_mode = seg_mode  # propagate forward
        elif last_year_mode is not None:
            # inherit previous qualifier if segment has none
            seg_mode = last_year_mode
        elif tail_mode is not None:
            # apply trailing qualifier to unqualified segments
            seg_mode = tail_mode

        # Find all quarter aliases inside this segment, preserve order
        hits: list[tuple[int, str]] = []
        for code, aliases in QUARTER_ALIASES.items():
            for alias in aliases:
                idx = seg.find(alias)
                if idx != -1:
                    hits.append((idx, code))
        hits.sort(key=lambda x: x[0])

        for _, base in hits:
            if base in ("CURRENT", "LAST"):
                results.append((base, current_year, quarter_names.get(base, base)))
                continue
            # Apply segment mode
            if seg_mode and seg_mode['type'] == 'year' and current_year:
                if seg_mode['year'] == current_year - 1:
                    code = f"PY_{base}"
                    disp_year = seg_mode['year']
                else:
                    code = base
                    disp_year = seg_mode['year']
            elif seg_mode and seg_mode['type'] == 'py':
                code = f"PY_{base}"
                disp_year = seg_mode['year']
            else:
                code = base
                disp_year = current_year

            results.append((code, disp_year, quarter_names.get(base, base)))

    # Deduplicate identical (code, year) while preserving order
    seen = set()
    deduped: list[tuple[str, int | None, str]] = []
    for code, dy, qn in results:
        key = (code, dy)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((code, dy, qn))

    return deduped


def build_multi_quarter_rule(query: str, time_reference: Dict) -> str:
    """
    Build a targeted rule snippet for multiple quarters in one request.
    Generates a single SQL with separate columns per requested quarter.
    """
    intents = _detect_quarters_from_query(query, time_reference)
    # Need at least 2 to qualify as multi
    if len(intents) < 2:
        return ""

    bal_type, metric_phrase = _detect_bal_type_and_metric(query)

    use_lines = []
    select_columns = []
    unavailable_notes = []

    for code, disp_year, qname in intents[:6]:  # safety cap
        # Resolve months/status
        if code in ("CURRENT", "LAST"):
            key = "Current_Quarter_Months" if code == "CURRENT" else "Last_Quarter_Months"
            months = time_reference.get(key, [])
            status = None
            error = None
        elif code.startswith("PY_"):
            months = time_reference.get(f"{code}_Months", [])
            status = time_reference.get(f"{code}_Status")
            error = None
        else:
            months = time_reference.get(f"{code}_Months", [])
            status = time_reference.get(f"{code}_Status")
            error = time_reference.get(f"{code}_Error")

        human = code.replace("PY_", "Previous Year ")
        if not months or (status and str(status).lower() == "not_available"):
            unavailable_notes.append(f"{human} unavailable{': ' + error if error else ''}")
            continue

        use_lines.append(f"- {human}: {', '.join(months)}")

        alias = human.lower().replace(' ', '_').replace('-', '_')
        if bal_type == 1:
            expr = "BUSINESS_DAY" if any(m.upper() == "MTD" for m in months) else months[-1]
            select_columns.append(f"    ROUND(SUM({expr}), 2) AS {alias}_value")
        else:
            expr = " + ".join(months)
            select_columns.append(f"    ROUND(SUM({expr}), 2) AS {alias}_value")

    if not select_columns:
        return ""

    sql = (
        "```sql\nSELECT\n" + ",\n".join(select_columns) +
        "\nFROM DM_MIS_DETAILS_VW1\nWHERE " + ("MRL_LINE IN (#MRL_LINES#)\n" if True else "") +
        f"AND BAL_TYPE = {bal_type}\n" +
        "AND CCY_TYPE = 'BCY';\n```"
    )

    header = (
        "\n==== QUARTER RULE (targeted - multiple) ====\n"
        "If BAL_TYPE = 1 → use only last month of each quarter (or BUSINESS_DAY when in-progress).\n"
        "If BAL_TYPE = 3 → sum all months for each requested quarter.\n\n"
        f"Requested: {metric_phrase} for " + ", ".join([f"{dy or ''} {qn}" for _, dy, qn in intents]) + "\n"
        "Use:\n" + "\n".join(use_lines) + ("\n\nNote: " + "; ".join(unavailable_notes) if unavailable_notes else "") + "\n\n"
    )

    return header + sql + "\n"
def build_targeted_quarter_rule(query: str, time_reference: Dict) -> str:
    """
    Build a minimal, targeted quarter rule snippet for the specific quarter mentioned in the query.
    Falls back to empty string if no specific quarter can be determined or if unavailable.
    """
    quarter_code, display_year, quarter_name = _detect_quarter_from_query(query, time_reference)
    if not quarter_code:
        return ""

    # Resolve months list and status
    months: list[str] = []
    status = None
    error = None
    if quarter_code in ("CURRENT", "LAST"):
        key = "Current_Quarter_Months" if quarter_code == "CURRENT" else "Last_Quarter_Months"
        months = time_reference.get(key, [])
    elif quarter_code.startswith("PY_"):
        py_key = f"{quarter_code}_Months"
        months = time_reference.get(py_key, [])
        status = time_reference.get(f"{quarter_code}_Status")
    else:
        # Q1..Q4
        months = time_reference.get(f"{quarter_code}_Months", [])
        status = time_reference.get(f"{quarter_code}_Status")
        error = time_reference.get(f"{quarter_code}_Error")

    # If not available or empty, return a helpful note
    if (status and str(status).lower() == "not_available") or not months:
        note = error or f"{quarter_code} months not available"
        return f"\n==== QUARTER RULE (targeted) ====\nRequested: {display_year or ''} {quarter_name or quarter_code}\nStatus: NOT AVAILABLE\nNote: {note}\n"

    # Choose BAL_TYPE and metric phrase
    bal_type, metric_phrase = _detect_bal_type_and_metric(query)

    # Build example SQL
    if bal_type == 1:
        # Balance sheet: use BUSINESS_DAY for in-progress; last month for completed
        last_label = months[-1] if months else "CURRENT_MONTH"
        # If any MTD present, treat as in-progress and use BUSINESS_DAY
        if any(m.upper() == "MTD" for m in months):
            agg_expr = "BUSINESS_DAY"
        else:
            agg_expr = last_label
        example_sql = (
            "Example: Give me "
            f"{display_year or ''} {quarter_name or quarter_code} {metric_phrase}\n"
            "```sql\nSELECT\n"
            f"    ROUND(SUM({agg_expr}), 2) AS requested_quarter_value\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 1\n"
            "AND CCY_TYPE = 'BCY';\n```")
    else:
        # P&L: sum all months
        sum_expr = " + ".join(months)
        example_sql = (
            "Example: Give me "
            f"{display_year or ''} {quarter_name or quarter_code} {metric_phrase}\n"
            "```sql\nSELECT\n"
            f"    ROUND(SUM({sum_expr}), 2) AS requested_quarter_value\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 3\n"
            "AND CCY_TYPE = 'BCY';\n```")

    # Compose snippet
    human_code = quarter_code.replace("PY_", "Previous Year ")
    return (
        "\n==== QUARTER RULE (targeted) ====\n"
        "If BAL_TYPE = 1 → include only last month of the quarter for balance sheet metrics.\n"
        "If BAL_TYPE = 3 → sum all months of the quarter for income statement metrics.\n\n"
        f"Requested: {display_year or ''} {quarter_name or human_code}\n"
        f"Use - {human_code}: {', '.join(months)}\n\n"
        f"{example_sql}\n"
    )


def build_quarter_rules_section(time_reference: Dict) -> str:
    """
    Build the quarter rules section with time reference placeholders replaced.
    
    Args:
        time_reference: Dictionary from create_time_reference_table()
        
    Returns:
        Complete quarter rules section as a formatted string
    """
    last_quarter_months = time_reference.get("Last_Quarter_Months", [])
    current_quarter_months = time_reference.get("Current_Quarter_Months", [])
    
    # Get quarter status information
    q1_status = time_reference.get("Q1_Status", "not_available")
    q2_status = time_reference.get("Q2_Status", "not_available")
    q3_status = time_reference.get("Q3_Status", "not_available")
    q4_status = time_reference.get("Q4_Status", "not_available")
    
    q1_months = time_reference.get("Q1_Months", [])
    q2_months = time_reference.get("Q2_Months", [])
    q3_months = time_reference.get("Q3_Months", [])
    q4_months = time_reference.get("Q4_Months", [])
    
    py_q1_months = time_reference.get("PY_Q1_Months", [])
    py_q2_months = time_reference.get("PY_Q2_Months", [])
    py_q3_months = time_reference.get("PY_Q3_Months", [])
    py_q4_months = time_reference.get("PY_Q4_Months", [])

    # Safely build previous-year quarter example blocks to avoid IndexError
    if py_q1_months:
        py_q1_deposits_last = py_q1_months[-1]
        py_q1_sum_expr = " + ".join(py_q1_months)
        py_q1_block = (
            "Example: Give me previous year Q1 deposits (BAL_TYPE = 1)\n"
            "```sql\n"
            "SELECT\n"
            f"    ROUND(SUM({py_q1_deposits_last}), 2) AS py_q1_deposits\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 1\n"
            "AND CCY_TYPE = 'BCY';\n"
            "```\n\n"
            "Example: Give me previous year Q1 interest income (BAL_TYPE = 3)\n"
            "```sql\n"
            "SELECT\n"
            f"    ROUND(SUM({py_q1_sum_expr}), 2) AS py_q1_interest_income\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 3\n"
            "AND CCY_TYPE = 'BCY';\n"
            "```"
        )
    else:
        py_q1_block = ""

    if py_q2_months:
        py_q2_sum_expr = " + ".join(py_q2_months)
        py_q2_block = (
            "Example: Give me last year Q2 revenue (BAL_TYPE = 3)\n"
            "```sql\n"
            "SELECT\n"
            f"    ROUND(SUM({py_q2_sum_expr}), 2) AS py_q2_revenue\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 3\n"
            "AND CCY_TYPE = 'BCY';\n"
            "```"
        )
    else:
        py_q2_block = ""

    if py_q3_months:
        py_q3_last = py_q3_months[-1]
        py_q3_block = (
            "Example: Give me last year Q3 deposits (BAL_TYPE = 1)\n"
            "```sql\n"
            "SELECT\n"
            f"    ROUND(SUM({py_q3_last}), 2) AS py_q3_deposits\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE MRL_LINE IN (#MRL_LINES#)\n"
            "AND BAL_TYPE = 1\n"
            "AND CCY_TYPE = 'BCY';\n"
            "```"
        )
    else:
        py_q3_block = ""

    if py_q4_months:
        py_q4_sum_expr = " + ".join(py_q4_months)
        py_q4_block = (
            "Example: Give me last year Q4 net income (BAL_TYPE = 3)\n"
            "```sql\n"
            "SELECT\n"
            f"    ROUND(SUM({py_q4_sum_expr}), 2) AS py_q4_net_income\n"
            "FROM DM_MIS_DETAILS_VW1\n"
            "WHERE BAL_TYPE = 3\n"
            "AND CCY_TYPE = 'BCY';\n"
            "```"
        )
    else:
        py_q4_block = ""
    
    # Build the section with actual values substituted
    section = f"""
==== QUARTER RULES ====
Rules for handling quarter-based queries (Q1, Q2, Q3, Q4, current quarter, last quarter, previous year quarters)
AVAILABLE QUARTERS:
- Last Quarter: {', '.join(last_quarter_months) if last_quarter_months else "NOT_AVAILABLE"}
- Current Quarter: {', '.join(current_quarter_months) if current_quarter_months else "NOT_AVAILABLE"}
-===========================================================================================================-
- PY Q1: {', '.join(py_q1_months) if py_q1_months else "NOT_AVAILABLE"}
- PY Q2: {', '.join(py_q2_months) if py_q2_months else "NOT_AVAILABLE"}
- PY Q3: {', '.join(py_q3_months) if py_q3_months else "NOT_AVAILABLE"}
- PY Q4: {', '.join(py_q4_months) if py_q4_months else "NOT_AVAILABLE"}
- Q1: {', '.join(q1_months) if q1_status != "not_available" and q1_months else "NOT AVAILABLE"}
- Q2: {', '.join(q2_months) if q2_status != "not_available" and q2_months else "NOT AVAILABLE"}
- Q3: {', '.join(q3_months) if q3_status != "not_available" and q3_months else "NOT AVAILABLE"}
- Q4: {', '.join(q4_months) if q4_status != "not_available" and q4_months else "NOT AVAILABLE"}

IMPORTANT NOTES:
1. "quarter-end" means "last quarter"
2. Before using any quarter, check its availability
3. Previous year quarters (PY Q1-Q4) are always available
4. Current year quarters depend on the current month

QUARTER HANDLING RULES:

=== CURRENT QUARTER / THIS QUARTER ===
BAL_TYPE = 1 (Balance Sheet): Use BUSINESS_DAY
BAL_TYPE = 3 (Income Statement): Sum all available months in Current_Quarter_Months

Example: Give me current quarter total deposits
```sql
SELECT
    ROUND(SUM(BUSINESS_DAY), 2) AS current_quarter_deposits
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me current quarter interest income
```sql
SELECT
    ROUND(SUM({"MTD" if len(current_quarter_months) == 1 else "CURRENT_MONTH + MTD" if len(current_quarter_months) == 2 else "PRIOR_MONTH + CURRENT_MONTH + MTD"}), 2) AS current_quarter_interest_income
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```

=== LAST QUARTER / PREVIOUS QUARTER ===
BAL_TYPE = 1 (Balance Sheet): Use last month of previous quarter
BAL_TYPE = 3 (Income Statement): Sum all 3 months of last quarter

Example: Give me last quarter total deposits
```sql
SELECT
    ROUND(SUM({last_quarter_months[2] if len(last_quarter_months) >= 3 else 'CURRENT_MONTH'}), 2) AS last_quarter_deposits
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me last quarter interest income
```sql
SELECT
    ROUND(SUM({last_quarter_months[0]} + {last_quarter_months[1]} + {last_quarter_months[2]}), 2) AS last_quarter_interest_income
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```

=== SPECIFIC QUARTERS (Q1, Q2, Q3, Q4) ===

Q1 QUARTER (Jan-Mar):
Status: {q1_status.upper()}
{f"Months: {', '.join(q1_months)}" if q1_status != "not_available" else f"Error: {time_reference.get('Q1_Error', 'Not available')}"}

{"" if q1_status == "not_available" else f'''Example: Give me Q1 deposits (BAL_TYPE = 1)
```sql
SELECT
    ROUND(SUM({"BUSINESS_DAY" if q1_status == "in_progress" else q1_months[-1]}), 2) AS q1_deposits
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me Q1 interest income (BAL_TYPE = 3)
```sql
SELECT
    ROUND(SUM({" + ".join(q1_months)}), 2) AS q1_interest_income
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```'''}

Q2 QUARTER (Apr-Jun):
Status: {q2_status.upper()}
{f"Months: {', '.join(q2_months)}" if q2_status != "not_available" else f"Error: {time_reference.get('Q2_Error', 'Not available')}"}

{"" if q2_status == "not_available" else f'''Example: Give me Q2 loans (BAL_TYPE = 1)
```sql
SELECT
    ROUND(SUM({"BUSINESS_DAY" if q2_status == "in_progress" else q2_months[-1]}), 2) AS q2_loans
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me Q2 revenue (BAL_TYPE = 3)
```sql
SELECT
    ROUND(SUM({" + ".join(q2_months)}), 2) AS q2_revenue
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```'''}

Q3 QUARTER (Jul-Sep):
Status: {q3_status.upper()}
{f"Months: {', '.join(q3_months)}" if q3_status != "not_available" else f"Error: {time_reference.get('Q3_Error', 'Not available')}"}

{"" if q3_status == "not_available" else f'''Example: Give me Q3 deposits (BAL_TYPE = 1)
```sql
SELECT
    ROUND(SUM({"BUSINESS_DAY" if q3_status == "in_progress" else q3_months[-1]}), 2) AS q3_deposits
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me Q3 interest income (BAL_TYPE = 3)
```sql
SELECT
    ROUND(SUM({" + ".join(q3_months)}), 2) AS q3_interest_income
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```'''}

Q4 QUARTER (Oct-Dec):
Status: {q4_status.upper()}
{f"Months: {', '.join(q4_months)}" if q4_status != "not_available" else f"Error: {time_reference.get('Q4_Error', 'Not available')}"}

{"" if q4_status == "not_available" else f'''Example: Give me Q4 loans (BAL_TYPE = 1)
```sql
SELECT
    ROUND(SUM({"BUSINESS_DAY" if q4_status == "in_progress" else q4_months[-1]}), 2) AS q4_loans
FROM DM_MIS_DETAILS_VW1
WHERE MRL_LINE IN (#MRL_LINES#)
AND BAL_TYPE = 1
AND CCY_TYPE = 'BCY';
```

Example: Give me Q4 net income (BAL_TYPE = 3)
```sql
SELECT
    ROUND(SUM({" + ".join(q4_months)}), 2) AS q4_net_income
FROM DM_MIS_DETAILS_VW1
WHERE BAL_TYPE = 3
AND CCY_TYPE = 'BCY';
```'''}

=== PREVIOUS YEAR QUARTERS (PY Q1, PY Q2, PY Q3, PY Q4) ===
Note: Previous year quarters are ALWAYS AVAILABLE

PY Q1 QUARTER (Previous Year Jan-Mar):
Months: {', '.join(py_q1_months)}

{py_q1_block}

PY Q2 QUARTER (Previous Year Apr-Jun):
Months: {', '.join(py_q2_months)}

{py_q2_block}

PY Q3 QUARTER (Previous Year Jul-Sep):
Months: {', '.join(py_q3_months)}

{py_q3_block}

PY Q4 QUARTER (Previous Year Oct-Dec):
Months: {', '.join(py_q4_months)}

{py_q4_block}

=== KEY RULES SUMMARY ===
1. BAL_TYPE = 1 (Balance Sheet - Point in Time):
   - In-progress quarters: Use BUSINESS_DAY
   - Completed quarters: Use last month of the quarter
   
2. BAL_TYPE = 3 (Income Statement - Period Total):
   - Always sum all months in the quarter
   - In-progress: Include MTD
   - Completed: Sum all 3 complete months

3. Quarter Availability:
   - Check status before using
   - If not_available, suggest alternatives
   - Previous year quarters always available

4. Net Income / Net Profit:
   - Do NOT include MRL_LINE filter
   - Only use BAL_TYPE and CCY_TYPE filters
"""
    
    return section


def inject_quarter_rules_into_prompt(base_prompt: str, user_query: str) -> str:
    """
    Inject quarter rules section into the prompt if quarter keywords are detected.
    Injects BEFORE the "==== REFERENCE EXAMPLES ====" section.
    
    Args:
        base_prompt: The base dynamic prompt string
        user_query: User's natural language query
        
    Returns:
        Enhanced prompt with quarter rules if applicable, otherwise unchanged
    """
    if not should_include_quarter_rules(user_query):
        return base_prompt
    
    # Get time reference (will fetch business date from database)
    time_ref = create_time_reference_table()

    # Prefer a targeted snippet: if multiple quarters detected, build multi; else single; else full section
    multi = build_multi_quarter_rule(user_query, time_ref)
    if multi:
        quarter_section = multi
    else:
        targeted = build_targeted_quarter_rule(user_query, time_ref)
        if targeted:
            quarter_section = targeted
        else:
            # Fallback to full section
            quarter_section = build_quarter_rules_section(time_ref)
    
    # Find appropriate injection point (before "==== REFERENCE EXAMPLES ====")
    injection_marker = "\n==== REFERENCE EXAMPLES ===="
    if injection_marker in base_prompt:
        parts = base_prompt.split(injection_marker, 1)
        enhanced_prompt = parts[0] + quarter_section + "\n" + injection_marker + parts[1]
        return enhanced_prompt
    else:
        # Fallback: try before "User Query:"
        injection_marker = "\nUser Query:"
        if injection_marker in base_prompt:
            parts = base_prompt.split(injection_marker, 1)
            enhanced_prompt = parts[0] + quarter_section + "\n" + injection_marker + parts[1]
            return enhanced_prompt
        else:
            # Last fallback: append at end
            return base_prompt + "\n" + quarter_section 