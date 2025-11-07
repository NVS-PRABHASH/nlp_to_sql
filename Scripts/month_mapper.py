from datetime import datetime
from typing import Optional, Dict, List, Tuple
import re
from time_reference import create_time_reference_table

class MonthMapper:
    def __init__(self, business_date: str = None):
        """
        Initialize with a business date to generate time reference data.
        If no date is provided, uses the current date.
        """
        # Generate time reference data using the provided or current date
        self.time_ref = create_time_reference_table(business_date) if business_date else create_time_reference_table()
        self.month_map = self._build_month_map()
        
        # Map full month names and abbreviations to numbers
        self.month_name_to_num = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

    def _build_month_map(self) -> Dict[str, str]:
        """Build a mapping of 'MonthName Year' to column name"""
        month_map = {}
        
        # Include all entries that expose a MONTH label (covers Current_Month, Prior_*_Month, PY_*, etc.)
        for col, data in self.time_ref.items():
            if isinstance(data, dict) and 'MONTH' in data:
                month_label = data['MONTH']
                if isinstance(month_label, str):
                    month_map[month_label.lower()] = col
        
        # Also include MTD if available
        if 'MTD' in self.time_ref and 'TIME_PERIOD' in self.time_ref['MTD']:
            # Extract just the month and year part from MTD (removing " MTD" suffix)
            mtd_month = self.time_ref['MTD']['TIME_PERIOD'].replace(' MTD', '')
            month_map[mtd_month.lower()] = 'MTD'
            
        return month_map

    def get_month_number(self, month_name: str) -> Optional[int]:
        """Convert month name or abbreviation to month number (1-12)"""
        return self.month_name_to_num.get(month_name.lower().strip())

    def find_matching_column(self, month_name: str, year: Optional[int] = None) -> Optional[Tuple[str, str]]:
        """
        Find the column name that matches the given month and optional year.
        
        Args:
            month_name: Name or abbreviation of the month (e.g., 'jan', 'january')
            year: Optional year to match (if None, matches any year)
            
        Returns:
            Tuple of (column_name, full_month_year) or None if no match found
        """
        # First try to get month number from the name
        month_num = self.get_month_number(month_name)
        if not month_num:
            return None

        # Search through our month map
        for month_year, col in self.month_map.items():
            # Extract month and year from the stored month_year string
            parts = month_year.lower().split()
            if not parts:
                continue
                
            stored_month_name = parts[0]
            stored_month_num = self.get_month_number(stored_month_name)
            
            # If we have a match on month number
            if stored_month_num == month_num:
                # If year is specified, check that too
                if year is not None and len(parts) > 1:
                    try:
                        stored_year = int(parts[1])
                        if stored_year != year:
                            continue
                    except (ValueError, IndexError):
                        continue
                return col, month_year

        return None

    def extract_month_year_from_text(self, text: str) -> Optional[Tuple[str, Optional[int]]]:
        """
        Extract month (and optionally year) from text.
        Returns (month_name, year) where year is None if not found.
        """
        # First try to match full month names
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b'
        match = re.search(month_pattern, text, re.IGNORECASE)
        if not match:
            return None
            
        month_name = match.group(1)
        
        # Try to extract explicit year (4-digit number)
        year_match = re.search(r'\b(20\d{2})\b', text)
        year = int(year_match.group(1)) if year_match else None

        # If no explicit year, handle relative year phrases
        if year is None:
            tl = text.lower()

            # Determine the reference current year from time_ref
            current_year: Optional[int] = None
            try:
                if 'Current_Month' in self.time_ref and 'MONTH' in self.time_ref['Current_Month']:
                    parts = self.time_ref['Current_Month']['MONTH'].split()
                    if len(parts) > 1 and parts[1].isdigit():
                        current_year = int(parts[1])
                elif 'MTD' in self.time_ref and 'TIME_PERIOD' in self.time_ref['MTD']:
                    period = self.time_ref['MTD']['TIME_PERIOD'].replace(' MTD', '')
                    parts = period.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        current_year = int(parts[1])
            except Exception:
                current_year = None

            if current_year is not None:
                if 'last year' in tl or 'previous year' in tl:
                    year = current_year - 1
                elif 'this year' in tl or 'current year' in tl:
                    year = current_year
                elif 'next year' in tl:
                    year = current_year + 1
        
        return month_name, year

    def get_column_for_text(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Find the column name that matches a month mentioned in the text.
        
        Args:
            text: Text that might contain a month (and optionally year)
            
        Returns:
            Tuple of (column_name, full_month_year) or None if no match found
        """
        # First try to match special cases
        text_lower = text.lower()
        if 'current month' in text_lower or 'this month' in text_lower:
            if 'MTD' in self.time_ref:
                return 'MTD', self.time_ref['MTD'].get('TIME_PERIOD', self.time_ref['Current_Month']['MONTH'] if 'Current_Month' in self.time_ref else '')
            return None
            
        if 'last month' in text_lower or 'previous month' in text_lower:
            if 'Current_Month' in self.time_ref:
                return 'Current_Month', self.time_ref['Current_Month']['MONTH']
            return None
            
        # If no special cases match, try to extract month/year from text
        result = self.extract_month_year_from_text(text)
        if not result:
            return None
            
        month_name, year = result
        return self.find_matching_column(month_name, year)

    def get_columns_for_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Return a list of (column_name, full_month_year) for all months mentioned in the text.
        Keeps existing single-month behavior available via get_column_for_text.
        """
        results: List[Tuple[str, str]] = []
        text_lower = text.lower()

        # Handle special phrases possibly appearing together
        if 'current month' in text_lower or 'this month' in text_lower:
            if 'MTD' in self.time_ref:
                results.append(('MTD', self.time_ref['MTD'].get('TIME_PERIOD', self.time_ref['Current_Month']['MONTH'] if 'Current_Month' in self.time_ref else '')))

        if 'last month' in text_lower or 'previous month' in text_lower:
            if 'Current_Month' in self.time_ref:
                results.append(('Current_Month', self.time_ref['Current_Month']['MONTH']))

        # Infer explicit or relative year if present
        inferred_year: Optional[int] = None
        ym = re.search(r"\b(20\d{2})\b", text)
        if ym:
            inferred_year = int(ym.group(1))
        else:
            current_year: Optional[int] = None
            try:
                if 'Current_Month' in self.time_ref and 'MONTH' in self.time_ref['Current_Month']:
                    parts = self.time_ref['Current_Month']['MONTH'].split()
                    if len(parts) > 1 and parts[1].isdigit():
                        current_year = int(parts[1])
                elif 'MTD' in self.time_ref and 'TIME_PERIOD' in self.time_ref['MTD']:
                    period = self.time_ref['MTD']['TIME_PERIOD'].replace(' MTD', '')
                    parts = period.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        current_year = int(parts[1])
            except Exception:
                current_year = None

            if current_year is not None:
                if 'last year' in text_lower or 'previous year' in text_lower:
                    inferred_year = current_year - 1
                elif 'this year' in text_lower or 'current year' in text_lower:
                    inferred_year = current_year
                elif 'next year' in text_lower:
                    inferred_year = current_year + 1

        # Find all month mentions
        month_pattern = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
        seen = set()
        for m in re.finditer(month_pattern, text, re.IGNORECASE):
            month_name = m.group(1)
            key = month_name.lower()
            if key in seen:
                continue
            seen.add(key)

            match = self.find_matching_column(month_name, inferred_year)
            if match:
                results.append(match)

        return results


# =========================================================================
# Prompt injection helpers for month handling
# =========================================================================

def _build_month_rules_section(pairs: List[Tuple[str, str]]) -> str:
    """
    Build a concise month handling section given (column, month_year) pairs.
    """
    if not pairs:
        return ""

    lines = ["\n==== MONTH HANDLING ===="]
    lines.append("Detected month references and corresponding time reference columns:")
    for col, month_year in pairs:
        # e.g., "- JAN 2024 -> Use column: Current_Month"
        lines.append(f"- {month_year} -> Use column: {col}")

    lines.append("")
    lines.append("Rules:")
    lines.append("- Use the mapped column(s) when building filters for the requested month(s).")
    lines.append("- If multiple months are mentioned, use a CASE statement to produce separate columns per month")
    return "\n".join(lines) + "\n"


def inject_month_rules_into_prompt(base_prompt: str, user_query: str) -> str:
    """
    Inject a month handling section into the prompt if a specific month is detected.
    Inject BEFORE the "==== REFERENCE EXAMPLES ====" section when present.
    """
    try:
        mapper = MonthMapper()
        pairs = mapper.get_columns_for_text(user_query)
        if not pairs:
            single = mapper.get_column_for_text(user_query)
            if single:
                pairs = [single]
        # If still nothing, return unchanged prompt
        if not pairs:
            return base_prompt

        month_section = _build_month_rules_section(pairs)
        if not month_section:
            return base_prompt

        injection_marker = "\n==== REFERENCE EXAMPLES ===="
        if injection_marker in base_prompt:
            parts = base_prompt.split(injection_marker, 1)
            return parts[0] + month_section + "\n" + injection_marker + parts[1]
        else:
            # Fallback: try before "User Query:"
            alt_marker = "\nUser Query:"
            if alt_marker in base_prompt:
                parts = base_prompt.split(alt_marker, 1)
                return parts[0] + month_section + "\n" + alt_marker + parts[1]
            # Last fallback: append
            return base_prompt + "\n" + month_section
    except Exception:
        return base_prompt

# Example usage:
if __name__ == "__main__":
    # Create mapper with current date (or pass a specific date like "2023-10-29")
    mapper = MonthMapper()  
    
    # # Example queries
    # test_queries = [
    #     "Show me data for january",
    #     "What was the value in feb 2023?",
    #     "march numbers",
    #     "apr-2023 report",
    #     "may data",
    #     "june 2023 MTD",
    #     "current month",
    #     "last month",
    #     "previous month"
    # ]
    
    # # Add some dynamic examples based on the actual time reference
    # if 'Current_Month' in mapper.time_ref:
    #     current_month = mapper.time_ref['Current_Month']['MONTH']
    #     test_queries.extend([
    #         f"Show {current_month} data",
    #         f"What was the value in {current_month}?"
    #     ])
    
    # print("Month Mapper Test:")
    # print("-----------------")
    # for query in test_queries:
    #     result = mapper.get_column_for_text(query)
    #     if result:
    #         col, month_year = result
    #         print(f"Query: '{query}' → Column: {col}, Month: {month_year}")
    #     else:
    #         print(f"Query: '{query}' → No matching month found")

    while True:
        query = input("Enter your query: ")
        multi = mapper.get_columns_for_text(query)
        if multi:
            for col, month_year in multi:
                print(f"Query: '{query}' → Column: {col}, Month: {month_year}")
        else:
            # If the query has an explicit year or relative year phrase, don't fall back
            ql = query.lower()
            has_year_hint = bool(re.search(r"\b(20\d{2})\b", query)) or (
                'last year' in ql or 'previous year' in ql or 'this year' in ql or 'current year' in ql or 'next year' in ql
            )
            if has_year_hint:
                print(f"Query: '{query}' → No matching month found")
            else:
                result = mapper.get_column_for_text(query)
                if result:
                    col, month_year = result
                    print(f"Query: '{query}' → Column: {col}, Month: {month_year}")
                else:
                    print(f"Query: '{query}' → No matching month found")
        