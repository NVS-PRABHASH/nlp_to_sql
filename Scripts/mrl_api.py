
import json
import pandas as pd
import requests
import re

import traceback
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool


# Import local modules
from mrl_rag import MRLChatbot
from config import get_config, setup_logging, build_oracle_url


# Configure logging
setup_logging()
logger = logging.getLogger("ai_insight.mrl_api")
# --------------------------
# 1. vLLM Server Management
# --------------------------
_cfg = get_config()
VLLM_API_URL = ((_cfg.get("vllm") or {}).get("api_url") or "")
MODEL_NAME = ((_cfg.get("vllm") or {}).get("model_name") or "")


# --------------------------
# 2. Database Connection
# --------------------------
def get_engine():
    db_cfg = _cfg.get("database", {})
    pool_cfg = db_cfg.get("pool", {})
    url = build_oracle_url(db_cfg)
    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=int(pool_cfg.get("pool_size", 5)),
        max_overflow=int(pool_cfg.get("max_overflow", 10)),
        pool_timeout=int(pool_cfg.get("pool_timeout", 30)),
        pool_recycle=int(pool_cfg.get("pool_recycle", 3600)),
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM DUAL"))
        return engine
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise

# --------------------------
# 3. Fetch Complete Hierarchy from DB
# --------------------------
def fetch_complete_hierarchy(engine):
    """
    Fetch complete hierarchy from the AI_MRL_EXPLANATION table.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        tuple: (valid_values, valid_values_summary, full_df)
    """
    # logger.info("Fetching complete hierarchy from database...")
    levels = {}
    
    with engine.connect() as conn:
        try:
            # Get the table name from config
            table_name = _cfg.get('database', {}).get('tables', {}).get('mrl_explanation', 'AI_MRL_EXPLANATION')
            
            # First, check if the table exists and get its actual name
            result = conn.execute(text(f"""
                SELECT table_name 
                FROM all_tables 
                WHERE UPPER(table_name) = UPPER(:table_name)
            """), {'table_name': table_name})
            
            table_info = result.fetchone()
            
            if not table_info:
                logger.error(f"Table {table_name} not found in the database")
                return {}, "Table not found", pd.DataFrame()
                
            actual_table_name = table_info[0]  # Get the actual case of the table name
            
            # Get all columns for the table
            result = conn.execute(text("""
                SELECT column_name 
                FROM all_tab_columns 
                WHERE UPPER(table_name) = UPPER(:table_name)
                ORDER BY column_id
            """), {'table_name': actual_table_name})
            
            # Filter for level columns (case-insensitive)
            all_columns = [row[0] for row in result.fetchall()]
            level_columns = [col for col in all_columns 
                           if col.upper().startswith('LEVEL_')]
            
            if not level_columns:
                logger.debug("No level_x columns found in the table")
                return {}, "No level columns found", pd.DataFrame()
                
            # logger.info(f"Found level columns: {', '.join(level_columns)}")
            
            # Get distinct values for each level column
            for col in level_columns:
                try:
                    # Use direct SQL with proper quoting for Oracle
                    sql = f"""
                        SELECT DISTINCT "{col}" as val 
                        FROM "{actual_table_name}" 
                        WHERE "{col}" IS NOT NULL 
                        AND RTRIM("{col}") IS NOT NULL
                        AND "{col}" NOT IN ('Not Applicable', 'NA', 'N/A', '-', ' ')
                        ORDER BY "{col}"
                    """
                    
                    result = conn.execute(text(sql))
                    values = [str(row[0]).strip() for row in result.fetchall() 
                             if row[0] is not None and str(row[0]).strip()]
                    
                    if values:
                        levels[col.lower()] = values
                    else:
                        pass
                        levels[col.lower()] = []
                        
                except Exception as e:
                    logger.debug(f"Error processing {col}: {str(e).split(chr(10))[0]}")
                    levels[col.lower()] = []
            
            # Get the full dataset using direct SQL
            full_df = pd.read_sql(f'SELECT * FROM "{actual_table_name}"', conn)
            # Loaded rows
            
        except Exception as e:
            logger.exception(f"Error in fetch_complete_hierarchy: {e}")
            return {}, "Error loading hierarchy", pd.DataFrame()
    
    # Create a summary of valid values for the prompt
    valid_values_summary = ""
    for level, values in levels.items():
        if values:  # Only include levels that have values
            sample = values[:10]
            valid_values_summary += f"{level.upper()}: {', '.join(sample)}{'...' if len(values) > 10 else ''} ({len(values)} total)\n"
    
    if not valid_values_summary:
        logger.debug("No valid level values found in the table")
    
    return levels, valid_values_summary, full_df

# --------------------------
# 4. Get Full Hierarchy from DB
# --------------------------
def get_full_hierarchy():
    """
    Get the full hierarchy from the database with improved error handling.
    """
    try:
        engine = get_engine()
        return fetch_complete_hierarchy(engine)
    except Exception as e:
        logger.exception(f"Error fetching hierarchy: {e}")
        return {}, "Error loading hierarchy", pd.DataFrame()


# --------------------------
# 6. Query LLM via vLLM API
# --------------------------
banking_concept_mapping = {
    "deposit": {
        "level_1": ["Liabilities"],
        "level_2": ["Funded Liability"],
        "level_3": ["Customer Deposits", "Deposits from Banks"],
        "level_4": ["Current Account", "Savings Account", "Fixed Deposit", "Call Deposits", 
                   "Margin Accounts", "Collections Accounts", "Bank Deposit"],
        "level_5": ["Current Account", "Savings Account", "Call Deposits", "Locker Deposits",
                   "Fixed Deposits", "Fixed Deposit Recurring", "Margin Accounts", 
                   "Suspense Term Deposits", "Collections Accounts", "Islamic Deposits",
                   "Private Bank", "Foreign Banks"],
        "level_6": ["Current Accounts (LCY)", "Current Accounts (FCY)", "Savings Accounts (LCY)",
                   "Savings Accounts (FCY)", "Call Deposits (LCY)", "Call Deposits (FCY)",
                   "Locker Deposits", "Fixed Deposits (LCY)", "Fixed Deposits (FCY)",
                   "Fixed Deposit Recurring (LCY)", "Fixed Deposit Recurring (FCY)",
                   "Margin Accounts", "Suspense Term Deposits (LCY)", "Suspense Term Deposits (FCY)",
                   "Collections Accounts", "Islamic Customer Deposits (LCY)", "Islamic Customer Deposits (FCY)",
                   "Current Deposit of Pvt Bank (IR)", "Current Deposits of Foreign Banks",
                   "Saving Deposits of Private Bank", "Fixed Deposits of Private Bank",
                   "Current - Banks FCY"],
        "level_7": ["Normal", "NPA"],
        "level_8": ["LCY", "FCY"]
    },
    "placement": {
        "level_1": ["Assets"],
        "level_2": ["Funded Assets"],
        "level_3": ["Cash Equivalents", "Interbank Placements"],
        "level_4": ["Placements", "Placement Abroad"],
        "level_5": ["Placement with Local Banks", "Overnight Placements", "Call Placements", 
                   "Fixed Placements", "Range Accrual Placements"],
        "level_6": ["Overnight Placements in Kenya", "Overnight Placements Abroad (LCY)", 
                   "Overnight Placements Abroad (FCY)", "Call Placements in Kenya", 
                   "Call Placements Abroad (LCY)", "Call Placements Abroad (FCY)",
                   "Fixed Placements in Kenya", "Fixed Placements Abroad", "DCD Placement",
                   "Range Accrual Placements Kenya", "Range Accrual Placements Abroad"],
        "level_7": ["Normal", "NPA"],
        "level_8": ["LCY", "FCY"]
    },
    "asset": {
        "level_1": ["Assets"],
        "level_2": ["Balances with Other Banks", "Cash and Cash Equivalents", "Contingent Asset", 
                   "Contra Assets", "Fixed Asset", "Loan Loss Provision", "Funded Assets", 
                   "Other Assets", "Prepayment", "Sundry"],
        "level_3": ["Cash", "Cash Equivalents", "Clearing/Control Accounts", "Commercial Papers", 
                   "Corporate Bonds", "Due from Banks", "Fixed Asset", "Funded Assets", 
                   "Interbank Placements", "Interest Receivable", "Investment Securities", 
                   "Loans & Advances", "Non-Funded Assets", "Other Asset", "Other Assets", 
                   "Prepayment", "Provision", "Reverse Repo", "Securities",
                   "Sundry", "Suspense Accounts", "Treasury Bills/Bonds"],
        "level_4": ["Accumated Depreciation", "Advances", "Cash", "Cash Equivalents", 
                        "Clearing/Control Accounts", "Commercial Papers", "Computed Depreciation", 
                        "Corporate Bonds", "Cost", "Cross Currency Swaps (Paid)", "Derivatives", 
                        "Due from Banks", "Fees Suspended", "Funded Assets", "General", 
                        "Infrastructure Bonds", "Interest Receivable", "Interest Suspended", 
                        "Investment Securities", "Loans", "Non-Funded Assets", "Other Asset", 
                        "Other Assets", "Other Investment Securities", "Placement Abroad", 
                        "Placement with Local Banks", "Placements", "Pool Provision", "Prepayment", 
                        "Prm Receivable", "Public Securities", "Reverse Repo", "Share Subscription", 
                        "Specific", "Sundry", "Treasury Bills", "Treasury Bonds", "UID"],
                "level_5": [
                    "AFS", "HFT", "Commercial Papers", "HTM Floating", "AFS Floating", 
                    "Floating Rate Notes", "Public Securities", "Cross Currency Swaps (Paid)", 
                    "Derivatives", "Other Investment Securities", "Investment Securities", 
                    "Share Subscription", "Overdraft", "Term Loan", "Mortgage Loans", 
                    "Demand Loans", "Asset Finance", "Staff Loan", "Other Loan", "Credit Card", 
                    "Personal Loan", "Cash", "Cash Equivalents", "Account with Central Bank", 
                    "Account with Local Banks", "Placement with Local Banks", "Overnight Placements", 
                    "Call Placements", "HTM", "HTM Short", "HTM Prem", "HTM Disc", "AFS Off Shore", 
                    "AFS Prem", "AFS Disc", "AFS Revaluation", "HFT Prem", "HFT Disc", 
                    "HFT Revaluation", "Corporate Bonds", "Prepayment", "Development Costs", 
                    "Clearing", "Transit", "Suspense", "Sundry Debtors", "Cash Shortages", 
                    "Due from Subsidiaries", "Proxy Accounts", "Receivable", "Deferred Expenses", 
                    "Deferred Tax", "Clearing/Control Accounts", "Other Assets", 
                    "Furniture; Fittings & Fixtures", "Computer Hardware", "Software", 
                    "Motor Vehicles", "Leasehold lmprovement", "Freehold Land", "Freehold Building", 
                    "Leasehold Building", "Prepaid Operating Lease", "Deals", "Overdarft", 
                    "Other Lending", "Card", "Advances", "Pool", "Interest in Suspense", 
                    "Fees Suspended", "Repo Deal", "Banks in Kenya", "Banks Abroad", 
                    "Over Night Placements", "Call Placements", "Fixed Placements", 
                    "Dual Currency Deposit - DCD Placements", "Range Accruals", 
                    "Term Auction Deposits", "HTM Treasury Bills", "AFS Treasury Bills", 
                    "HFT Treasury Bills", "Interest Rate Swap Assets", 
                    "Cross Currency Swap Assets", "Funding Swap Assets", "Currency Swap Gain", 
                    "Treasury Bonds", "Govt Bonds", "Infrastructure Bonds", "Commercial Paper", 
                    "Bills Discounted", "Overdrafts", "Long Term Loans", "MID Term Loans", 
                    "Short Term Loans", "Demand Loans", "Housing Mortgage", "Hire Purchase", 
                    "Insurance Premium", "Asset Finance Loans", "Personal Loans", "IPO Finance", 
                    "LPO Finance", "Women Entp", "SME Finance", "Staff Loans", "Credit Cards", 
                    "Islamic Finance", "Placements", "Borrowings", "Cont Spot Buy", "Outlong", 
                    "Outshort", "Interest Rate Swaps", "Cross Currency Swaps", "Funding Swaps", 
                    "Letters Of Credit", "Acceptances/Collections", "Guarantees", 
                    "Treasury Bills", "Corporate Bond", "Other Contingent Assets"
                ],
            },
            "loan": {
                "level_1": ["Assets"],
                "level_2": ["Funded Assets"],
                "level_3": ["Loans & Advances"],
                "level_4": ["Loans"],
                "level_5": ["Term Loan", "Asset Finance", "Other Loan","Mortgage Loans","Demand Loans","Staff Loan","Personal Loan"],
                "level_6": ["Hire Purchase Loans","Long Term Loans","Mid Term Loan", "Short Term Loans","Mortgage Loans",
                "Demand Loans","Insurance Premium Financing","Staff Loan","Invoice Discounting",
                "Bills & Notes Discounted / NPA","Bills / Cheques Purchased","Personal Loan","IPO Finance",
                "TAALIMIKA (STUDENT LOAN)","LPO Finance","Medium Term Loans MMK","Islamic Finance","Women Entp",
                "SME Finance","Asset Finance Loans Accounts"],
                "level_7": ["NPA", "Normal"],
                "level_8": ["FCY", "LCY"]
            },
            "advance": {
                "level_1": ["Assets"],
                "level_2": ["Funded Assets"],
                "level_3": ["Loans & Advances"],
                "level_4": ["Advances"],
                "level_5": ["Overdraft", "Credit Card"],
                "level_6": ["Overdraft Accounts", "Credit Card"],
                "level_7": ["NPA", "Normal"],
                "level_8": ["FCY", "LCY"]
            },
            "revenue": {
                "level_1": ["Assets", "Investments", "Income"],  # Include ALL revenue sources
                "level_2": ["Funded Assets", "Non Funded Income", "Other Income", "Treasury Interest Income"],
                "level_3": ["Cash Equivalents", "Interbank Placements", "Loans & Advances", "Treasury Bills/Bonds",
                           "Corporate Bonds", "Commercial Papers", "Securities", "Commission", "Fee", 
                           "Gain on Sale", "Other Income", "Treasury Interest Income"],
                "level_4": ["Placement with Local Banks", "Placement Abroad", "Loans", "Advances", 
                           "Term Loan", "Mortgage Loans", "Demand Loans", "Staff Loan",
                           "Other Loan", "Credit Card", "Overdraft", "Treasury Bills", 
                           "Treasury Bonds", "Corporate Bonds", "Commercial Papers"],
                "level_7": ["Normal", "NPA"],
                "level_8": ["LCY", "FCY"]
            },
            "expense": {
                "level_1": ["Expenses"],
                "level_2": ["Expenses","Staff Expense","Interest Expense", "Premises Expense","Other Expense"],
                "level_3": [ "Suspense Accounts", "Deposit Expense", "Borrowing Expense", "Staff Expense", "Treasury Interest Expense", 
                           "Premises Expense","Tax", "Depreciation & Ammortization", "Other Expense"],
                "level_4": ["Withholding Tax","Staff Expense", "VAT", "PAYE", "TAX", "Cheque Leaves", "Withholding VAT", "Excise Duty",
                           "Transit Account", "Interest Suspended", "Write off", "Other Expense","Corporation Tax"]
            },
            "liability": {
                "level_1": ["Liabilities"],
                "level_2": ["Current Liabilities", "Funded Liability", "Other Liabilities", 
                            "Deferred Revenue", "Equity", "Contingent Liability"],
                "level_3": ["Deferred Revenue", "Borrowings From Banks", 
                            "Borrowings From Non-Banks", "Customer Deposits", "Bills Payable",
                            "Deposits from Banks", "Interest Payable", "Unearned Income",
                            "Other Liability", "Interbank", "Brokerage fees", "Other Liabilities", "Equity", "Contingent Liability"],
                "level_4": ["Unearned Income", "Borrowing Local Bank", "Borrowings Abroad", "Borrowing Local Non-Bank", "Current Account", "Savings Account",
                            "Call Deposits", "Fixed Deposit", "Bills Payable", "Margin Accounts", "Collections Accounts", "Bank Deposit",
                            "Overnight Borrowings", "Call Borrowings", "Fixed Borrowings", "DCD", "Range Accrual Borrowings", "Prm Payable",
                            "Interest Payable","Unearned Income","Sundry Account","Suspense Account","Transit","Unclaimed",
                            "Cheques","Drafts","Prepaid Card","Transfers","Settlement","Accruals","Guarantees","Brokerage fees payable",
                            "Creditors","Provision",
                            "Other Liabilities","Dividend","offset","Equity","Revenue Reserves",
                            "Unappropriated Profit","Profit & Loss","Proposed Dividend","Fixed Assets","Reserve - AFS",
                            "Revaluation Reserves","Statutory Reserves","Other Reserves","General Reserves","Contingent Liability"],
                "level_5":["Mid Term Borrowing","General","Borrowings with Banks in  Kenya","Borrowings with Banks Abroad","Overnight Borrowings",
                            "Call Borrowings","Call Borrowings Abroad",
                            "Fixed Borrowings","Fixed Borrowings Abroad",
                            "DCD Borrowings","Range Accrual Borrowings  Kenya","Range Accrual Borrowings Abroad",
                            "Cross Currency Swap Received","Short Term Borrowing",
                            "Long Term Borrowing","Money Market","Debt Capital","Borrowing from Non-Banks"],
                "level_6": ["Borrowings with Banks in Kenya (LCY)","Borrowings with Banks in Kenya (FCY)",
                            "Borrowings with Banks Abroad (LCY)","Borrowings with Banks Abroad (FCY)",
                            "Overnight Borrowings in Kenya (LCY)","Overnight Borrowings in Kenya (FCY)",
                            "Overnight Borrowings Abroad (LCY)","Overnight Borrowings Abroad (FCY)",
                            "Call Borrowings in Kenya (LCY)","Call Borrowings in Kenya (FCY)",
                            "Call Borrowings Abroad (LCY)","Call Borrowings Abroad (FCY)",
                            "Fixed Borrowings in Kenya (LCY)","Fixed Borrowings in Kenya (FCY)",
                            "Fixed Borrowings Abroad (LCY)","Fixed Borrowings Abroad (FCY)",
                            "DCD Borrowings (FCY)","DCD Borrowings (LCY)",
                            "Range Accrual Borrowings Kenya (LCY)","Range Accrual Borrowings Kenya (FCY)",
                            "Range Accrual Borrowings Abroad (LCY)","Range Accrual Borrowings Abroad (FCY)",
                            "Cross Currency Swap Received (LCY)","Cross Currency Swap Received (FCY)",
                            "Cross Currency Swap Received (FCY)","Short Term Borrowing","Short Term Borrowing (FCY)",
                            "Mid Term Borrowing","Long Term Borrowing","Money Market","Debt Capital","Borrowing from Non-Banks",
                            "Current Accounts (LCY)","Current Accounts (FCY)","Savings Accounts (LCY)","Savings Accounts (FCY)",
                            "Call Deposits (LCY)","Call Deposits (FCY)","Locker Deposits","Fixed Deposits (LCY)","Fixed Deposits (FCY)",
                            "Fixed Deposit Recurring (LCY)","Fixed Deposit Recurring (FCY)","Margin Accounts","Suspense Term Deposits (LCY)",
                            "Suspense Term Deposits (FCY)","Collections Accounts","Islamic Customer Deposits (LCY)","Islamic Customer Deposits (FCY)",
                            "Current Deposit of Pvt Bank (IR)","Current Deposits of Foreign Banks","Saving Deposits of Private Bank","Fixed Deposits of Private Bank",
                            "Current - Banks FCY","Prepayments - Deposit Protection Fund","Prepayments - Banking Licence","Prepayments - Insurance","Prepayments - Medical Outpatient Insurance","Prepayments - Group Life Insurance","Prepayments - Group Personal Accident","Prepayments - Rent","Prepayments - Leased assets","Prepayments - Mobile Banking",
                            "Prepayments - Mobile Top-ups","Prepayments - Other expenses","Deposits - Rent","Deposits - Telephone","Deposits - Electricity",
                            "Deposits - Water","Guarantee Deposits","WIP - Development Costs","Placements (LCY)","Placements (FCY)","Borrowings (LCY)",
                            "Borrowings (FCY)","Cont Spot Buy (LCY)","Cont Spot Buy (FCY)","Outlong (LCY)",
                            "Outlong (FCY)","Outshort (LCY)","Outshort (FCY)", "Interest Rate Swaps (LCY)",
                            "Interest Rate Swaps (FCY)","Cross Currency Swaps (LCY)","Cross Currency Swaps (FCY)","Funding Swaps (LCY)",
                            "Funding Swaps (FCY)","Letters Of Credit","Acceptances/Collections","Guarantees",
                            "Treasury Bills","Treasury Bonds","Corporate Bond","Other Contingent Assets",
                            "Int Payable - Overnight Brrwgs in Kenya (LCY)","Int Payable - Overnight Brrwgs in Kenya (FCY)",
                            "Int Payable - Overnight Brrwgs Abroad (LCY)","Int Payable - Overnight Brrwgs Abroad (FCY)",
                            "Int Payable - Call Borrowings in Kenya (LCY)","Int Payable - Call Borrowings in Kenya (FCY)",
                            "Int Payable - Call Borrowings Abroad (LCY)","Int Payable - Call Borrowings Abroad (FCY)",
                            "Int Payable - Fixed Borrowings in Kenya (LCY)","Int Payable - Fixed Borrowings in Kenya (FCY)",
                            "Int Payable - Fixed Borrowings Abroad (LCY)","Int Payable - Fixed Borrowings Abroad (FCY)",
                            "Interest Payable - DCD","Int Payable - Range Acc Brwg Kenya (LCY)","Int Payable - Range Acc Brwg Kenya (FCY)",
                            "Int Payable - Range Acc Brwg Abroad (LCY)","Int Payable - Range Acc Brwg Abroad (FCY)",
                            "Prm Payable - Interest Rate Swap L (LCY)","Int Payable - Interest Rate Swap L (LCY)",
                            "Prm Payable - Interest Rate Swap L (FCY)","Int Payable - Interest Rate Swap L (FCY)",
                            "Prm Payable - Cross Currency Swap (LCY)","Int Payable - Cross Currency Swap (LCY)",
                            "Prm Payable - Cross Currency Swap (FCY)","Int Payable - Cross Currency Swap (FCY)",
                            "Int Payable - Funding Swap Liabilites (LCY)","Int Payable - Funding Swap Liabilites (FCY)",
                            "Interest Payable - Cur Accounts (LCY)","Interest Payable - Cur Accounts (FCY)",
                            "Interest Payable - Custodian Current Account","Interest Payable - Sav Accounts (LCY)",
                            "Interest Payable - Sav Accounts (FCY)","Interest Payable - Online Savers",
                            "Interest Payable - Call Deposits (LCY)","Interest Payable - Call Deposits (FCY)",
                            "Interest Payable - Locker Deposits","Interest Payable - Fixed Deposits (LCY)",
                            "Interest Payable - Fixed Deposits (FCY)","Interest Payable - Fixed Dep Recurring (LCY)",
                            "Interest Payable - Fixed Dep Recurring (FCY)","Interest Payable - Islamic Customer Deposits",
                            "Interest Payable - Cur - Banks (LCY)","Interest Payable - Cur - Banks (FCY)",
                            "Interest Payable - Long Term Borrowings (LCY)","Interest Payable - Long Term Borrowings (FCY)"
                            ],
            },

            "income":{
                "level_1":["Income"],
                "level_2":["Non Funded Income","Other Income","Treasury Interest Income"],
                "level_3":["Commission","Gain on Sale","foreign exchange dealings","Trading Profit/Loss","Revaluation Profit/Loss","Other Income","Treasury Interest Income","Fee","Concessions"]
            },

            "non_funded_income": {
                "level_1": ["Income"],
                "level_2": ["Non Funded Income"],
                "level_3": ["Commission","Gain on Sale","foreign exchange dealings","Trading Profit/Loss","Revaluation Profit/Loss","Fee","Concessions"],
                "level_4": ["Collections","Transfers","Cash Deposits","Cash Handling","Central Bnk Bonds","Direct Debit","Mobile Top-Ups","Changamka Cards",
                    "Others","Operations","Swift Charges","Courier Charges","Export LC's","Import LC's",
                    "Bills Acceptances","Avalised Bills","LC's Swift Charges","Trade Services Rebates",
                    "Release Of Title Documents","Guarantees","Cards","HTM Securities","AFS Securities","HFT Securities","Mortgage Processing fees","Insurance Cost","Loan Fees","Option Fees","Commitment Fees","Commitment  Reversals","Fees","Accounts","Fin Instruments"],
                "level_8": ["LCY", "FCY"]
            },
        }
     
def extract_json_object(text: str):
    """Attempt to extract the first JSON object/array from arbitrary text."""
    if not text:
        return None

    text = text.strip()

    # Remove Markdown code fences if present
    if text.startswith("```") and text.endswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                return obj
            except json.JSONDecodeError:
                continue

    return None


def get_hierarchy_from_query(query: str) -> dict:
    """
    Enhanced function to extract hierarchy from natural language query using vLLM API.
    Uses a more structured prompt to ensure valid JSON output and better adherence to hierarchy values.
    """
    prompt = f"""
    You are a senior banker with IFRS GL expertise. 
    Interpret the query and map it to the set of possible hierarchy values in {banking_concept_mapping} for each significant banking terms. 
    Correct spelling, expand abbreviations (HP→ Hire Purchase, FD→Fixed Deposit, CA→Current Account, SA→Savings Account, OD→Overdraft, NFI→Non Funded Income, NII→Net Interest Income, NIM→Net Interest Margin, etc.).
    You must extract all applicable attributes across the following hierarchy, with precise and exact matches to terminology from the official MRL taxonomy.
    
    [CATEGORY CLASSIFICATION RULE]

    - Always classify a question into the correct top-level financial category before generating SQL.  
    - If the query explicitly mentions "liability" → classify under "liability".  
    - If the query explicitly mentions "deposit(s)" → classify under "deposit".  
    - If the query explicitly mentions "loan(s)" or "asset finance" → classify under "loan".  
    - If the query explicitly mentions "advance(s)" → classify under "advance".  
    - If the query explicitly mentions "revenue" or "income" not linked to loans/deposits → classify under "revenue".  
    - If the query explicitly mentions "non-funded income" (fees, commissions, charges) → classify under "non_funded_income".    
    - If the query explicitly mentions "expense(s)" → classify under "expense".
    RULES TO STRICTLY FOLLOW:
    1. Output ONLY valid JSON (parsable by json.loads).
    2. Top-level keys = relevant banking terms from the query (e.g., "loan", "deposit", "placement", "interest income",...).
    3. Each level value = array (even single value). Use exact hierarchy terms only.
    4. Omit unknown/uncertain levels, do not invent values.
    5. ONLY use the exact hierarchy values provided. NEVER assign a value to a higher or lower level than it exists in the official hierarchy.
    6. Always map currency: LCY→level_8["LCY"], FCY→level_8["FCY"].
    7. For parentheses in query (e.g. "Current Accounts (FCY)"):
    - Use account name in correct level (e.g. level_6)
    - Map currency separately in level_8.
    8. When the user mentions "Kenya" (or the Kenya entity),
        → the model should include both LCY and FCY — not just LCY.
    9. Only include relevant hierarchy levels from query keywords (ignore filler words).
    10. If query only mentions "deposits" (no detail) → always include both Customer Deposits + Deposits from Banks.
    11. For "profit"-related queries, always ensure both income and expense perspectives are included.
    12. If you are not sure about the correct mapping, or the query terms do not match any valid hierarchy values, return an empty JSON object: {{}}.
    13. If the query does not clearly map to a valid banking term in the taxonomy, return {{}}.
    
    **LOAN RULES**
    1. Always classify at the DEEPEST hierarchy level explicitly mentioned in the query.
    - Priority: level_6 → level_5 → level_4 → level_3.

    2. If a product exists in level_6: 
    - Classify it ONLY at level_6.
    - Do NOT include the same product at level_5 or higher.
    - Example: "Hire Purchase Loan" → level_6 only, its level_5 should be "Asset Finance".
    - Example: "Long Term Loans" → level_6 only, its level_5 should be "Term Loan".... like this do it for all the products
    - If a product exists in level_6, classify it only at level_6, and its level_5 should be the parent category, not a duplicate of the same product.
    
    3. If a product exists in level_5 but not in level_6:
    - Classify it at level_5.
    - Example: "Term Loan" → level_5 only.

    4. Never auto-expand between levels.
    - Do NOT infer related products at deeper levels unless the query explicitly mentions them.
    - Example: "Term Loan" ≠ "Long Term Loan". Level_5 should be "Term Loan". Level_6 should be "Long Term Loans". Do the same for "Short Term Loans" and "Mid Term Loan"
    Remember this hierarchy structure for term loan:
    level_1: [Assets]
    level_2: [Funded Assets]
    level_3: [Loans & Advances]
    level_4: [Loans]
    level_5: [Term Loan]
    level_6: [Long Term Loans, Mid Term Loan, Short Term Loans, Medium Term Loans MMK]
    level_7: [Normal, NPA]
    level_8: [LCY, FCY]
    - If the query mentions "total term loan", "outstanding term loan", "term loan" stop at level_5.
    5. Remove duplicates.
    - Once a product is classified at its deepest valid level, higher-level matches must be dropped.

    6. ✅ Always drop higher-level duplicates when a deeper level match is found.

    7. Advances are restricted.
    - "Advances" category includes only Overdraft and Credit Cards.
    - Do NOT classify general loans as "Advances".

    8. Normalization rule.
    - If the query uses plural terms (e.g., "Loans"), normalize to singular form before classification.

    **CRITICAL MAPPINGS - HIGHEST PRIORITY**:
    - "interest income": Assets → Funded Assets
    - "interest expense": Liabilities → Funded Liability
    NOTE : "interest expense" must be mapped to Level_1: Liabilities, Level_2: ["Funded Liability"]. Do not map it to Expense-> Interest Expense
    - "NII"/"net interest income": Assets, Liabilities -> Funded Assets, Funded Liability
    - "NIE"/"net interest expense" must always mapped to Assets, Liabilities -> Funded Assets, Funded Liability
    NOTE : interest expense and net interest expense are different
    - "NIM"/"net interest margin": Assets, Liabilities -> Funded Assets, Funded Liability
    - "NFI"/"non funded income": Income -> Non Funded Income.
    - "total revenue": Interest Income + Non- Funded Income + Other Income
         Level_1:["Income","Assets"]
         Level_2:["Funded Assets","Non Funded Income","Other Income"]
    - "total expense" / "expense" : 
         Level_1:["Expenses","Liabilities"]
         Level_2:["Funded Liability", "Expenses","Other Expense","Interest Expense","Staff Expense","Premises Expense"]
    - "total income" / "income" : 
         Level_1:["Income","Assets"]
         Level_2:["Funded Assets","Non Funded Income","Other Income","Treasury Interest Income"]
    - "expense":
         Level_1:["Expenses","Liabilities"]
         Level_2:["Funded Liability", "Expenses","Other Expense","Interest Expense","Staff Expense","Premises Expense"]
    NOTE : Expense must follow the above rules, never map to level_1: Expenses, level_2: ["Expenses"]. It must map to level_1 : ["Expenses","Liabilities"] and level_2 : ["Funded Liability","Expenses","Other Expense","Interest Expense","Staff Expense","Premises Expense"]
    - "income":
         Level_1:["Income","Assets"]
         Level_2:["Funded Assets","Non Funded Income","Other Income","Treasury Interest Income"]
    - "net income"/"net profit": Liabilities → Funded Liability -> Deposits from Banks, Customer Deposits.
    - "interest receivable": Assets → Other Assets → Interest Receivable.
    - "total deposits" / "deposit": include both Customer Deposits + Deposits from Banks.
    - "total assets": level_1=["Assets"].
    - "loans": Loans & Advances → Loans. (include all loan types)
    - "advances"/"overdrafts": Loans & Advances → Advances. (include all advance types)
    - "tax": Level-1 Expenses → Level_2: Expenses → Level_3: Tax
    - for the query about both "loans and advances": Loans & Advances → [Loans, Advances]. // RETURN AS ARRAY to include both
    - "placements": Assets → Funded Assets
    - "account with local banks": Assets → Cash and Cash Equivalents → Due From Banks -> Due From Banks
    - Always capture important banking terms (eg., loan, deposit, income, advances, placement, etc.).
    - ✅ Always drop higher-level duplicates when a deeper level match is found.
    **STRICT HIERARCHY RULES**
    1. Always map each value to its correct level as defined in the official hierarchy (do NOT move values up or down).  
    - Example: A value defined under level_6 must never appear in level_5 or level_7.  
    - Example: A value defined under level_3 must never appear in level_2.  
    **EXCLUSION RULES**
    1. If the query mentions "excluding X" or "except X", "other than X", **remove X from all hierarchy levels** before returning the JSON.  
    - Example: "excluding Asset Finance" → drop "Asset Finance" from level_5.  
    - Example: "except NFI" → drop "NFI" from any level it appears.  
    2. Apply exclusions **after mapping all possible values** but before generating the final JSON.  
    3. Only exclude exact matches. Do not remove parent categories unless explicitly mentioned.  
    
    OUTPUT FORMAT:
    {{
    "<banking_term>": {{
        "level_1": [...],
        "level_2": [...],
        "level_3": [...],
        "level_4": [...],
        "level_5": [...],
        "level_6": [...],
        "level_7": [...],
        "level_8": [...]
    }},
    "<another_term>": {{
        ...
    }}
    }}

    OUTPUT (ONLY JSON, NO TEXT):
    """.strip()


    # print(prompt)
    try:
        # Prepare payload for vLLM chat completions API
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", 
                 "content": prompt
                },
                {"role": "user", 
                 "content": query
                }
            ],
            "max_tokens": 2000,  
            "temperature": 0.0,
            "top_p": 0.9,       
            "stream": False    
        }

        print("🚀 Sending request to vLLM API...")
        response = requests.post(VLLM_API_URL, json=payload, timeout=300)  
        response.raise_for_status()
        result = response.json()

        # Extract the content from the response
        reply = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

        if not reply:
            logger.error("Empty response from model")
            logger.debug(f"Full API response: {json.dumps(result, indent=2)}")
            return {}

        # Try to extract JSON from the response (handle markdown code blocks)
        if '```json' in reply:
            reply = reply.split('```json')[1].split('```')[0].strip()
        elif '```' in reply:
            reply = reply.split('```')[1].strip()
            if reply.startswith('json\n'):
                reply = reply[5:]
        


        try:
            # Parse the JSON response
            parsed = json.loads(reply)
            logger.info("Hierarchy fetched by LLM")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {}

    except Exception as e:
        logger.exception(f"Request to vLLM API failed: {e}")
        return {}

# --------------------------
# 7. Fetch Results â€” FIXED: UPPERCASE, UNQUOTED
# --------------------------
def fetch_results(validated: dict, engine) -> pd.DataFrame:
    all_dfs = []
    with engine.connect() as conn:
        for product, levels in validated.items():
            clauses = []
            params = {}
            base = "SELECT * FROM AI_MRL_EXPLANATION WHERE 1=1"
            
            for level, values in levels.items():
                LEVEL = level.upper()  # ðŸ”¥ UPPERCASE, no quotes
                phs = [f":{level}_{i}" for i in range(len(values))]
                clauses.append(f"{LEVEL} IN ({','.join(phs)})")
                for i, v in enumerate(values):
                    params[f"{level}_{i}"] = v

            sql = f"{base} AND {' AND '.join(clauses)}"
            # logger.debug(f"SQL: {sql}")
            # logger.debug(f"Params: {params}")
            df = pd.read_sql(text(sql), conn, params=params)
            all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --------------------------
# 8. Find Matching MRL Lines
# --------------------------
def find_matching_mrl(engine, data):
    """
    Find MRL lines matching the given hierarchy levels.
    
    Args:
        engine: SQLAlchemy engine
        data: Dictionary containing the hierarchy levels to filter by, or directly the levels dictionary
              Format 1: {'product_type': {'level_1': [...], 'level_2': [...]}}
              Format 2: {'level_1': [...], 'level_2': [...]}
              
    Returns:
        Dict[str, DataFrame]: mapping of product_type to matching MRL lines.
        For Format 2 (direct levels), the key will be 'direct_query'.
    """
    try:
        # Create a connection from the engine
        with engine.connect() as conn:
            # Read the MRL data from the database
            query = "SELECT * FROM AI_MRL_EXPLANATION"
            df = pd.read_sql(query, conn)

            results_by_product = {}
            
            # Check if data is already in the levels format (Format 2)
            if all(level.startswith('level_') for level in data.keys()):
                levels_list = [('direct_query', data)]
            else:
                levels_list = data.items()
            
            for product_type, levels in levels_list:
                if not isinstance(levels, dict):
                    logger.debug(f"Unexpected format for levels: {type(levels)}")
                    continue
                
                # Start with all rows
                mask = pd.Series(True, index=df.index)
                
                # Apply filters for each level
                for level, values in levels.items():
                    if not values:  # Skip empty lists
                        continue
                        
                    # Get the actual column name (case-insensitive)
                    level_col = None
                    for col in df.columns:
                        if col.lower() == level.lower():
                            level_col = col
                            break
                    
                    if level_col is None:
                        logger.debug(f"Column {level} not found in the database")
                        continue
                    
                    # Apply the filter
                    # normalized_col = df[level_col].astype(str).str.strip().str.upper()
                    # normalized_vals = [str(v).strip().upper() for v in values]
                    # Managing the Extra spaces in the values
                    normalized_col = df[level_col].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip().upper())
                    normalized_vals = [re.sub(r"\s+", " ", str(v)).strip().upper() for v in values]
                    level_mask = normalized_col.isin(normalized_vals)
                    logger.debug(f"{level_col}: {level_mask.sum()} rows match values {values}")
                    mask = mask & level_mask
                
                # Apply all filters
                filtered = df[mask].copy()
                
                if not filtered.empty:
                    # Matching lines for this product
                    results_by_product[product_type] = filtered
                else:
                    logger.info("No matching MRL lines found")
                    
                    # Show available values for debugging
                    logger.debug("Available values:")
                    for level in levels.keys():
                        level_col = None
                        for col in df.columns:
                            if col.lower() == level.lower():
                                level_col = col
                                break
                        
                        if level_col is not None and level_col in df.columns:
                            unique_vals = df[level_col].dropna().unique()
                            logger.debug(f"{level_col} has {len(unique_vals)} unique values")
                            if len(unique_vals) <= 10:
                                logger.debug(f"Values: {sorted(unique_vals.tolist())}")
                            else:
                                logger.debug(f"Sample: {sorted(unique_vals.tolist())[:5]}...")
            
            # Return per-product dataframes (no merge) to preserve separation
            if results_by_product:
                return results_by_product
                
    except Exception as e:
        logger.error(f"Error in find_matching_mrl: {str(e)}")
        traceback.print_exc()
    
    return {}  # Return empty dict if no matches or error


# --------------------------
# 8. Main Function
# --------------------------

def fallback_to_rag(query: str) -> dict:
    """
    Fallback to RAG when direct hierarchy matching fails.
    """
    logger.info("Fallback to RAG")
    
    try:
        # Initialize RAG chatbot
        rag_chatbot = MRLChatbot()
        
        # Get similar results from RAG
        rag_results = rag_chatbot.process_query(query)
        
        if 'error' in rag_results or not rag_results.get('hierarchy_data'):
            logger.info("No relevant matches found in RAG system")
            return {}
        
        # Get the top 10 results from RAG
        top_results = rag_results['hierarchy_data'][:10]
        top_details = rag_results.get('details', [])
        similarity_scores = rag_results.get('similarity_scores', [])

        # Prepare structured payload including query, hierarchy data, and similarity score
        structured_results = []
        for idx, hierarchy_json in enumerate(top_results):
            detail = top_details[idx] if idx < len(top_details) else {}
            score = similarity_scores[idx] if idx < len(similarity_scores) else None
            structured_results.append({
                "rag_query": detail.get('query'),
                "hierarchy_data": hierarchy_json,
                "similarity_score": score,
                # "document_id": detail.get('_doc_id')
            })

        # Create a prompt for the LLM to analyze the RAG results
        prompt = f"""
            You are a banking data expert. Analyze the query and the top RAG results to return the most accurate hierarchy.

            QUERY: {query}

            RAG RESULTS (ranked by relevance):
            {json.dumps(structured_results, indent=2)}

            INSTRUCTIONS:
            1. Use only values that appear in the hierarchy of the RAG results (no invention).
            2. Higher-ranked results are more relevant, but you may combine consistent elements across multiple.
            3. Levels must reflect hierarchy without duplication:
            - Parent terms at higher levels only, child terms at lower levels only.
            4. Currency rules:
            - If query mentions LCY â†’ set level_8 = ["LCY"].
            - If query mentions FCY â†’ set level_8 = ["FCY"].
            - If hierarchy already includes â€œ(...LCY/FCY)â€, keep it at that level AND also set level_8 accordingly.
            5. If query mentions "Normal" or "NPA", add level_7 with the respective value.
            6. Do not include empty arrays; omit levels with no values.

            OUTPUT: Return ONLY a valid JSON object in this format:
            {{
            "<banking_term>": {{
                "level_1": [...],
                "level_2": [...],
                "level_3": [...],
                "level_4": [...],
                "level_5": [...],
                "level_6": [...],
                "level_7": [...],
                "level_8": [...]
            }}
            }}
            """.strip()

        # Call the LLM to analyze the RAG result
        # Analyze RAG results with LLM
        response = requests.post(
            VLLM_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a banking data expert. Analyze the RAG results and provide the most accurate hierarchy data."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        analyzed_hierarchy = extract_json_object(content)

        if analyzed_hierarchy is None:
            preview = content.strip().replace("\n", " ")[:500]
            logger.error(f"Unable to parse JSON from LLM response. Preview: {preview}")
            return {}
        return analyzed_hierarchy
        
    except Exception as e:
        logger.error(f"Error in RAG fallback: {e}")
        traceback.print_exc()
        return {}

def query_to_data(query: str, valid_values):
    """
    Process a natural language query and return matching MRL data with RAG fallback.
    Args:
        query: Natural language query string
        valid_values: Dictionary of valid values for each hierarchy level
    Returns:
        List containing values from the 'mrl_line' column of matching records.
    """
    engine = get_engine()
    
    # Show available values for reference
    # logger.info("ðŸ“‹ Available values in database (first few items shown):")
    for level in sorted(valid_values.keys()):
        vals = valid_values[level]
        # logger.info(f"  {level}: {vals[:3]}{'...' if len(vals) > 3 else ''} ({len(vals)} total)")
    
    # Step 1: Get the hierarchy from the query using LLM
    logger.info("Processing query with LLM...")
    raw_hierarchy = get_hierarchy_from_query(query)
    logger.info(raw_hierarchy)
    if not raw_hierarchy:
        logger.error("No valid hierarchy data extracted from query")
        raw_hierarchy = fallback_to_rag(query)
        logger.info("Successfully extracted hierarchy from RAG")
        logger.info(raw_hierarchy)
    
    # print("\nðŸ”„ Validating extracted hierarchy...")
    # # Step 2: Validate the hierarchy against database values
    # validated = validate_hierarchy(raw_hierarchy, valid_values)
    
    # if not validated:
    #     print("âŒ No valid hierarchy after validation")
    #     validated = await fallback_to_rag(query)
    

    # Searching for matching MRL lines
    # Step 3: Fetch matching records (now returns dict of DataFrames by product)
    results_by_product = find_matching_mrl(engine, raw_hierarchy)

    # Helper to convert results dict to {product: [mrl_line, ...]}
    def to_output_dict(results_dict):
        output = {}
        for product, df in results_dict.items():
            try:
                lines = df['mrl_line'].tolist() if 'mrl_line' in df.columns else []
            except Exception:
                lines = []
            output[product] = lines
        return output

    # If no results, try RAG fallback
    if not results_by_product:
        logger.error("No matching records found in database")
        raw_hierarchy = fallback_to_rag(query)
        logger.info(raw_hierarchy)
        results_by_product = find_matching_mrl(engine, raw_hierarchy)
        if not results_by_product:
            logger.error("No matching records found in database after RAG fallback")
            return {}
        # Summaries per product
        summary_counts = {k: len(v) for k, v in results_by_product.items()}
        logger.info(f"Retrieved rows by product: {summary_counts}")
        output = to_output_dict(results_by_product)
        # If direct_query only, return list (backward compatible); else return dict per product
        if list(output.keys()) == ['direct_query']:
            return output['direct_query']
        return output

    # We have results; return dict per product
    summary_counts = {k: len(v) for k, v in results_by_product.items()}
    logger.info(f"Retrieved rows by product: {summary_counts}")
    output = to_output_dict(results_by_product)
    # If direct_query only, return list (backward compatible); else return dict per product
    if list(output.keys()) == ['direct_query']:
        return output['direct_query']
    return output  # Return mapping: {product: [mrl_lines]}



    
