
import pandas as pd
import sqlite3

def get_ticket_data():
    try:
        conn = sqlite3.connect("ams_config.db")
        cursor = conn.cursor()
        cursor.execute("SELECT source, enabled FROM config ORDER BY priority ASC")
        sources = cursor.fetchall()
        conn.close()

        for source, enabled in sources:
            if enabled and source.lower() == "servicenow":
                # Simulate failure or implement ServiceNow connector
                raise ConnectionError("Simulated ServiceNow failure")

    except Exception as e:
        print("⚠️ Falling back to CSV. Reason:", e)

    # Fallback to CSV
    try:
        df = pd.read_csv("sap_ticket_combined_allinfo.csv")
        if "Ticket ID" in df.columns:
            return df
        else:
            raise KeyError("Missing 'Ticket ID' in CSV")
    except Exception as e:
        print("❌ CSV fallback also failed:", e)

    return pd.DataFrame()
