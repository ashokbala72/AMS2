# Agentic AMS Final Dashboard — Rebuilt with Full Requirements

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
import openai
import os
import textwrap
from dotenv import load_dotenv
from integration_loader import get_ticket_data

st.set_page_config(page_title="Agentic AMS Ticket Flow", layout="wide")
# Safely determine active tab index
if "active_tab_index" not in st.session_state:
    query_params = st.query_params
    tab_index_raw = query_params.get("tab", [0])[0]
    try:
        tab_index = int(tab_index_raw)
    except (ValueError, TypeError):
        tab_index = 0
    st.session_state["active_tab_index"] = tab_index

tab_index = st.session_state["active_tab_index"]

# ✅ Load ticket data if not already in session
if "ticket_context_df" not in st.session_state:
    st.session_state.ticket_context_df = get_ticket_data()

# ✅ Perform auto-refresh and reload ticket data only on tab 0
if tab_index == 0:
    st_autorefresh(interval=5000, key='auto_summary_tab_refresh')
    st.session_state.ticket_context_df = get_ticket_data()

# Auto-refresh and ticket reload only on Summary tab
def refresh_ticket_data():
    st.session_state.ticket_context_df = get_ticket_data()

if tab_index == 0:
    st_autorefresh(interval=5000, key="summary_refresh")
    refresh_ticket_data()




# ✅ Safe loader to guarantee Ticket ID is always present
def safe_ticket_load():
    df = get_ticket_data()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=['Ticket ID', 'Issue Summary'])
    if 'Ticket ID' not in df.columns:
        print("⚠️ 'Ticket ID' missing. Returning empty DataFrame.")
        return pd.DataFrame(columns=['Ticket ID', 'Issue Summary'])
    df['Ticket ID'] = df['Ticket ID'].astype(str).str.strip().str.upper()
    return df



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()



# ---------------- CHROMA VECTOR DB SETUP ----------------
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import tempfile

if "vector_index" not in st.session_state:
    chroma_dir = tempfile.mkdtemp()
    st.session_state.vector_index = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=chroma_dir)


if "active_tab_index" not in st.session_state:
    query_params = st.query_params
    tab_index_raw = query_params.get("tab", [0])[0]
    try:
        tab_index = int(tab_index_raw)
    except (ValueError, TypeError):
        tab_index = 0
    st.session_state["active_tab_index"] = tab_index

tab_index = st.session_state["active_tab_index"]




if "tickets" not in st.session_state:
    st.session_state.tickets = []
    st.session_state.counter = 0
    st.session_state.why_map = {}
    st.session_state.fix_map = {}
    st.session_state.eight_d_map = {}

def load_ticket_context():
    if "ticket_context_df" not in st.session_state:
        try:
            st.session_state.ticket_context_df = pd.read_csv("sap_ticket_combined_allinfo.csv")
            st.session_state.ticket_context_df["Ticket ID"] = st.session_state.ticket_context_df["Ticket ID"].astype(str).str.strip().str.upper()
        except Exception:
            st.session_state.ticket_context_df = pd.DataFrame()

def add_new_tickets():
    if "ticket_context_df" not in st.session_state:
        return  # no ticket data available

    df = st.session_state.ticket_context_df
    if df.empty or "Ticket ID" not in df.columns:
        return

    existing_ids = set(t["id"] for t in st.session_state.tickets)
    new_ids = df[~df["Ticket ID"].isin(existing_ids)]["Ticket ID"].tolist()
    if not new_ids:
        return

    ticket_id = new_ids[0]
    match = df[df["Ticket ID"] == ticket_id]
    desc = match.iloc[0].get("Issue Description") or match.iloc[0].get("Recent Log Snippet") or "Description missing"

    ticket = {
        "id": ticket_id,
        "desc": desc,
        "step": 0,
        "start_time": time.time(),
        "color": "grey"
    }

    st.session_state.tickets.append(ticket)
    st.session_state.counter += 1
    time.sleep(2)

STATUS_FLOW = [
    ("Triaging Agent", "In Queue"),
    ("Triaging Agent", "Classification"),
    ("KB Agent", "Resolved by KB"),
    ("Analysis Agent", "Parsing logs"),
    ("5 Whys Agent", "5 Whys"),
    ("Fix Agent", "Submitting Fix"),
    ("8D Agent", "8D Complete")
]

# Initialize Tabs
tabs = st.tabs([
    "Ticket Status Summary", "5 Whys Root Cause Analysis", "Fix Suggestion",
    "8D Final Report", "Learning Agent", "Audit Agent",
    "KB Article Generator", "RCA Validator", "Incident Mapper",
    "Business Impact Estimator"
])

# ------------------------ TAB 0 ------------------------
with tabs[0]:
    st.title("📊 Ticket Status Summary")

    agents = ["Triaging Agent", "KB Agent", "Analysis Agent", "5 Whys Agent", "Fix Agent", "8D Agent"]

    # 🔍 Search box
    search_term = st.text_input("Search by Ticket ID or Issue Summary", "").strip().lower()
    filtered_tickets = []
    for ticket in st.session_state.tickets:
        ticket_id = ticket["id"]
        issue_summary = ""
        if "ticket_context_df" in st.session_state and not st.session_state.ticket_context_df.empty:
            match_row = st.session_state.ticket_context_df[st.session_state.ticket_context_df["Ticket ID"] == ticket_id]
            if not match_row.empty:
                issue_summary = match_row.iloc[0].get("Issue Summary", "")[:80]
        if (search_term in ticket_id.lower()) or (search_term in issue_summary.lower()):
            filtered_tickets.append((ticket, issue_summary))

    triage_data = []
    for ticket, summary in filtered_tickets[-50:]:  # show last 50 matching
        ticket_id = ticket['id']
        priority = ["Low", "Medium", "High"][(int(ticket_id.split('-')[1]) % 3)]
        row = {"Ticket": ticket_id, "Issue Summary": summary, "Priority": priority}
        for agent in agents:
            substeps = [i for i, (a, _) in enumerate(STATUS_FLOW) if a == agent]
            cell = ""
            for step in substeps:
                if ticket["step"] == step:
                    if agent in ["KB Agent", "8D Agent"]:
                        emoji = "🟢" if ticket["color"] == "green" else "⚫"
                    else:
                        emoji = "⚫"
                    cell += emoji + "<br>"
            row[agent] = cell
        triage_data.append(row)

    df_triage = pd.DataFrame(triage_data)

    st.markdown("""
        <style>
        .scroll-wrapper {
            max-height: 550px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 5px;
        }
        .ticket-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
        }
        .ticket-table th, .ticket-table td {
            border: 1px solid #999;
            padding: 6px;
            text-align: center;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="scroll-wrapper">' + df_triage.to_html(escape=False, index=False, classes="ticket-table") + '</div>', unsafe_allow_html=True)
    st.info("Status is auto-refreshed every 5 seconds. 🕐")


# ------------------------ TAB 1 ------------------------
with tabs[1]:
    st.header("🔍 5 Whys Root Cause Analysis")
    load_ticket_context()
    analyzed = [t for t in st.session_state.tickets if t["step"] >= 4]
    if not analyzed:
        st.warning("No tickets have reached the 5 Whys stage yet.")
    else:
        if "why_selected_id" not in st.session_state:
            st.session_state.why_selected_id = analyzed[0]["id"]
        selected = st.selectbox("Select Ticket", [t["id"] for t in analyzed], key="why_select", index=[t["id"] for t in analyzed].index(st.session_state.why_selected_id))
        if st.session_state.why_selected_id != selected:
            st.session_state.why_selected_id = selected
        st.session_state.why_selected_id = selected
        ticket = next(t for t in analyzed if t["id"] == selected)
        context = st.session_state.ticket_context_df[st.session_state.ticket_context_df["Ticket ID"] == selected]
        desc = context.iloc[0].get("Issue Summary", ticket["desc"]) if not context.empty else ticket["desc"]
        st.markdown(f"**Issue Summary:** {desc}")

        if selected not in st.session_state.why_map:
            with st.spinner("Running 5 Whys Analysis..."):
                prompt = textwrap.dedent(f"""
                    You are a SAP root cause expert. Provide 5 Why analysis on:
                    Ticket ID: {selected}
                    Issue Summary: {desc}
                """)
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    output = response.choices[0].message.content.strip()
                    st.session_state.why_map[selected] = output
                    lines = output.splitlines()
                    for line in reversed(lines):
                        if line.strip() and not line.lower().startswith("why"):
                            st.session_state.eight_d_map[selected] = line.strip()
                            break
                except Exception as e:
                    st.session_state.why_map[selected] = f"Error: {e}"
                    st.session_state.eight_d_map[selected] = "Unavailable"

        st.subheader("5 Whys Analysis")
        st.markdown(st.session_state.why_map.get(selected, "Not available."))

# ------------------------ TAB 2 ------------------------
with tabs[2]:
    st.header("🛠️ Fix Suggestion")
    ready = [t for t in st.session_state.tickets if t["step"] >= 5]
    if not ready:
        st.info("No tickets ready for fix suggestion.")
    else:
        if "fix_selected_id" not in st.session_state:
            st.session_state.fix_selected_id = ready[0]["id"]
        selected = st.selectbox("Select Ticket", [t["id"] for t in ready], key="fix_select", index=[t["id"] for t in ready].index(st.session_state.fix_selected_id))
        if st.session_state.fix_selected_id != selected:
            st.session_state.fix_selected_id = selected
        st.session_state.fix_selected_id = selected
        root = st.session_state.eight_d_map.get(selected, "Root cause not found.")
        if selected not in st.session_state.fix_map:
            with st.spinner("Fetching fix recommendation..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Suggest a technical fix for: {root}"}],
                        max_tokens=300
                    )
                    st.session_state.fix_map[selected] = response.choices[0].message.content.strip()
                except Exception as e:
                    st.session_state.fix_map[selected] = f"Error: {e}"
        st.subheader("Suggested Fix")
        st.markdown(st.session_state.fix_map[selected])

# ------------------------ TAB 3 ------------------------
with tabs[3]:
    st.header("📄 8D Final Report")
    completed = [t for t in st.session_state.tickets if t["step"] == 6]
    if not completed:
        st.info("No completed tickets.")
    else:
        if "eight_d_selected_id" not in st.session_state:
            st.session_state.eight_d_selected_id = completed[0]["id"]
        selected = st.selectbox("Select Ticket", [t["id"] for t in completed], key="8d_select", index=[t["id"] for t in completed].index(st.session_state.eight_d_selected_id))
        if st.session_state.eight_d_selected_id != selected:
            st.session_state.eight_d_selected_id = selected
        st.session_state.eight_d_selected_id = selected
        summary = st.session_state.ticket_context_df.loc[
            st.session_state.ticket_context_df["Ticket ID"] == selected, "Issue Summary"
        ].values[0] if selected in st.session_state.ticket_context_df["Ticket ID"].values else "Summary not found"
        full_why = st.session_state.why_map.get(selected, "5 Whys not found.")
        root = st.session_state.eight_d_map.get(selected, "Root cause missing")
        fix = st.session_state.fix_map.get(selected, "Fix missing")

        st.markdown(f"""
### 8D Report for {selected}

**Issue Summary:** {summary}

**5 Whys Analysis:**
{full_why}

**Root Cause:**
{root}

**Fix:**
{fix}

**Status:** Closed
        """)

# ------------------------ TAB 4 ------------------------
with tabs[4]:
    st.header("📚 Learning Agent Insights")
    closed = [t for t in st.session_state.tickets if t["step"] == 6]
    if not closed:
        st.info("No completed tickets to analyze.")
    else:
        keyword_counts = {}
        cause_counts = {}
        for t in closed:
            desc = t["desc"].lower()
            for word in ["invoice", "memory", "timeout", "network", "FB60", "F110"]:
                if word in desc:
                    keyword_counts[word] = keyword_counts.get(word, 0) + 1
            cause = st.session_state.eight_d_map.get(t["id"], "").strip()
            if cause:
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
        st.subheader("📌 Frequent SAP Keywords")
        for k, v in sorted(keyword_counts.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{k}**: {v} times")
        st.subheader("📌 Frequent Root Causes")
        for k, v in sorted(cause_counts.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{k}**: {v} times")


# ------------------------ TAB 5 ------------------------

with tabs[5]:
    st.header("📈 Agent Audit and SLA Monitor")

    if "parsed_log_map" not in st.session_state:
        st.session_state.parsed_log_map = {}
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = {}

    now = time.time()
    for t in st.session_state.tickets:
        ticket_id = t["id"]
        elapsed = now - t["start_time"]

        if elapsed > 5:
            if t["step"] == 0:
                t["step"] = 1
            elif t["step"] == 1:
                if int(ticket_id.split("-")[1]) % 2 == 0:
                    t["step"] = 2; t["color"] = "green"
                else:
                    t["step"] = 3; t["color"] = "grey"

            elif t["step"] == 3:
                t["step"] = 4
                try:
                    row = st.session_state.ticket_context_df[
                        st.session_state.ticket_context_df["Ticket ID"] == ticket_id
                    ]
                    log_content = row.iloc[0].get("Log", "No logs found")
                    parsed = f"Parsed log insight: {log_content[:200]}..."
                    st.session_state.parsed_log_map[ticket_id] = parsed
                except Exception as e:
                    st.session_state.parsed_log_map[ticket_id] = f"Error parsing log: {e}"

            elif t["step"] == 4:
                t["step"] = 5
                try:
                    issue_desc = t["desc"]
                    prompt = f"""Perform a 5 Whys root cause analysis for this SAP ticket.

Issue: {issue_desc}"""
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    output = response.choices[0].message.content.strip()
                    st.session_state.why_map[ticket_id] = output
                    last_line = output.strip().split("\n")[-1]
                    st.session_state.eight_d_map[ticket_id] = last_line
                except Exception as e:
                    st.session_state.why_map[ticket_id] = f"GPT error: {e}"
                    st.session_state.eight_d_map[ticket_id] = "Unknown root cause"

            elif t["step"] == 5:
                t["step"] = 6
                try:
                    row = st.session_state.ticket_context_df[
                        st.session_state.ticket_context_df["Ticket ID"] == ticket_id
                    ]
                    context = "\n".join([
                        f"{col}: {row.iloc[0][col]}" for col in row.columns if pd.notnull(row.iloc[0][col])
                    ])
                    rca = st.session_state.eight_d_map.get(ticket_id, "RCA not available")
                    fix_prompt = f"""Given this root cause, suggest a fix.

Root Cause: {rca}

Context:
{context}"""
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": fix_prompt}]
                    )
                    fix = response.choices[0].message.content.strip()
                    st.session_state.fix_map[ticket_id] = fix
                    st.session_state.vector_store[ticket_id] = fix
                    t["color"] = "green"
                except Exception as e:
                    st.session_state.fix_map[ticket_id] = f"GPT error: {e}"

            t["start_time"] = now

    add_new_tickets()

    audit = []
    for t in st.session_state.tickets:
        audit.append({
            "Ticket": t["id"],
            "Stage": STATUS_FLOW[t["step"]][1],
            "Agent": STATUS_FLOW[t["step"]][0],
            "Color": t["color"],
            "Seconds in Stage": 0 if t["step"] == 6 else round(now - t["start_time"], 1)
        })
    st.dataframe(pd.DataFrame(audit))


# ------------------------ TAB 6 ------------------------
with tabs[6]:
    st.header("📝 KB Article Generator")
    
    kb_ready = [t for t in st.session_state.tickets if t["step"] >= 4]
    if not kb_ready:
        st.info("No tickets available for KB article generation.")
    else:
        ticket_ids = [t["id"] for t in kb_ready]

        if "kb_selected_id" not in st.session_state:
            st.session_state.kb_selected_id = ticket_ids[0]

        selected = st.selectbox(
            "Select Ticket",
            ticket_ids,
            index=ticket_ids.index(st.session_state.kb_selected_id),
            key="kb_selectbox"
        )

        st.session_state.kb_selected_id = selected


        issue_summary = st.session_state.ticket_context_df.loc[
            st.session_state.ticket_context_df["Ticket ID"] == selected, "Issue Summary"
        ].values[0] if selected in st.session_state.ticket_context_df["Ticket ID"].values else "Summary missing"

        root = st.session_state.eight_d_map.get(selected, "Root cause unavailable")
        fix = st.session_state.fix_map.get(selected, "Fix unavailable")

        # 🔄 Auto-generate KB if not already generated
        kb_key = f"kb_{selected}"
        if kb_key not in st.session_state:
            with st.spinner("Auto-generating KB article via GPT..."):
                prompt = f"""
Ticket ID: {selected}
Issue Summary: {issue_summary}
Root Cause: {root}
Fix: {fix}
Preventative Measures: What can avoid this in the future?
Please write a SAP-style KB article.
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    st.session_state[kb_key] = response.choices[0].message.content.strip()
                except Exception as e:
                    st.session_state[kb_key] = f"Error: {e}"

        # 🖨️ Display generated KB article or fallback
        st.subheader("📄 KB Article")
        st.markdown(st.session_state.get(kb_key, "KB article not generated yet for the selected ticket."))


# ------------------------ TAB 7 ------------------------
with tabs[7]:
    st.header("🧪 RCA Validator")

    with_rca = [t for t in st.session_state.tickets if t["id"] in st.session_state.eight_d_map]

    if not with_rca:
        st.info("No RCA to validate.")
    else:
        ticket_ids = [t["id"] for t in with_rca]

        # Ensure rca_selected_id is initialized and persists correctly
        if "rca_selected_id" not in st.session_state:
            st.session_state.rca_selected_id = ticket_ids[0]

        selected = st.selectbox(
            "Select Ticket",
            ticket_ids,
            index=ticket_ids.index(st.session_state.rca_selected_id),
            key="rca_selectbox"
        )

        if selected != st.session_state.rca_selected_id:
            st.session_state.rca_selected_id = selected

        # Safe access using selected ID
        selected_id = st.session_state.rca_selected_id

        issue = st.session_state.ticket_context_df.loc[
            st.session_state.ticket_context_df["Ticket ID"] == selected_id, "Issue Summary"
        ].values[0] if selected_id in st.session_state.ticket_context_df["Ticket ID"].values else "Summary missing"

        five_whys = st.session_state.why_map.get(selected_id, "Missing 5 Whys")
        root = st.session_state.eight_d_map.get(selected_id, "Missing root cause")

        prompt = f"""Evaluate RCA for this SAP AMS issue.

Summary: {issue}
Root Cause: {root}
5 Whys Analysis: {five_whys}

Is the RCA logically sound? Suggest improvements if needed."""

        with st.spinner("Validating RCA with GPT..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300
                )
                result = response.choices[0].message.content.strip()
            except Exception as e:
                result = f"Error: {e}"

        st.subheader("✅ GPT Feedback on RCA")
        st.markdown(result)



# ------------------------ TAB 8 ------------------------
with tabs[8]:
    st.header("🗺️ Incident Mapper (Fuzzy Grouping Enabled)")

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    root_texts = []
    ticket_ids = []

    for t in st.session_state.tickets:
        root = st.session_state.eight_d_map.get(t["id"], "").strip()
        if root:
            root_texts.append(root)
            ticket_ids.append(t["id"])

    if not root_texts:
        st.info("No root causes available for mapping.")
    else:
        vectorizer = TfidfVectorizer().fit_transform(root_texts)
        cosine_sim = cosine_similarity(vectorizer)

        threshold = 0.75
        clustered = {}
        for i, ticket_i in enumerate(ticket_ids):
            for j, ticket_j in enumerate(ticket_ids):
                if i != j and cosine_sim[i, j] >= threshold:
                    key = f"{ticket_i} ~ {ticket_j}"
                    clustered.setdefault(key, []).extend([ticket_i, ticket_j])

        # Remove duplicates
        clusters = {k: list(set(v)) for k, v in clustered.items() if len(set(v)) > 1}

        if not clusters:
            st.info("No similar incidents found.")
        else:
            cluster_names = list(clusters.keys())
            selected_cluster = st.selectbox("Select Similar Incident Cluster", cluster_names)
            st.markdown(f"**Cluster Group:** `{selected_cluster}`")
            st.markdown(f"**Affected Tickets:** {', '.join(clusters[selected_cluster])}")



# ------------------------ TAB 9 ------------------------
with tabs[9]:
    st.header("💼 Business Impact Estimator")

    closed = [t for t in st.session_state.tickets if t["step"] == 6]
    if not closed:
        st.info("No closed tickets available.")
    else:
        if "impact_selected_id" not in st.session_state:
            st.session_state.impact_selected_id = closed[0]["id"]

        ticket_ids = [t["id"] for t in closed]
        selected = st.selectbox(
            "Select Ticket",
            ticket_ids,
            index=ticket_ids.index(st.session_state.impact_selected_id),
            key="impact_ticket_select"
        )

        if selected != st.session_state.impact_selected_id:
            st.session_state.impact_selected_id = selected

        selected_id = st.session_state.impact_selected_id

        # Get ticket context from DataFrame
        row = st.session_state.ticket_context_df[
            st.session_state.ticket_context_df["Ticket ID"] == selected_id
        ]
        context = row.iloc[0] if not row.empty else {}
        issue = context.get("Issue Summary", "Issue missing")
        business_area = context.get("Affected Business Process", "Area unknown")
        fix = st.session_state.fix_map.get(selected_id, "Fix not found")

        # GPT Prompt
        prompt = f"""You are a SAP AMS expert tasked with estimating the business impact of resolving this issue.

Issue Summary: {issue}
Fix Applied: {fix}
Affected Business Area: {business_area}

Please estimate:
- Time saved
- Effort avoided
- Business risks mitigated

Provide this as a short summary."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            impact = response.choices[0].message.content.strip()
        except Exception as e:
            impact = f"Error: {e}"

        st.subheader("📊 Estimated Impact")
        st.markdown(impact)

        # Add PDF export
        from fpdf import FPDF
        if st.button("📥 Download Impact Report (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Ticket: {selected_id}\n\nBusiness Impact Summary:\n\n{impact}")
            pdf_path = f"/tmp/impact_{selected_id}.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("Click to Download", f, file_name=f"Impact_{selected_id}.pdf")

