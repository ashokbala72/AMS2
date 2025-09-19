# Agentic AMS Final Dashboard — Azure OpenAI Integrated

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
import os
import textwrap
from dotenv import load_dotenv
from integration_loader import get_ticket_data
from openai import AzureOpenAI

def sanitize_text(text: str) -> str:
    """Remove characters not supported by FPDF (latin-1 only)."""
    return text.encode("ascii", "ignore").decode()


# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Agentic AMS Ticket Flow", layout="wide")

# ---------------- Azure OpenAI Setup ----------------
load_dotenv()

def get_secret(name):
    return st.secrets.get(name) or os.getenv(name)

client = AzureOpenAI(
    api_key=get_secret("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_NAME = "gpt-4o-raj"  # Chat / Completions
EMBEDDING_DEPLOYMENT_NAME = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or "text-embedding-3-small"

# ---------------- Active Tab Management ----------------
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

# ✅ Auto-refresh only on Tab 0
if tab_index == 0:
    st_autorefresh(interval=5000, key="summary_refresh")
    st.session_state.ticket_context_df = get_ticket_data()

# ---------------- Safe Ticket Loader ----------------
def safe_ticket_load():
    df = get_ticket_data()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=['Ticket ID', 'Issue Summary'])
    if 'Ticket ID' not in df.columns:
        return pd.DataFrame(columns=['Ticket ID', 'Issue Summary'])
    df['Ticket ID'] = df['Ticket ID'].astype(str).str.strip().str.upper()
    return df

# ---------------- FAISS Embeddings Setup ----------------
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document

if "vector_index" not in st.session_state:
    embeddings = AzureOpenAIEmbeddings(
        model=EMBEDDING_DEPLOYMENT_NAME,
        openai_api_key=get_secret("AZURE_OPENAI_API_KEY"),
        openai_api_version="2024-12-01-preview",
        azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT")
    )
    dummy_doc = [Document(page_content="Placeholder", metadata={"source": "init"})]
    st.session_state.vector_index = FAISS.from_documents(dummy_doc, embeddings)

# ---------------- Session State Initialization ----------------
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
            st.session_state.ticket_context_df["Ticket ID"] = (
                st.session_state.ticket_context_df["Ticket ID"]
                .astype(str).str.strip().str.upper()
            )
        except Exception:
            st.session_state.ticket_context_df = pd.DataFrame()

def add_new_tickets():
    if "ticket_context_df" not in st.session_state:
        return

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

# ---------------- Tabs Setup ----------------
tabs = st.tabs([
    "Ticket Status Summary", "5 Whys Root Cause Analysis", "Fix Suggestion",
    "8D Final Report", "Learning Agent", "Audit Agent",
    "KB Article Generator", "RCA Validator", "Incident Mapper",
    "Business Impact Estimator"
])

# ------------------------ TAB 0 ------------------------
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
            match_row = st.session_state.ticket_context_df[
                st.session_state.ticket_context_df["Ticket ID"] == ticket_id
            ]
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
                    if agent == "8D Agent":
                        # ✅ Force green if step = 6 (8D Complete)
                        emoji = "🟢" if ticket["step"] == 6 else "⚫"
                    elif agent == "KB Agent":
                        emoji = "🟢" if ticket.get("color") == "green" else "⚫"
                    else:
                        emoji = "⚫"
                    cell += emoji + "<br>"
            row[agent] = cell
        triage_data.append(row)

    df_triage = pd.DataFrame(triage_data)

    st.markdown("""
        <style>
        .scroll-wrapper { max-height: 550px; overflow-y: auto; border: 1px solid #ccc; padding: 5px; }
        .ticket-table { border-collapse: collapse; width: 100%; font-size: 14px; }
        .ticket-table th, .ticket-table td {
            border: 1px solid #999; padding: 6px; text-align: center; vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="scroll-wrapper">' +
        df_triage.to_html(escape=False, index=False, classes="ticket-table") +
        '</div>', unsafe_allow_html=True
    )
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
        selected = st.selectbox("Select Ticket", [t["id"] for t in analyzed],
                                key="why_select",
                                index=[t["id"] for t in analyzed].index(st.session_state.why_selected_id))
        st.session_state.why_selected_id = selected
        ticket = next(t for t in analyzed if t["id"] == selected)
        context = st.session_state.ticket_context_df[
            st.session_state.ticket_context_df["Ticket ID"] == selected
        ]
        desc = context.iloc[0].get("Issue Summary", ticket["desc"]) if not context.empty else ticket["desc"]
        st.markdown(f"**Issue Summary:** {desc}")

        if selected not in st.session_state.why_map:
            with st.spinner("Running 5 Whys Analysis..."):
                prompt = f"You are a SAP root cause expert. Provide 5 Why analysis for Ticket ID {selected}: {desc}"
                try:
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    output = response.choices[0].message.content.strip()
                    st.session_state.why_map[selected] = output
                    # last non-empty line as root cause
                    for line in reversed(output.splitlines()):
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
        selected = st.selectbox("Select Ticket", [t["id"] for t in ready],
                                key="fix_select",
                                index=[t["id"] for t in ready].index(st.session_state.fix_selected_id))
        st.session_state.fix_selected_id = selected
        root = st.session_state.eight_d_map.get(selected, "Root cause not found.")
        if selected not in st.session_state.fix_map:
            with st.spinner("Fetching fix recommendation..."):
                try:
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
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
        selected = st.selectbox("Select Ticket", [t["id"] for t in completed],
                                key="8d_select",
                                index=[t["id"] for t in completed].index(st.session_state.eight_d_selected_id))
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
        keyword_counts, cause_counts = {}, {}
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
    if "parsed_log_map" not in st.session_state: st.session_state.parsed_log_map = {}
    if "vector_store" not in st.session_state: st.session_state.vector_store = {}

    now = time.time()
    for t in st.session_state.tickets:
        ticket_id = t["id"]
        elapsed = now - t["start_time"]

        if elapsed > 5:
            if t["step"] == 0: t["step"] = 1
            elif t["step"] == 1:
                if int(ticket_id.split("-")[1]) % 2 == 0:
                    t["step"] = 2; t["color"] = "green"
                else:
                    t["step"] = 3; t["color"] = "grey"
            elif t["step"] == 3:
                t["step"] = 4
                try:
                    row = st.session_state.ticket_context_df[st.session_state.ticket_context_df["Ticket ID"] == ticket_id]
                    log_content = row.iloc[0].get("Log", "No logs found")
                    st.session_state.parsed_log_map[ticket_id] = f"Parsed log insight: {log_content[:200]}..."
                except Exception as e:
                    st.session_state.parsed_log_map[ticket_id] = f"Error parsing log: {e}"
            elif t["step"] == 4:
                t["step"] = 5
                try:
                    issue_desc = t["desc"]
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=[{"role": "user", "content": f"Perform 5 Whys root cause analysis for: {issue_desc}"}]
                    )
                    output = response.choices[0].message.content.strip()
                    st.session_state.why_map[ticket_id] = output
                    st.session_state.eight_d_map[ticket_id] = output.strip().split("\n")[-1]
                except Exception as e:
                    st.session_state.why_map[ticket_id] = f"GPT error: {e}"
                    st.session_state.eight_d_map[ticket_id] = "Unknown root cause"
            elif t["step"] == 5:
                t["step"] = 6
                try:
                    row = st.session_state.ticket_context_df[st.session_state.ticket_context_df["Ticket ID"] == ticket_id]
                    context = "\n".join([f"{col}: {row.iloc[0][col]}" for col in row.columns if pd.notnull(row.iloc[0][col])])
                    rca = st.session_state.eight_d_map.get(ticket_id, "RCA not available")
                    fix_prompt = f"Given root cause {rca}, suggest a fix.\n\nContext:\n{context}"
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
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
    st.dataframe(pd.DataFrame([{
        "Ticket": t["id"],
        "Stage": STATUS_FLOW[t["step"]][1],
        "Agent": STATUS_FLOW[t["step"]][0],
        "Color": t["color"],
        "Seconds in Stage": 0 if t["step"] == 6 else round(now - t["start_time"], 1)
    } for t in st.session_state.tickets]))


# ------------------------ TAB 6 ------------------------
# (KB Article Generator – same logic, just model=DEPLOYMENT_NAME)

# ------------------------ TAB 7 ------------------------
# (RCA Validator – same logic, just model=DEPLOYMENT_NAME)

# ------------------------ TAB 8 ------------------------
# (Incident Mapper – no GPT calls, just TF-IDF similarity)

# ------------------------ TAB 9 ------------------------
# (Business Impact Estimator – same logic, just model=DEPLOYMENT_NAME)
# ------------------------ TAB 6 ------------------------
with tabs[6]:
    st.header("📝 KB Article Generator")
    from fpdf import FPDF
    import tempfile

    kb_ready = [t for t in st.session_state.tickets if t["step"] >= 4]
    if not kb_ready:
        st.info("No tickets available for KB article generation.")
    else:
        generated_this_round = False
        for t in kb_ready:
            ticket_id = t["id"]
            issue_summary = st.session_state.ticket_context_df.loc[
                st.session_state.ticket_context_df["Ticket ID"] == ticket_id, "Issue Summary"
            ].values[0] if ticket_id in st.session_state.ticket_context_df["Ticket ID"].values else "Summary missing"

            root = st.session_state.eight_d_map.get(ticket_id, "Root cause unavailable")
            fix = st.session_state.fix_map.get(ticket_id, "Fix unavailable")
            kb_key = f"kb_{ticket_id}"

            st.markdown(f"**Ticket ID: {ticket_id} | Summary: {issue_summary}**")

            if kb_key not in st.session_state and not generated_this_round:
                with st.spinner(f"Generating KB for {ticket_id}..."):
                    try:
                        prompt = f"""Ticket ID: {ticket_id}
Issue Summary: {issue_summary}
Root Cause: {root}
Fix: {fix}
Preventative Measures: What can avoid this in the future?

Please write a SAP-style KB article."""
                        response = client.chat.completions.create(
                            model=DEPLOYMENT_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500
                        )
                        article = response.choices[0].message.content.strip()
                        st.session_state[kb_key] = article

                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, sanitize_text(article))

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            pdf.output(tmp_file.name)
                            pdf_path = tmp_file.name
                            st.session_state[f"{kb_key}_pdf"] = pdf_path

                        generated_this_round = True
                        st.success("✅ KB Article Generated")

                        with open(pdf_path, "rb") as f:
                            st.download_button("📎 Download PDF", f, file_name=f"KB_{ticket_id}.pdf", key=f"dl_{ticket_id}")
                    except Exception as e:
                        st.session_state[kb_key] = f"Error: {e}"
                        st.error(f"❌ Failed to generate: {e}")
            elif kb_key in st.session_state:
                if not st.session_state[kb_key].startswith("Error"):
                    st.success("✅ KB Article Generated")
                    pdf_file_path = st.session_state.get(f"{kb_key}_pdf", "")
                    if os.path.exists(pdf_file_path):
                        with open(pdf_file_path, "rb") as f:
                            st.download_button("📎 Download PDF", f, file_name=f"KB_{ticket_id}.pdf", key=f"dl_{ticket_id}")
                    else:
                        st.warning("⚠️ PDF not found for this ticket.")
                else:
                    st.error("❌ KB Article could not be generated.")
            else:
                st.info("⏳ KB Article will be generated in next refresh.")
            st.markdown("---")


# ------------------------ TAB 7 ------------------------
with tabs[7]:
    st.header("🧪 RCA Validator")
    from fpdf import FPDF
    import tempfile

    st.info("⏳ Waiting for RCA validation to start...")
    validated_this_round = False

    for t in [t for t in st.session_state.tickets if t["step"] >= 4]:
        ticket_id = t["id"]
        rca_key = f"rca_{ticket_id}"
        issue_summary = st.session_state.ticket_context_df.loc[
            st.session_state.ticket_context_df["Ticket ID"] == ticket_id, "Issue Summary"
        ].values[0] if ticket_id in st.session_state.ticket_context_df["Ticket ID"].values else "Summary missing"

        five_whys = st.session_state.why_map.get(ticket_id, "Missing 5 Whys")
        root = st.session_state.eight_d_map.get(ticket_id, "Missing root cause")

        st.markdown(f"**Ticket ID: {ticket_id} | Summary: {issue_summary}**")

        if rca_key not in st.session_state and not validated_this_round:
            with st.spinner(f"Validating RCA for {ticket_id}..."):
                try:
                    prompt = f"""Evaluate RCA for this SAP AMS issue.

Summary: {issue_summary}
Root Cause: {root}
5 Whys Analysis: {five_whys}

Is the RCA logically sound? Suggest improvements if needed."""
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500
                    )
                    result = response.choices[0].message.content.strip()
                    st.session_state[rca_key] = result

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, sanitize_text(f"""Ticket ID: {ticket_id}

Issue Summary:
{issue_summary}

RCA Validation Result:
{result}"""))


                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        pdf.output(tmp_file.name)
                        pdf_path = tmp_file.name
                        st.session_state[f"{rca_key}_pdf"] = pdf_path

                    validated_this_round = True
                    st.success("✅ RCA Validation Generated")

                    with open(pdf_path, "rb") as f:
                        st.download_button("📎 Download PDF", f, file_name=f"RCA_{ticket_id}.pdf", key=f"dl_rca_{ticket_id}")
                except Exception as e:
                    st.session_state[rca_key] = f"Error: {e}"
                    st.error(f"❌ Failed to validate: {e}")
        elif rca_key in st.session_state:
            if not st.session_state[rca_key].startswith("Error"):
                st.success("✅ RCA Validation Generated")
                pdf_file_path = st.session_state.get(f"{rca_key}_pdf", "")
                if os.path.exists(pdf_file_path):
                    with open(pdf_file_path, "rb") as f:
                        st.download_button("📎 Download PDF", f, file_name=f"RCA_{ticket_id}.pdf", key=f"dl_rca_{ticket_id}")
                else:
                    st.warning("⚠️ PDF not found for this ticket.")
            else:
                st.error("❌ RCA could not be validated.")
        else:
            st.info("⏳ RCA validation will be generated in next refresh.")
        st.markdown("---")


# ------------------------ TAB 8 ------------------------
with tabs[8]:
    st.header("🗺️ Incident Mapper (Fuzzy Grouping Enabled)")

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    root_texts, ticket_ids = [], []
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
    from fpdf import FPDF
    import tempfile

    st.info("⏳ Waiting for Business Impact estimation to start...")
    impact_this_round = False

    for t in [t for t in st.session_state.tickets if t["step"] == 6]:
        ticket_id = t["id"]
        impact_key = f"impact_{ticket_id}"
        row = st.session_state.ticket_context_df[
            st.session_state.ticket_context_df["Ticket ID"] == ticket_id
        ]
        context = row.iloc[0] if not row.empty else {}
        issue = context.get("Issue Summary", "Issue missing")
        business_area = context.get("Affected Business Process", "Area unknown")
        fix = st.session_state.fix_map.get(ticket_id, "Fix not found")

        st.markdown(f"**Ticket ID: {ticket_id} | Summary: {issue}**")

        if impact_key not in st.session_state and not impact_this_round:
            with st.spinner(f"Estimating business impact for {ticket_id}..."):
                try:
                    prompt = f"""You are a SAP AMS expert tasked with estimating the business impact of resolving this issue.

Issue Summary: {issue}
Fix Applied: {fix}
Affected Business Area: {business_area}

Please estimate:
- Time saved
- Effort avoided
- Business risks mitigated

Provide this as a short summary."""
                    response = client.chat.completions.create(
                        model=DEPLOYMENT_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300
                    )
                    impact = response.choices[0].message.content.strip()
                    st.session_state[impact_key] = impact

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, sanitize_text(f"""Ticket ID: {ticket_id}

Business Impact Summary:

{impact}"""))


{impact}""")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        pdf.output(tmp_file.name)
                        pdf_path = tmp_file.name
                        st.session_state[f"{impact_key}_pdf"] = pdf_path

                    impact_this_round = True
                    st.success("✅ Impact Report Generated")

                    with open(pdf_path, "rb") as f:
                        st.download_button("📎 Download PDF", f, file_name=f"Impact_{ticket_id}.pdf", key=f"dl_impact_{ticket_id}")
                except Exception as e:
                    st.session_state[impact_key] = f"Error: {e}"
                    st.error(f"❌ Failed to estimate impact: {e}")
        elif impact_key in st.session_state:
            if not st.session_state[impact_key].startswith("Error"):
                st.success("✅ Impact Report Generated")
                pdf_file_path = st.session_state.get(f"{impact_key}_pdf", "")
                if os.path.exists(pdf_file_path):
                    with open(pdf_file_path, "rb") as f:
                        st.download_button("📎 Download PDF", f, file_name=f"Impact_{ticket_id}.pdf", key=f"dl_impact_{ticket_id}")
                else:
                    st.warning("⚠️ PDF not found for this ticket.")
            else:
                st.error("❌ Impact could not be estimated.")
        else:
            st.info("⏳ Business impact will be generated in next refresh.")
        st.markdown("---")

