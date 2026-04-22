import streamlit as st
import re, io, subprocess, nltk, spacy
from collections import Counter
from pypdf import PdfReader

st.set_page_config(
    page_title="CV Preprocessing",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_nlp():
    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    subprocess.run(
        ["python", "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True
    )
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

SKILLS = {
    "Data Science": {
        "Programming":    ["python","java","sql","r","bash","scala"],
        "ML / DL":        ["tensorflow","pytorch","scikit","keras","huggingface","xgboost"],
        "NLP":            ["nltk","spacy","bert","gpt","transformer","word2vec"],
        "Cloud":          ["aws","gcp","azure","sagemaker","databricks"],
        "Tools":          ["docker","git","jupyter","pandas","numpy","spark"],
    },
    "Marketing": {
        "Paid Ads":       ["google","meta","tiktok","snapchat","youtube","display"],
        "Social Media":   ["instagram","linkedin","twitter","facebook","hootsuite"],
        "Content":        ["canva","photoshop","premiere","capcut","copywriting"],
        "Analytics":      ["analytics","semrush","ga4","tableau","looker"],
        "Email / CRM":    ["mailchimp","hubspot","salesforce","klaviyo"],
        "SEO / SEM":      ["seo","sem","keyword","organic"],
    },
    "Finance": {
        "Analysis":       ["excel","bloomberg","valuation","modelling","forecasting"],
        "Accounting":     ["ifrs","gaap","audit","reconciliation","taxation"],
        "Tools":          ["sap","oracle","quickbooks","tableau","powerbi","vba"],
    },
    "Engineering": {
        "Languages":      ["python","java","cpp","javascript","typescript","go"],
        "Web":            ["react","angular","nodejs","django","flask","fastapi"],
        "DevOps":         ["docker","kubernetes","jenkins","terraform","cicd"],
        "Databases":      ["mysql","postgresql","mongodb","redis","elasticsearch"],
    },
    "General": {
        "Soft Skills":    ["leadership","management","communication","teamwork","strategy"],
        "Languages":      ["arabic","english","french","hindi","spanish"],
        "Office Tools":   ["excel","powerpoint","word","slack","trello","jira"],
    },
}

with st.sidebar:
    st.header("Settings")
    domain = st.selectbox("CV domain", list(SKILLS.keys()))
    st.divider()
    st.markdown("**Toggle steps on / off**")
    do_urls      = st.checkbox("Remove URLs",         value=True)
    do_handles   = st.checkbox("Remove @handles",     value=True)
    do_contacts  = st.checkbox("Remove contacts",     value=True)
    do_punct     = st.checkbox("Remove punctuation",  value=True)
    do_stopwords = st.checkbox("Remove stopwords",    value=True)
    do_short     = st.checkbox("Remove short tokens", value=True)
    do_lemma     = st.checkbox("Lemmatise",           value=True)
    st.divider()
    st.caption("Upload any CV PDF — the app adapts to your domain selection.")

st.title("CV Text Preprocessing")
st.caption("Upload any CV PDF and watch every preprocessing step in real time.")

uploaded = st.file_uploader("Upload a CV PDF", type="pdf")

if not uploaded:
    st.info("Upload fatima_cv.pdf, omar_cv.pdf, or any CV to get started.")
    st.stop()

reader = PdfReader(io.BytesIO(uploaded.read()))
raw = ""
for page in reader.pages:
    raw += page.extract_text() or ""

st.divider()
st.subheader("Preprocessing pipeline")

with st.expander("Step 0 — Raw text extracted from PDF", expanded=True):
    c1, c2 = st.columns([3, 1])
    c1.text_area("", raw, height=220, label_visibility="collapsed")
    c2.metric("Characters", f"{len(raw):,}")
    c2.metric("Words (approx)", f"{len(raw.split()):,}")
    st.info("This is what pypdf extracts — notice URLs, symbols, contact details, and spacing noise.")

text = raw

with st.expander("Step 1 — Lowercase"):
    text = text.lower()
    c1, c2 = st.columns([3, 1])
    c1.text_area("", text[:700], height=150, label_visibility="collapsed")
    c2.metric("Characters", f"{len(text):,}")
    st.info("PYTHON, Python, python — all the same token now.")

if do_urls:
    with st.expander("Step 2 — Remove URLs"):
        found = re.findall(r"https?://\S+|www\.\S+", text)
        text  = re.sub(r"https?://\S+|www\.\S+", "", text)
        c1, c2 = st.columns([3, 1])
        c1.text_area("", text[:700], height=150, label_visibility="collapsed")
        c2.metric("URLs removed", len(found))
        if found: c2.caption("\n".join(found[:4]))
        st.info("LinkedIn, GitHub, portfolio links removed — no skill signal.")

if do_handles:
    with st.expander("Step 3 — Remove @handles"):
        found = re.findall(r"@\S+", text)
        text  = re.sub(r"@\S+", "", text)
        c1, c2 = st.columns([3, 1])
        c1.text_area("", text[:700], height=150, label_visibility="collapsed")
        c2.metric("Handles removed", len(found))
        if found: c2.caption("  ".join(found[:6]))
        st.info("Social handles are identifiers, not skills.")

if do_contacts:
    with st.expander("Step 4 — Remove email and phone"):
        emails = re.findall(r"\S+@\S+", text)
        text   = re.sub(r"\S+@\S+", "", text)
        text   = re.sub(r"[\+]?[\d\s\-\(\)]{9,}", "", text)
        c1, c2 = st.columns([3, 1])
        c1.text_area("", text[:700], height=150, label_visibility="collapsed")
        c2.metric("Emails found", len(emails))
        st.info("Emails and phones are privacy-sensitive and not useful for skill matching.")

if do_punct:
    with st.expander("Step 5 — Remove punctuation and symbols"):
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        c1, c2 = st.columns([3, 1])
        c1.text_area("", text[:700], height=150, label_visibility="collapsed")
        c2.metric("Characters now", f"{len(text):,}")
        st.info("Bullets, dashes, brackets, %, & removed. Only letters and digits remain.")

with st.expander("Step 6 — Tokenisation"):
    from nltk.tokenize import word_tokenize
    text   = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    c1, c2 = st.columns([3, 1])
    c1.write(tokens[:60])
    c2.metric("Total tokens", len(tokens))
    st.info("The cleaned string is split into individual word units.")

if do_stopwords:
    with st.expander("Step 7 — Remove stopwords"):
        from nltk.corpus import stopwords
        sw         = set(stopwords.words("english"))
        removed_sw = [t for t in tokens if t in sw]
        tokens     = [t for t in tokens if t not in sw]
        c1, c2     = st.columns([3, 1])
        c1.write(tokens[:60])
        c2.metric("Removed",   len(removed_sw))
        c2.metric("Remaining", len(tokens))
        c2.caption("e.g. " + ", ".join(list(set(removed_sw))[:8]))
        st.info("the, and, with, for — too common to distinguish candidates.")

if do_short:
    with st.expander("Step 8 — Remove short tokens"):
        before = len(tokens)
        tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
        c1, c2 = st.columns([3, 1])
        c1.write(tokens[:60])
        c2.metric("Removed",   before - len(tokens))
        c2.metric("Remaining", len(tokens))
        st.info("Single characters and lone digits are leftover noise.")

if do_lemma:
    with st.expander("Step 9 — Lemmatisation"):
        with st.spinner("Lemmatising..."):
            before_s = tokens[5:15]
            doc      = nlp(" ".join(tokens))
            tokens   = [t.lemma_ for t in doc]
            after_s  = tokens[5:15]
        c1, c2 = st.columns(2)
        c1.write("**Before:**"); c1.write(before_s)
        c2.write("**After:**");  c2.write(after_s)
        st.info("built → build, managing → manage, campaigns → campaign.")

st.divider()
st.subheader("Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw characters",  f"{len(raw):,}")
m2.metric("Clean tokens",    len(tokens))
m3.metric("Unique tokens",   len(set(tokens)))
m4.metric("Compression",     f"{100 - int(len(tokens) / max(len(raw.split()), 1) * 100)}%")

tab1, tab2, tab3 = st.tabs(["Clean tokens", "Word frequency", "Skill extraction"])

with tab1:
    st.write(tokens)
    st.download_button("Download tokens as .txt", " ".join(tokens), "preprocessed_cv.txt")

with tab2:
    freq = Counter(tokens)
    top  = freq.most_common(25)
    if top:
        for word, count in top:
            w, b = st.columns([1, 5])
            w.write(f"`{word}`")
            b.progress(count / top[0][1], text=str(count))

with tab3:
    st.caption(f"Showing skills for domain: **{domain}**")
    token_set  = set(tokens)
    skill_dict = SKILLS[domain]
    any_found  = False
    for cat, skill_list in skill_dict.items():
        found = [s for s in skill_list if s in token_set]
        if found:
            any_found = True
            st.markdown(
                f"<div style='background:#EEEDFE;padding:8px 14px;"
                f"border-radius:8px;margin-bottom:6px'>"
                f"<b>{cat}</b> &nbsp;→&nbsp; {', '.join(found)}</div>",
                unsafe_allow_html=True
            )
    if not any_found:
        st.warning("No skills matched. Try switching the domain in the sidebar.")
