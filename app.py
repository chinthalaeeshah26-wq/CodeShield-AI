import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Streamlit Cloud Secrets fallback
if not HF_TOKEN:
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"].strip()
    except Exception:
        HF_TOKEN = ""

# Hugging Face Router API
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# Model available through Hugging Face Inference Providers
HF_MODEL = "openai/gpt-oss-120b:fastest"


# ============================================================
# 2. PYDANTIC MODELS
# ============================================================

class SecurityIssue(BaseModel):
    title: str
    severity: str
    line: str
    explanation: str
    impact: str
    vulnerable_code: str
    secure_code: str
    recommendation: str


class SecurityReport(BaseModel):
    security_score: int
    code_explanation: str
    summary: str
    issues: list[SecurityIssue]


# ============================================================
# 3. PARSE AI RESPONSE
# ============================================================

def parse_analysis(text):
    """
    Converts the structured text returned by the AI
    into a SecurityReport.

    The AI is NOT required to return JSON.
    """

    text = text.strip()

    score_match = re.search(
        r"SECURITY_SCORE\s*:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    score = 50

    if score_match:
        score = int(score_match.group(1))

    score = max(0, min(100, score))

    code_explanation = extract_section(
        text,
        "CODE_EXPLANATION",
        "SUMMARY"
    )

    summary = extract_section(
        text,
        "SUMMARY",
        "ISSUE"
    )

    issues = []

    issue_blocks = re.split(
        r"\n\s*ISSUE\s+\d+\s*:\s*",
        text,
        flags=re.IGNORECASE
    )

    for block in issue_blocks[1:]:

        title = extract_field(
            block,
            "TITLE"
        )

        severity = extract_field(
            block,
            "SEVERITY"
        )

        line = extract_field(
            block,
            "LINE"
        )

        explanation = extract_field(
            block,
            "EXPLANATION"
        )

        impact = extract_field(
            block,
            "IMPACT"
        )

        vulnerable_code = extract_field(
            block,
            "VULNERABLE_CODE"
        )

        secure_code = extract_field(
            block,
            "SECURE_CODE"
        )

        recommendation = extract_field(
            block,
            "RECOMMENDATION"
        )

        if title:

            issues.append(
                SecurityIssue(
                    title=title,
                    severity=severity or "Informational",
                    line=line or "Not specified",
                    explanation=explanation,
                    impact=impact,
                    vulnerable_code=vulnerable_code,
                    secure_code=secure_code,
                    recommendation=recommendation
                )
            )

    return SecurityReport(
        security_score=score,
        code_explanation=code_explanation,
        summary=summary,
        issues=issues
    )


# ============================================================
# 4. TEXT PARSING HELPERS
# ============================================================

def extract_field(text, field_name):

    pattern = (
        rf"{re.escape(field_name)}\s*:\s*"
        rf"(.*?)(?=\n[A-Z_]+\s*:|\Z)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return ""

    return match.group(1).strip()


def extract_section(
    text,
    start_name,
    end_name
):

    pattern = (
        rf"{re.escape(start_name)}\s*:\s*"
        rf"(.*?)(?=\n{re.escape(end_name)}\s*:|\Z)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return "Not provided."

    return match.group(1).strip()


# ============================================================
# 5. HUGGING FACE ANALYSIS
# ============================================================

def analyze_code(code, language):

    if not HF_TOKEN:

        raise RuntimeError(
            "Hugging Face API token was not found. "
            "Create a .env file with HF_TOKEN=YOUR_TOKEN "
            "or add HF_TOKEN to Streamlit Secrets."
        )

    prompt = f"""
You are CodeShield AI, a secure source-code analysis assistant.

Analyze the following {language} source code.

Your job is to:

1. Explain what the code does.
2. Identify genuine security vulnerabilities.
3. Give the approximate line number.
4. Give severity.
5. Explain the issue.
6. Explain the security impact.
7. Show the vulnerable code.
8. Show safer replacement code.
9. Explain how to fix it.
10. Give an overall security score from 0 to 100.

Check for:

- SQL Injection
- Command Injection
- Cross-Site Scripting
- Hardcoded passwords
- Hardcoded API keys
- Hardcoded secrets
- Path Traversal
- Unsafe file operations
- Unsafe deserialization
- Weak authentication
- Weak authorization
- Sensitive information exposure
- Improper input validation
- Weak cryptography
- Insecure random number generation
- Other obvious security problems

IMPORTANT:

Only report vulnerabilities supported by the supplied code.

Do not invent vulnerabilities.

If there are no significant security issues,
do not create an ISSUE block.

Return ONLY the following structured format.

SECURITY_SCORE: number from 0 to 100

CODE_EXPLANATION:
Explain the code in simple language.

SUMMARY:
Give an overall security assessment.

ISSUE 1:
TITLE: issue name
SEVERITY: Critical, High, Medium, Low, or Informational
LINE: approximate line number
EXPLANATION: explain the vulnerability
IMPACT: explain the security impact
VULNERABLE_CODE: show the vulnerable code
SECURE_CODE: show safer replacement code
RECOMMENDATION: explain how to fix it

ISSUE 2:
TITLE: issue name
SEVERITY: Critical, High, Medium, Low, or Informational
LINE: approximate line number
EXPLANATION: explain the vulnerability
IMPACT: explain the security impact
VULNERABLE_CODE: show the vulnerable code
SECURE_CODE: show safer replacement code
RECOMMENDATION: explain how to fix it

Only create as many ISSUE blocks as are actually supported by the code.

SOURCE CODE:

---------------- START SOURCE CODE ----------------

{code}

---------------- END SOURCE CODE ----------------
"""

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 3000,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(
            HF_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Hugging Face request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Hugging Face. "
            "Please check your internet connection."
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Hugging Face connection error: {error}"
        )

    if response.status_code != 200:

        try:
            error_data = response.json()

            error_message = error_data.get(
                "error",
                response.text
            )

        except Exception:

            error_message = response.text

        raise RuntimeError(
            f"Hugging Face API error "
            f"({response.status_code}): "
            f"{error_message}"
        )

    try:

        data = response.json()

    except ValueError:

        raise RuntimeError(
            "Hugging Face returned an invalid response."
        )

    try:

        generated_text = (
            data["choices"][0]["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError
    ):

        raise RuntimeError(
            "Hugging Face returned an unexpected response."
        )

    if not generated_text.strip():

        raise RuntimeError(
            "Hugging Face returned an empty analysis."
        )

    return parse_analysis(
        generated_text
    )


# ============================================================
# 6. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CodeShield AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 7. HEADER
# ============================================================

st.title("🛡️ CodeShield AI")

st.subheader(
    "AI-Powered Source Code Security Analyzer"
)

st.write(
    "Analyze source code, understand what it does, "
    "identify security issues, and receive safer fixes."
)

st.divider()


# ============================================================
# 8. API KEY CHECK
# ============================================================

if not HF_TOKEN:

    st.error(
        "❌ Hugging Face API token is not configured."
    )

    st.info(
        """
For local testing, create a `.env` file:

HF_TOKEN=YOUR_HUGGING_FACE_TOKEN

For Streamlit deployment, add HF_TOKEN
to Streamlit Secrets.

Never upload `.env` to GitHub.
"""
    )

    st.stop()


# ============================================================
# 9. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

language = st.sidebar.selectbox(
    "Programming Language",
    [
        "Python",
        "JavaScript",
        "Java",
        "C",
        "C++"
    ]
)

st.sidebar.divider()

st.sidebar.subheader(
    "🔍 Security Checks"
)

st.sidebar.write("✓ SQL Injection")
st.sidebar.write("✓ Command Injection")
st.sidebar.write("✓ Hardcoded Secrets")
st.sidebar.write("✓ Path Traversal")
st.sidebar.write("✓ Input Validation")
st.sidebar.write("✓ Authentication")
st.sidebar.write("✓ Authorization")
st.sidebar.write("✓ Sensitive Information")
st.sidebar.write("✓ Weak Cryptography")


# ============================================================
# 10. SOURCE CODE INPUT
# ============================================================

st.header("1️⃣ Source Code")

uploaded_file = st.file_uploader(
    "Upload source code",
    type=[
        "py",
        "js",
        "java",
        "c",
        "cpp",
        "txt"
    ]
)

code = ""

if uploaded_file is not None:

    try:

        code = uploaded_file.read().decode(
            "utf-8"
        )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

    except UnicodeDecodeError:

        st.error(
            "The uploaded file is not valid UTF-8 text."
        )

        st.stop()

else:

    code = st.text_area(
        "Or paste your source code:",
        height=300,
        placeholder="Paste source code here..."
    )


# ============================================================
# 11. CODE PREVIEW
# ============================================================

if code.strip():

    st.header("2️⃣ Code Preview")

    lexer_names = {
        "Python": "python",
        "JavaScript": "javascript",
        "Java": "java",
        "C": "c",
        "C++": "cpp"
    }

    try:

        lexer = get_lexer_by_name(
            lexer_names[language]
        )

        formatter = HtmlFormatter(
            noclasses=True
        )

        highlighted = highlight(
            code,
            lexer,
            formatter
        )

        st.components.v1.html(
            highlighted,
            height=350,
            scrolling=True
        )

    except Exception:

        st.code(
            code,
            language="text"
        )


# ============================================================
# 12. ANALYZE BUTTON
# ============================================================

st.header("3️⃣ Security Analysis")

if st.button(
    "🔍 Analyze Code",
    type="primary",
    use_container_width=True
):

    if not code.strip():

        st.warning(
            "Please upload or paste source code."
        )

    else:

        with st.spinner(
            "🛡️ CodeShield AI is analyzing..."
        ):

            try:

                report = analyze_code(
                    code,
                    language
                )

                st.session_state["report"] = report

                st.success(
                    "✅ Security analysis completed."
                )

            except Exception as error:

                st.error(
                    "❌ Security analysis failed."
                )

                st.warning(
                    str(error)
                )


# ============================================================
# 13. DISPLAY REPORT
# ============================================================

if "report" in st.session_state:

    report = st.session_state["report"]

    st.divider()

    st.header("4️⃣ Security Report")

    score = report.security_score

    if score >= 80:

        status = "🟢 Good"

    elif score >= 60:

        status = "🟡 Needs Improvement"

    else:

        status = "🔴 High Risk"

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Security Score",
            f"{score}/100"
        )

    with col2:

        st.metric(
            "Issues Found",
            len(report.issues)
        )

    with col3:

        st.metric(
            "Status",
            status
        )

    st.progress(
        score / 100
    )

    # --------------------------------------------------------
    # CODE EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "📖 Code Explanation"
    )

    st.info(
        report.code_explanation
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📋 Security Summary"
    )

    st.write(
        report.summary
    )

    # --------------------------------------------------------
    # ISSUES
    # --------------------------------------------------------

    st.subheader(
        "🚨 Security Issues"
    )

    if not report.issues:

        st.success(
            "🎉 No significant security issues detected."
        )

    else:

        for number, issue in enumerate(
            report.issues,
            start=1
        ):

            with st.expander(
                f"{number}. {issue.title} "
                f"— {issue.severity}",
                expanded=True
            ):

                st.write(
                    f"**📍 Line:** {issue.line}"
                )

                st.write(
                    f"**⚠️ Severity:** "
                    f"{issue.severity}"
                )

                st.markdown(
                    "### 🔍 Issue Explanation"
                )

                st.write(
                    issue.explanation
                )

                st.markdown(
                    "### 💥 Security Impact"
                )

                st.warning(
                    issue.impact
                )

                st.markdown(
                    "### ❌ Vulnerable Code"
                )

                st.code(
                    issue.vulnerable_code,
                    language=language.lower()
                )

                st.markdown(
                    "### ✅ Secure Code"
                )

                st.code(
                    issue.secure_code,
                    language=language.lower()
                )

                st.markdown(
                    "### 🛠️ Recommendation"
                )

                st.success(
                    issue.recommendation
                )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ CodeShield AI | AI-assisted secure coding analysis"
)