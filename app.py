import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


# ============================================================
# CODE SHIELD AI
# AI-POWERED SOURCE CODE SECURITY ANALYZER
# ============================================================


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/interactions"
)


# ============================================================
# 2. PYDANTIC DATA MODELS
# ============================================================

class Vulnerability(BaseModel):
    title: str
    severity: str
    line: int | None = None

    issue_explanation: str
    impact: str
    recommendation: str

    vulnerable_code: str
    secure_code: str
    fix_explanation: str


class SecurityReport(BaseModel):
    security_score: int = Field(
        ge=0,
        le=100
    )

    summary: str

    code_explanation: str

    vulnerabilities: list[Vulnerability]


# ============================================================
# 3. GEMINI API FUNCTION
# ============================================================

def analyze_code(
    code: str,
    language: str
) -> SecurityReport:

    prompt = f"""
You are CodeShield AI.

You are a professional secure-code review assistant.

Your job is to analyze source code for genuine security
vulnerabilities and explain the problems clearly to a
student/developer.

Programming language:
{language}

Analyze the following source code.

IMPORTANT SECURITY CHECKS:

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
- Insecure configuration
- Improper input validation
- Insecure random values
- Weak cryptography
- Other obvious secure-coding problems

IMPORTANT RULES:

1. Do not invent vulnerabilities.
2. Only report issues that are supported by the supplied code.
3. Do not report a problem simply because a library is old.
4. Give an approximate line number when possible.
5. Severity must be one of:
   Critical
   High
   Medium
   Low
   Informational
6. Explain every issue in simple language.
7. Explain why the issue is dangerous.
8. Show the relevant vulnerable code.
9. Show a secure replacement.
10. Explain why the replacement is safer.
11. Give a practical recommendation.
12. Give a security score from 0 to 100.
13. 100 means the supplied code has no significant
    security problems.
14. Do not claim the code is perfectly secure.
15. If no significant vulnerability is found, return an
    empty vulnerabilities array.

RETURN ONLY VALID JSON.

The JSON MUST follow this exact structure:

{{
    "security_score": 75,

    "summary": "Short overall security assessment.",

    "code_explanation": "Explain what the submitted code does in simple language.",

    "vulnerabilities": [
        {{
            "title": "SQL Injection",

            "severity": "High",

            "line": 10,

            "issue_explanation": "Explain exactly what is wrong and why it is a security problem.",

            "impact": "Explain what an attacker could potentially do.",

            "recommendation": "Explain the recommended security improvement.",

            "vulnerable_code": "The relevant vulnerable code snippet.",

            "secure_code": "A safer replacement code snippet.",

            "fix_explanation": "Explain why the secure code is safer."
        }}
    ]
}}

If there are no significant security vulnerabilities,
return:

{{
    "security_score": 100,

    "summary": "No significant security vulnerabilities were detected in the supplied code.",

    "code_explanation": "Explain what the code does.",

    "vulnerabilities": []
}}

SOURCE CODE:

---------------- START SOURCE CODE ----------------

{code}

----------------- END SOURCE CODE -----------------
"""

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Please create a .env file and add your Gemini API key."
        )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    payload = {
        "model": GEMINI_MODEL,
        "input": prompt
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:

        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "The Gemini API request timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Could not connect to Gemini API: {error}"
        )

    # --------------------------------------------------------
    # API ERROR
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_data = response.json()

            error_message = (
                error_data
                .get("error", {})
                .get("message", response.text)
            )

        except (ValueError, AttributeError):

            error_message = response.text

        raise RuntimeError(
            f"Gemini API error "
            f"({response.status_code}): "
            f"{error_message}"
        )

    # --------------------------------------------------------
    # PARSE RESPONSE
    # --------------------------------------------------------

    try:

        response_data = response.json()

    except ValueError:

        raise RuntimeError(
            "Gemini returned an invalid response."
        )

    # --------------------------------------------------------
    # GET OUTPUT TEXT
    # --------------------------------------------------------

    generated_text = ""

    # Some Gemini responses expose output_text directly.
    output_text = response_data.get(
        "output_text"
    )

    if isinstance(output_text, str):

        generated_text = output_text.strip()

    # --------------------------------------------------------
    # Fallback: inspect interaction steps
    # --------------------------------------------------------

    if not generated_text:

        steps = response_data.get(
            "steps",
            []
        )

        if isinstance(steps, list):

            for step in steps:

                if not isinstance(step, dict):
                    continue

                if step.get("type") != "model_output":
                    continue

                content = step.get(
                    "content",
                    []
                )

                if not isinstance(content, list):
                    continue

                for item in content:

                    if not isinstance(item, dict):
                        continue

                    if item.get("type") == "text":

                        text_value = item.get(
                            "text",
                            ""
                        )

                        if isinstance(
                            text_value,
                            str
                        ):

                            generated_text = (
                                text_value.strip()
                            )

                        if generated_text:
                            break

                if generated_text:
                    break

    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not generated_text:

        raise RuntimeError(
            "Gemini returned an empty analysis."
        )

    # --------------------------------------------------------
    # REMOVE MARKDOWN JSON FENCES
    # --------------------------------------------------------

    if generated_text.startswith(
        "```json"
    ):

        generated_text = generated_text[
            len("```json"):
        ]

    elif generated_text.startswith(
        "```"
    ):

        generated_text = generated_text[
            len("```"):
        ]

    if generated_text.endswith(
        "```"
    ):

        generated_text = generated_text[
            :-len("```")
        ]

    generated_text = generated_text.strip()

    # --------------------------------------------------------
    # CONVERT JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            generated_text
        )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "Gemini did not return valid JSON.\n\n"
            f"JSON parsing error: {error}\n\n"
            "Gemini response:\n"
            f"{generated_text}"
        )

    # --------------------------------------------------------
    # VALIDATE WITH PYDANTIC
    # --------------------------------------------------------

    try:

        report = SecurityReport.model_validate(
            result
        )

    except Exception as error:

        raise RuntimeError(
            "Gemini returned an unexpected "
            "security report format.\n\n"
            f"{error}"
        )

    return report


# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CodeShield AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 5. HEADER
# ============================================================

st.title(
    "🛡️ CodeShield AI"
)

st.subheader(
    "AI-Powered Source Code Security Analyzer"
)

st.write(
    """
CodeShield AI analyzes source code for security issues,
explains the vulnerabilities in simple language, shows
the affected code, and provides safer alternatives.
"""
)

st.divider()


# ============================================================
# 6. API KEY CHECK
# ============================================================

if not GEMINI_API_KEY:

    st.error(
        "❌ Gemini API key not found."
    )

    st.info(
        """
Create a file named `.env` in the same folder as
`app.py` and add:

GEMINI_API_KEY=YOUR_API_KEY_HERE
"""
    )

    st.stop()


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ CodeShield Settings"
    )

    language = st.selectbox(
        "Programming Language",
        [
            "Python",
            "JavaScript",
            "Java",
            "C",
            "C++"
        ]
    )

    st.divider()

    st.subheader(
        "🔍 Security Checks"
    )

    st.write(
        "✓ SQL Injection"
    )

    st.write(
        "✓ Command Injection"
    )

    st.write(
        "✓ Hardcoded Secrets"
    )

    st.write(
        "✓ Path Traversal"
    )

    st.write(
        "✓ Unsafe Input"
    )

    st.write(
        "✓ Authentication"
    )

    st.write(
        "✓ Sensitive Information"
    )

    st.write(
        "✓ Weak Cryptography"
    )

    st.divider()

    st.caption(
        "Gemini-powered security analysis"
    )


# ============================================================
# 8. CODE INPUT
# ============================================================

st.header(
    "1️⃣ Submit Source Code"
)

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


# ============================================================
# 9. FILE UPLOAD
# ============================================================

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
            "❌ The uploaded file is not valid UTF-8 text."
        )

        st.stop()


# ============================================================
# 10. CODE TEXT AREA
# ============================================================

else:

    code = st.text_area(
        "Or paste your source code here:",
        height=320,
        placeholder=(
            "Example:\n\n"
            "username = input('Username: ')\n"
            "query = \"SELECT * FROM users WHERE name='\" "
            "+ username + \"'\""
        )
    )


# ============================================================
# 11. CODE PREVIEW
# ============================================================

if code.strip():

    st.header(
        "2️⃣ Code Preview"
    )

    try:

        lexer_name = language.lower()

        if language == "C++":

            lexer_name = "cpp"

        lexer = get_lexer_by_name(
            lexer_name
        )

        formatter = HtmlFormatter(
            noclasses=True
        )

        highlighted_code = highlight(
            code,
            lexer,
            formatter
        )

        st.components.v1.html(
            highlighted_code,
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

st.header(
    "3️⃣ Security Analysis"
)

analyze_button = st.button(
    "🔍 Analyze Code",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not code.strip():

        st.warning(
            "⚠️ Please upload or paste source code first."
        )

    else:

        with st.spinner(
            "🛡️ CodeShield AI is analyzing your code..."
        ):

            try:

                report = analyze_code(
                    code,
                    language
                )

                st.session_state[
                    "security_report"
                ] = report

                st.success(
                    "✅ Analysis completed successfully."
                )

            except Exception as error:

                st.error(
                    "❌ Security analysis failed."
                )

                st.code(
                    str(error)
                )


# ============================================================
# 13. DISPLAY SECURITY REPORT
# ============================================================

if "security_report" in st.session_state:

    report = st.session_state[
        "security_report"
    ]

    st.divider()

    st.header(
        "4️⃣ Security Report"
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

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
            len(report.vulnerabilities)
        )

    with col3:

        st.metric(
            "Security Status",
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

    if not report.vulnerabilities:

        st.success(
            "🎉 No significant security "
            "vulnerabilities were detected."
        )

    else:

        for index, issue in enumerate(
            report.vulnerabilities,
            start=1
        ):

            severity = (
                issue.severity.lower()
            )

            if severity == "critical":

                icon = "🔴"

            elif severity == "high":

                icon = "🟠"

            elif severity == "medium":

                icon = "🟡"

            elif severity == "low":

                icon = "🟢"

            else:

                icon = "🔵"

            issue_title = (
                f"{icon} {index}. "
                f"{issue.title} "
                f"— {issue.severity}"
            )

            with st.expander(
                issue_title,
                expanded=True
            ):

                # ------------------------------------------------
                # ISSUE DETAILS
                # ------------------------------------------------

                if issue.line is not None:

                    st.write(
                        f"**📍 Line:** {issue.line}"
                    )

                st.write(
                    f"**⚠️ Severity:** "
                    f"{issue.severity}"
                )

                st.markdown(
                    "### 🔍 What is the issue?"
                )

                st.write(
                    issue.issue_explanation
                )

                # ------------------------------------------------
                # IMPACT
                # ------------------------------------------------

                st.markdown(
                    "### 💥 Security Impact"
                )

                st.warning(
                    issue.impact
                )

                # ------------------------------------------------
                # VULNERABLE CODE
                # ------------------------------------------------

                st.markdown(
                    "### ❌ Vulnerable Code"
                )

                st.code(
                    issue.vulnerable_code,
                    language=language.lower()
                )

                # ------------------------------------------------
                # SECURE CODE
                # ------------------------------------------------

                st.markdown(
                    "### ✅ Secure Code"
                )

                st.code(
                    issue.secure_code,
                    language=language.lower()
                )

                # ------------------------------------------------
                # FIX EXPLANATION
                # ------------------------------------------------

                st.markdown(
                    "### 💡 Why does this fix work?"
                )

                st.info(
                    issue.fix_explanation
                )

                # ------------------------------------------------
                # RECOMMENDATION
                # ------------------------------------------------

                st.markdown(
                    "### 🛠️ Recommendation"
                )

                st.success(
                    issue.recommendation
                )

    # ========================================================
    # 14. DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "📥 Export Report"
    )

    report_json = json.dumps(
        report.model_dump(),
        indent=4
    )

    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name="codeshield_security_report.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ CodeShield AI | "
    "AI-assisted secure coding analysis"
)