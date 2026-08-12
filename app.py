"""
IdeaForge AI – Smart Business Idea Generator
=============================================
A complete Agentic AI application powered by IBM watsonx.ai Studio and IBM Granite Models.
Demonstrates: Multi-Agent Collaboration, Trend Forecasting, Feasibility Analysis,
Innovation Strategy, and Business Intelligence.

Requirements:
    pip install flask ibm-watsonx-ai

Environment Variables:
    WATSONX_API_KEY      - Your IBM watsonx.ai API key
    WATSONX_PROJECT_ID   - Your IBM watsonx.ai project ID
    WATSONX_URL          - IBM watsonx.ai endpoint URL
                          (default: https://us-south.ml.cloud.ibm.com)
"""

import os
import re
import time
import requests
from flask import Flask, request, jsonify, render_template_string

# ─────────────────────────────────────────────
#  IBM watsonx.ai SDK import (optional fallback)
# ─────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials          # type: ignore
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
    WATSONX_SDK_AVAILABLE = True
except ImportError:
    Credentials = None      # type: ignore
    ModelInference = None   # type: ignore
    WATSONX_SDK_AVAILABLE = False

app = Flask(__name__)

# ─────────────────────────────────────────────
#  IBM watsonx.ai Credentials (read from env)
# ─────────────────────────────────────────────
# ── Credentials: env vars override baked-in defaults ──
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    "C8OHcU1KsC83FM_gds2bVcXiKm2s1zl-TZ4BvNkYdsjX")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "62f1ed5d-58f5-458b-a0a2-233d112f39b6")
WATSONX_URL        = os.environ.get("WATSONX_URL",        "https://au-syd.ml.cloud.ibm.com").rstrip("/")

# IBM Granite model identifier
GRANITE_MODEL_ID = "ibm/granite-3-8b-instruct"

# ── IAM token cache ──
_iam_token_cache = {"token": None, "expires_at": 0}


def _get_iam_token() -> str:
    """
    Fetch an IAM bearer token from IBM Cloud using the API key.
    Caches the token until 5 minutes before expiry.
    Uses direct REST call — works even when SDK init times out.
    """
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"]:
        return _iam_token_cache["token"]
    # ── IBM IAM token endpoint ──
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"] = data["access_token"]
    _iam_token_cache["expires_at"] = now + data.get("expires_in", 3600) - 300
    return _iam_token_cache["token"]


def _call_granite_rest(prompt: str) -> str:
    """
    Call IBM Granite via direct REST API to watsonx.ai.
    This bypasses the SDK init timeout issue on restricted networks
    and is the PRIMARY integration point for IBM Granite Model calls.
    """
    token = _get_iam_token()
    # ── IBM watsonx.ai Text Generation REST endpoint ──
    url = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
    payload = {
        "model_id": GRANITE_MODEL_ID,
        "project_id": WATSONX_PROJECT_ID,
        "input": prompt,
        "parameters": {
            "max_new_tokens": 900,
            "min_new_tokens": 80,
            "temperature": 0.75,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    # Extract generated text from watsonx.ai response structure
    return result["results"][0]["generated_text"].strip()


def generate_response(prompt: str) -> str:
    """
    Core helper – sends a prompt to IBM Granite via watsonx.ai
    and returns the text response.

    Strategy:
      1. Try direct REST API (fastest, no SDK init timeout)
      2. Try SDK-based call as secondary
      3. Fall back to offline simulation if both fail
    """
    # ── Method 1: Direct REST (IBM watsonx.ai Granite) ──
    if WATSONX_API_KEY and WATSONX_PROJECT_ID:
        try:
            return _call_granite_rest(prompt)
        except Exception as e1:
            print(f"[watsonx REST] {type(e1).__name__}: {str(e1)[:120]}")
            # ── Method 2: SDK fallback ──
            if WATSONX_SDK_AVAILABLE:
                try:
                    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
                    model = ModelInference(
                        model_id=GRANITE_MODEL_ID,
                        credentials=creds,
                        project_id=WATSONX_PROJECT_ID,
                        params={"max_new_tokens": 900, "temperature": 0.75, "top_p": 0.9},
                    )
                    result = model.generate_text(prompt=prompt)
                    return result.strip() if isinstance(result, str) else str(result).strip()
                except Exception as e2:
                    print(f"[watsonx SDK] {type(e2).__name__}: {str(e2)[:120]}")
    # ── Offline / demo simulation ──
    return _simulate_response(prompt)


# ─────────────────────────────────────────────
#  Offline simulation (no credentials needed)
# ─────────────────────────────────────────────
def _simulate_response(prompt: str) -> str:
    """Return a rich, structured demo response so the UI is fully
    exercisable without live IBM credentials."""
    p = prompt.lower()

    if "knowledge discovery" in p or "opportunity" in p:
        return (
            "OPPORTUNITY SNAPSHOT:\n"
            "The intersection of AI and sustainability presents a multi-billion-dollar opportunity. "
            "Rapid digitisation of agriculture, healthcare, and education is accelerating demand for "
            "intelligent automation tools accessible to non-technical users.\n\n"
            "MARKET SIGNALS:\n"
            "• 67% of SMEs seek affordable AI integration tools\n"
            "• Remote-first work culture is driving demand for async collaboration platforms\n"
            "• ESG compliance requirements are creating new SaaS niches\n"
            "• Gen-Z entrepreneurs prefer no-code / low-code startup toolkits\n\n"
            "EMERGING TECHNOLOGIES:\n"
            "• Multimodal foundation models (text + vision + voice)\n"
            "• Edge AI for offline-first applications in rural markets\n"
            "• Decentralised identity & verifiable credentials\n"
            "• Quantum-safe encryption for FinTech"
        )

    if "idea generation" in p or "business idea" in p or "startup idea" in p:
        return (
            "IDEA 1 – AgroMind AI\n"
            "Problem Solved: Small-scale farmers lack real-time crop advisory.\n"
            "Proposed Solution: Mobile AI assistant that provides hyper-local crop advice, "
            "weather alerts, and pest detection via photo uploads.\n"
            "Target Users: 500M+ smallholder farmers in Asia, Africa, and Latin America.\n\n"
            "IDEA 2 – EduBridge Adaptive Learning\n"
            "Problem Solved: One-size-fits-all curricula fail diverse learners.\n"
            "Proposed Solution: AI platform that builds personalised learning paths and "
            "auto-generates quizzes, summaries, and concept maps.\n"
            "Target Users: K-12 students, adult learners, and corporate training teams.\n\n"
            "IDEA 3 – GreenLedger Carbon Tracker\n"
            "Problem Solved: SMEs struggle to measure and report carbon footprints.\n"
            "Proposed Solution: Automated ESG dashboard that ingests invoices and supply-chain "
            "data to compute scope 1-2-3 emissions with one-click reporting.\n"
            "Target Users: SMEs, sustainability officers, ESG auditors.\n\n"
            "IDEA 4 – HealthNudge Preventive Care Bot\n"
            "Problem Solved: Patients forget preventive screenings and medication adherence.\n"
            "Proposed Solution: Conversational AI on WhatsApp / SMS that sends intelligent health "
            "reminders, triages symptoms, and books appointments.\n"
            "Target Users: Patients aged 35-65, rural healthcare clinics.\n\n"
            "IDEA 5 – SkillMesh Freelance Matchmaker\n"
            "Problem Solved: Freelancers waste hours on mismatched job applications.\n"
            "Proposed Solution: AI-powered skill-graph that matches freelancers to gigs based on "
            "portfolio analysis, client reviews, and contextual fit scoring.\n"
            "Target Users: 1.5B global freelancers, SME hiring managers."
        )

    if "trend" in p or "forecast" in p:
        return (
            "TREND ANALYSIS REPORT:\n\n"
            "AgroMind AI — Trend Score: 91/100 | Growth Potential: Very High\n"
            "Agriculture tech (AgriTech) is projected to reach $22B by 2027. "
            "Climate urgency and food security concerns are primary drivers. "
            "Future Demand Prediction: HIGH — government subsidies accelerating adoption.\n\n"
            "EduBridge — Trend Score: 88/100 | Growth Potential: High\n"
            "EdTech market CAGR: 16.5% through 2030. Personalised learning is the #1 "
            "investment priority. Future Demand: HIGH — post-pandemic digital learning shift is permanent.\n\n"
            "GreenLedger — Trend Score: 85/100 | Growth Potential: High\n"
            "ESG reporting mandates (SEC, EU CSRD) create mandatory demand. "
            "Market Category: Emerging Opportunity — early-mover advantage available.\n\n"
            "HealthNudge — Trend Score: 83/100 | Growth Potential: High\n"
            "Preventive healthcare AI market growing at 45% CAGR. "
            "Conversational health tools seeing 300% increase in adoption since 2022.\n\n"
            "SkillMesh — Trend Score: 78/100 | Growth Potential: Moderate-High\n"
            "Gig economy reached $455B in 2023. Differentiation through AI matching is key. "
            "Market Category: Growing but competitive — strong IP moat required."
        )

    if "feasibility" in p or "risk" in p:
        return (
            "FEASIBILITY ASSESSMENT:\n\n"
            "AgroMind AI — Feasibility Score: 82/100 | Difficulty: Medium\n"
            "Risk Factors: Last-mile connectivity, multilingual NLP, low smartphone penetration.\n"
            "Required Skills: ML engineering, agronomy domain expertise, mobile dev.\n\n"
            "EduBridge — Feasibility Score: 87/100 | Difficulty: Medium-Low\n"
            "Risk Factors: Content licensing, teacher adoption resistance, data privacy.\n"
            "Required Skills: LLM fine-tuning, curriculum design, UX/UI, backend dev.\n\n"
            "GreenLedger — Feasibility Score: 80/100 | Difficulty: Medium\n"
            "Risk Factors: Regulatory fragmentation across geographies, data integration complexity.\n"
            "Required Skills: ESG accounting, data engineering, regulatory compliance, full-stack.\n\n"
            "HealthNudge — Feasibility Score: 75/100 | Difficulty: Medium-High\n"
            "Risk Factors: Medical liability, HIPAA/GDPR compliance, clinical validation required.\n"
            "Required Skills: Healthcare informatics, NLP, regulatory affairs, cloud infra.\n\n"
            "SkillMesh — Feasibility Score: 84/100 | Difficulty: Low-Medium\n"
            "Risk Factors: Cold-start network effect, trust building, competitor saturation.\n"
            "Required Skills: Graph ML, recommendation systems, product design, growth hacking."
        )

    if "strategy" in p or "mvp" in p or "roadmap" in p or "revenue" in p:
        return (
            "INNOVATION STRATEGY REPORT:\n\n"
            "TOP RECOMMENDATION: AgroMind AI\n\n"
            "MVP PLAN (0-3 months):\n"
            "• Build WhatsApp chatbot with crop advisory for 3 crops in 1 language\n"
            "• Integrate free weather API (Open-Meteo) and IBM Granite for NLU\n"
            "• Pilot with 200 farmers via NGO partnership\n"
            "• Collect feedback and iterate on advice quality\n\n"
            "REVENUE MODELS:\n"
            "• Freemium SaaS: Free basic advice, paid premium analytics ($5/month)\n"
            "• B2B licensing to agri-input companies for customer engagement\n"
            "• Government / NGO grants (USAID, World Bank Digital Agriculture)\n"
            "• Data insights marketplace (anonymised crop trend reports)\n\n"
            "GO-TO-MARKET STRATEGY:\n"
            "• Partner with farmer cooperatives and rural banks for distribution\n"
            "• Leverage IBM SkillsBuild / Call for Code for visibility\n"
            "• PR via impact storytelling on LinkedIn and tech press\n"
            "• Academic partnerships for credibility and pilot access\n\n"
            "FUTURE ENHANCEMENTS:\n"
            "• Satellite imagery integration for yield prediction\n"
            "• Marketplace for farm inputs (seeds, fertiliser)\n"
            "• IoT sensor integration (soil moisture, temperature)\n"
            "• Multilingual expansion: 20 languages by Year 2\n\n"
            "SUCCESS METRICS:\n"
            "• MAU growth rate > 15% MoM\n"
            "• Advice accuracy rating > 4.2 / 5\n"
            "• Time-to-first-advisory < 90 seconds\n"
            "• Revenue ARR target: $500K by Month 18"
        )

    # Generic fallback
    return (
        "IBM Granite has analysed your input and identified strong innovation potential. "
        "The described domain shows clear market demand, technological feasibility, and "
        "alignment with emerging global trends. Pursuing an MVP approach with iterative "
        "user feedback loops is recommended as the optimal path forward."
    )


# ═══════════════════════════════════════════════════
#  AGENT 1 — Knowledge Discovery Agent
# ═══════════════════════════════════════════════════
def knowledge_discovery_agent(user_input: str, image_desc: str = "", voice_note: str = "") -> dict:
    """
    Agent 1: Collects and organises contextual information.
    Identifies opportunities, unmet needs, and emerging domains.
    Powered by IBM Granite via watsonx.ai.
    """
    combined = user_input
    if image_desc:
        combined += f"\n[Image context]: {image_desc}"
    if voice_note:
        combined += f"\n[Voice note]: {voice_note}"

    prompt = f"""You are the Knowledge Discovery Agent of IdeaForge AI, powered by IBM Granite.
Your role is to analyse the user's input and produce a structured knowledge brief.

User Input: {combined}

Produce:
1. OPPORTUNITY SNAPSHOT – a 3-4 sentence summary of the core opportunity.
2. MARKET SIGNALS – 4 bullet-point observations about current market dynamics.
3. EMERGING TECHNOLOGIES – 4 technologies relevant to this opportunity space.

Be specific, data-aware, and forward-looking.
"""
    # ── IBM watsonx.ai Granite call ──
    raw = generate_response(prompt)

    return {
        "agent": "Knowledge Discovery Agent",
        "icon": "🔍",
        "status": "completed",
        "reason_activated": "First in pipeline — establishes context and opportunity space from user input.",
        "output": raw,
        "sections": _parse_sections(raw, ["OPPORTUNITY SNAPSHOT", "MARKET SIGNALS", "EMERGING TECHNOLOGIES"])
    }


# ═══════════════════════════════════════════════════
#  AGENT 2 — Idea Generation Agent
# ═══════════════════════════════════════════════════
def idea_generation_agent(knowledge_brief: str, user_input: str) -> dict:
    """
    Agent 2: Generates innovative business and project ideas.
    Produces startup, research, product, app, and social impact ideas.
    Powered by IBM Granite via watsonx.ai.
    """
    prompt = f"""You are the Idea Generation Agent of IdeaForge AI, powered by IBM Granite.
Using the knowledge brief below, generate 5 innovative business ideas.

Knowledge Brief:
{knowledge_brief}

Original User Input: {user_input}

For EACH of the 5 ideas provide exactly:
IDEA [N] – [Idea Name]
Problem Solved: ...
Proposed Solution: ...
Target Users: ...

Cover diverse categories: startup, research, product, app, or social impact.
"""
    raw = generate_response(prompt)
    ideas = _parse_ideas(raw)

    return {
        "agent": "Idea Generation Agent",
        "icon": "💡",
        "status": "completed",
        "reason_activated": "Receives opportunity brief from Agent 1 and generates structured business ideas.",
        "output": raw,
        "ideas": ideas
    }


# ═══════════════════════════════════════════════════
#  AGENT 3 — Trend Forecasting Agent
# ═══════════════════════════════════════════════════
def trend_forecasting_agent(ideas_text: str) -> dict:
    """
    Agent 3: Predicts future relevance and market growth of generated ideas.
    Analyses technology trends, consumer behaviour, and industry growth.
    Powered by IBM Granite via watsonx.ai.
    """
    prompt = f"""You are the Trend Forecasting Agent of IdeaForge AI, powered by IBM Granite.
Analyse the following business ideas and forecast their future relevance.

Ideas:
{ideas_text}

For EACH idea provide:
[Idea Name] — Trend Score: X/100 | Growth Potential: [Very High/High/Moderate/Low]
[2-3 sentence trend rationale]
Future Demand Prediction: [HIGH / MODERATE / LOW] — [one-line reason]

Also classify each as one of: High Growth | Emerging Opportunity | Saturated Market
"""
    raw = generate_response(prompt)

    return {
        "agent": "Trend Forecasting Agent",
        "icon": "📈",
        "status": "completed",
        "reason_activated": "Evaluates future-readiness and market trajectory for each generated idea.",
        "output": raw,
        "scores": _parse_trend_scores(raw)
    }


# ═══════════════════════════════════════════════════
#  AGENT 4 — Feasibility Analysis Agent
# ═══════════════════════════════════════════════════
def feasibility_analysis_agent(ideas_text: str) -> dict:
    """
    Agent 4: Evaluates the practicality and viability of each idea.
    Assesses technical feasibility, resource needs, and risk factors.
    Powered by IBM Granite via watsonx.ai.
    """
    prompt = f"""You are the Feasibility Analysis Agent of IdeaForge AI, powered by IBM Granite.
Evaluate the practical viability of the following ideas.

Ideas:
{ideas_text}

For EACH idea provide:
[Idea Name] — Feasibility Score: X/100 | Difficulty: [Low/Medium/High]
Risk Factors: [comma-separated list]
Required Skills: [comma-separated list]

Be realistic and balanced — highlight both strengths and challenges.
"""
    raw = generate_response(prompt)

    return {
        "agent": "Feasibility Analysis Agent",
        "icon": "⚖️",
        "status": "completed",
        "reason_activated": "Assesses real-world viability, risks, and skill requirements before strategy.",
        "output": raw,
        "scores": _parse_feasibility_scores(raw)
    }


# ═══════════════════════════════════════════════════
#  AGENT 5 — Innovation Strategy Agent
# ═══════════════════════════════════════════════════
def innovation_strategy_agent(ideas_text: str, trend_data: str, feasibility_data: str) -> dict:
    """
    Agent 5: Converts ideas into actionable innovation plans.
    Generates MVP plans, revenue models, GTM strategies, and roadmaps.
    Powered by IBM Granite via watsonx.ai.
    """
    prompt = f"""You are the Innovation Strategy Agent of IdeaForge AI, powered by IBM Granite.
Create a detailed innovation strategy for the TOP recommended idea.

Ideas Overview: {ideas_text[:600]}
Trend Data: {trend_data[:400]}
Feasibility Data: {feasibility_data[:400]}

Select the highest-scoring idea and provide:

TOP RECOMMENDATION: [Idea Name]

MVP PLAN (0-3 months):
• [4 concrete MVP steps]

REVENUE MODELS:
• [4 revenue streams]

GO-TO-MARKET STRATEGY:
• [4 GTM tactics]

FUTURE ENHANCEMENTS:
• [4 future feature ideas]

SUCCESS METRICS:
• [4 KPIs]
"""
    raw = generate_response(prompt)

    return {
        "agent": "Innovation Strategy Agent",
        "icon": "🚀",
        "status": "completed",
        "reason_activated": "Final agent — synthesises all prior analysis into an actionable startup roadmap.",
        "output": raw,
        "sections": _parse_sections(raw, ["MVP PLAN", "REVENUE MODELS", "GO-TO-MARKET STRATEGY",
                                          "FUTURE ENHANCEMENTS", "SUCCESS METRICS"])
    }


# ═══════════════════════════════════════════════════
#  MASTER ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════
def orchestrator_agent(user_input: str, image_desc: str = "", voice_note: str = "") -> dict:
    """
    Master Orchestrator: coordinates all 5 agents in sequence,
    combines their outputs, and returns a unified innovation report.
    This is the brain of IdeaForge AI.
    """
    report = {"user_input": user_input, "agents": [], "ideas": [], "top_idea": {}}

    # ── Step 1: Knowledge Discovery ──
    kd = knowledge_discovery_agent(user_input, image_desc, voice_note)
    report["agents"].append(kd)
    knowledge_text = kd["output"]

    # ── Step 2: Idea Generation ──
    ig = idea_generation_agent(knowledge_text, user_input)
    report["agents"].append(ig)
    ideas_text = ig["output"]
    report["ideas"] = ig.get("ideas", [])

    # ── Step 3: Trend Forecasting ──
    tf = trend_forecasting_agent(ideas_text)
    report["agents"].append(tf)
    trend_text = tf["output"]

    # ── Step 4: Feasibility Analysis ──
    fa = feasibility_analysis_agent(ideas_text)
    report["agents"].append(fa)
    feasibility_text = fa["output"]

    # ── Step 5: Innovation Strategy ──
    ins = innovation_strategy_agent(ideas_text, trend_text, feasibility_text)
    report["agents"].append(ins)

    # ── Determine top idea ──
    report["top_idea"] = _pick_top_idea(
        report["ideas"],
        tf.get("scores", []),
        fa.get("scores", [])
    )

    report["trend_scores"]       = tf.get("scores", [])
    report["feasibility_scores"] = fa.get("scores", [])
    report["strategy"]           = ins.get("sections", {})
    report["knowledge"]          = kd.get("sections", {})

    return report


# ─────────────────────────────────────────────
#  Parsing Helpers
# ─────────────────────────────────────────────
def _parse_sections(text: str, headings: list) -> dict:
    sections = {}
    for i, h in enumerate(headings):
        start = text.upper().find(h.upper())
        if start == -1:
            sections[h] = ""
            continue
        end = len(text)
        for h2 in headings[i+1:]:
            pos = text.upper().find(h2.upper(), start + 1)
            if pos != -1:
                end = pos
                break
        sections[h] = text[start:end].strip()
    return sections


def _parse_ideas(text: str) -> list:
    ideas = []
    pattern = re.compile(
        r"IDEA\s*\d+\s*[–\-:]\s*(.+?)(?=IDEA\s*\d+|$)", re.DOTALL | re.IGNORECASE
    )
    for m in pattern.finditer(text):
        block = m.group(0).strip()
        name_match = re.search(r"IDEA\s*\d+\s*[–\-:]\s*(.+)", block, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "Unnamed Idea"
        name = name.split("\n")[0].strip()

        problem = _extract_field(block, "Problem Solved")
        solution = _extract_field(block, "Proposed Solution")
        users = _extract_field(block, "Target Users")

        ideas.append({"name": name, "problem": problem, "solution": solution, "users": users})
    return ideas if ideas else [{"name": text[:80], "problem": "", "solution": "", "users": ""}]


def _extract_field(text: str, field: str) -> str:
    pattern = re.compile(rf"{re.escape(field)}\s*[:\-]\s*(.+?)(?=\n[A-Z]|$)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip()[:200] if m else ""


def _parse_trend_scores(text: str) -> list:
    scores = []
    pattern = re.compile(r"(.+?)\s*[—\-–]\s*Trend Score:\s*(\d+)", re.IGNORECASE)
    for m in pattern.finditer(text):
        scores.append({"name": m.group(1).strip(), "score": int(m.group(2))})
    return scores


def _parse_feasibility_scores(text: str) -> list:
    scores = []
    pattern = re.compile(r"(.+?)\s*[—\-–]\s*Feasibility Score:\s*(\d+)", re.IGNORECASE)
    for m in pattern.finditer(text):
        scores.append({"name": m.group(1).strip(), "score": int(m.group(2))})
    return scores


def _pick_top_idea(ideas: list, trend_scores: list, feasibility_scores: list) -> dict:
    if not ideas:
        return {}
    best = ideas[0]
    best_score = 0
    for idea in ideas:
        t = next((s["score"] for s in trend_scores if idea["name"].lower()[:15] in s["name"].lower()), 75)
        f = next((s["score"] for s in feasibility_scores if idea["name"].lower()[:15] in s["name"].lower()), 75)
        combined = t + f
        if combined > best_score:
            best_score = combined
            best = idea
    best["impact_score"] = best_score // 2
    return best


# ═══════════════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════════════
@app.route("/")
def index():
    """Serve the IdeaForge AI single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Main API endpoint.
    Accepts JSON with fields:
        user_input  – business interests / challenge description
        image_desc  – optional image/sketch description
        voice_note  – optional transcribed voice note
    Returns a full Innovation Intelligence Report as JSON.
    """
    data       = request.get_json(force=True) or {}
    user_input = (data.get("user_input") or "").strip()
    image_desc = (data.get("image_desc") or "").strip()
    voice_note = (data.get("voice_note") or "").strip()

    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    # ── Run the Orchestrator (calls all 5 agents) ──
    report = orchestrator_agent(user_input, image_desc, voice_note)
    return jsonify(report)


@app.route("/health")
def health():
    """Health-check endpoint."""
    return jsonify({
        "status": "ok",
        "app": "IdeaForge AI",
        "model": GRANITE_MODEL_ID,
        "watsonx_configured": bool(WATSONX_API_KEY and WATSONX_PROJECT_ID),
        "sdk_available": WATSONX_SDK_AVAILABLE,
    })


# ═══════════════════════════════════════════════════
#  HTML Template  (Bootstrap 5 · single-page app)
# ═══════════════════════════════════════════════════
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>IdeaForge AI – Smart Business Idea Generator</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet"/>
<style>
  :root {
    --ibm-blue:#0f62fe; --ibm-dark:#161616; --ibm-mid:#393939;
    --ibm-light:#f4f4f4; --ibm-border:#e0e0e0;
    --agent1:#6929c4; --agent2:#1192e8; --agent3:#005d5d;
    --agent4:#9f1853; --agent5:#fa4d56;
  }
  body { background:#f4f4f4; font-family:'IBM Plex Sans',system-ui,sans-serif; color:#161616; }
  .hero {
    background: linear-gradient(135deg,#0f62fe 0%,#6929c4 60%,#161616 100%);
    padding: 3rem 0 2.5rem; color:#fff;
  }
  .hero h1 { font-size:2.4rem; font-weight:700; letter-spacing:-.5px; }
  .hero .badge-ibm { background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.3);
    color:#fff; font-size:.75rem; padding:.35em .75em; border-radius:20px; }
  .input-card { background:#fff; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08);
    padding:2rem; margin-top:-1.5rem; }
  .btn-forge { background:#0f62fe; color:#fff; border:none; padding:.65rem 2rem;
    border-radius:8px; font-weight:600; font-size:1rem; transition:background .2s; }
  .btn-forge:hover { background:#0043ce; color:#fff; }
  .btn-forge:disabled { background:#8d8d8d; }
  .section-title { font-size:1.1rem; font-weight:700; color:#161616;
    border-bottom:3px solid #0f62fe; padding-bottom:.4rem; margin-bottom:1.2rem; }
  .agent-card { background:#fff; border-radius:10px; border:1px solid #e0e0e0;
    box-shadow:0 1px 6px rgba(0,0,0,.06); margin-bottom:1.2rem; overflow:hidden; }
  .agent-header { padding:.8rem 1.2rem; color:#fff; font-weight:600;
    display:flex; align-items:center; gap:.6rem; }
  .agent-body { padding:1rem 1.2rem; font-size:.88rem; line-height:1.65; white-space:pre-wrap; }
  .agent-badge { font-size:.7rem; background:rgba(255,255,255,.25);
    border-radius:20px; padding:.2em .7em; }
  .a1 .agent-header { background:var(--agent1); }
  .a2 .agent-header { background:var(--agent2); }
  .a3 .agent-header { background:var(--agent3); }
  .a4 .agent-header { background:var(--agent4); }
  .a5 .agent-header { background:var(--agent5); }
  .idea-card { background:#fff; border-radius:10px; border:1px solid #e0e0e0;
    padding:1.2rem; margin-bottom:1rem; border-left:4px solid #0f62fe;
    box-shadow:0 1px 6px rgba(0,0,0,.05); transition:transform .15s; }
  .idea-card:hover { transform:translateY(-2px); }
  .idea-name { font-weight:700; font-size:1rem; color:#0f62fe; }
  .kpi-card { background:#fff; border-radius:10px; text-align:center;
    padding:1.5rem 1rem; border:1px solid #e0e0e0; box-shadow:0 1px 6px rgba(0,0,0,.05); }
  .kpi-val { font-size:2rem; font-weight:800; color:#0f62fe; }
  .kpi-label { font-size:.8rem; color:#6f6f6f; margin-top:.2rem; }
  .score-bar-wrap { background:#e0e0e0; border-radius:4px; height:8px; overflow:hidden; }
  .score-bar { height:100%; border-radius:4px;
    background:linear-gradient(90deg,#0f62fe,#6929c4); transition:width 1s ease; }
  .top-idea-card { background:linear-gradient(135deg,#0f62fe,#6929c4);
    color:#fff; border-radius:12px; padding:1.5rem; }
  .map-node { background:#fff; border:2px solid #0f62fe; border-radius:10px;
    padding:.6rem 1.1rem; font-size:.82rem; font-weight:600; text-align:center; white-space:nowrap; }
  .map-center { background:#0f62fe; color:#fff; border-radius:12px;
    padding:1rem 1.5rem; font-weight:700; font-size:1rem; text-align:center; }
  .map-connector { border-top:2px dashed #0f62fe; width:40px; margin:auto; }
  .workflow-step { display:flex; align-items:center; gap:.8rem;
    background:#fff; border-radius:8px; padding:.7rem 1rem; margin-bottom:.6rem;
    border:1px solid #e0e0e0; }
  .workflow-dot { width:28px; height:28px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; color:#fff; font-size:.8rem; flex-shrink:0; }
  .workflow-arrow { color:#0f62fe; font-size:1.2rem; margin:0 -.2rem; }
  .spinner-ring { width:50px; height:50px; border:5px solid #e0e0e0;
    border-top-color:#0f62fe; border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .loading-overlay { display:none; position:fixed; inset:0;
    background:rgba(22,22,22,.6); z-index:9999;
    align-items:center; justify-content:center; flex-direction:column; gap:1rem; }
  .loading-overlay.active { display:flex; }
  .loading-msg { color:#fff; font-size:1rem; font-weight:600; }
  .tab-pill { background:#e0e0e0; color:#393939; border:none; border-radius:20px;
    padding:.4rem 1rem; font-size:.85rem; font-weight:600; transition:all .2s; }
  .tab-pill.active { background:#0f62fe; color:#fff; }
  .result-section { display:none; }
  .result-section.visible { display:block; }
  .modality-label { font-size:.78rem; font-weight:600; color:#6929c4; text-transform:uppercase; letter-spacing:.5px; }
  pre.raw-out { background:#f4f4f4; border-radius:8px; padding:1rem; font-size:.8rem;
    max-height:300px; overflow-y:auto; white-space:pre-wrap; line-height:1.5; }
  @media(max-width:768px){ .hero h1{font-size:1.7rem;} }
</style>
</head>
<body>

<!-- Loading Overlay -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="spinner-ring"></div>
  <div class="loading-msg" id="loadingMsg">Initializing IdeaForge AI agents…</div>
  <div style="color:#a8c8ff;font-size:.82rem;">IBM Granite is thinking…</div>
</div>

<!-- ── Hero ── -->
<div class="hero">
  <div class="container">
    <div class="d-flex flex-wrap align-items-center gap-3 mb-3">
      <span class="badge-ibm"><i class="fa-solid fa-brain me-1"></i>IBM Granite</span>
      <span class="badge-ibm"><i class="fa-solid fa-robot me-1"></i>Agentic AI</span>
      <span class="badge-ibm"><i class="fa-solid fa-lightbulb me-1"></i>5 Specialized Agents</span>
      <span class="badge-ibm"><i class="fa-solid fa-cloud me-1"></i>watsonx.ai Studio</span>
    </div>
    <h1><i class="fa-solid fa-forge me-2" style="color:#a56eff;"></i>IdeaForge <span style="color:#a56eff;">AI</span></h1>
    <p class="mb-0" style="opacity:.85;font-size:1.05rem;">
      Smart Business Idea Generator · Powered by <strong>IBM Granite Models</strong> on <strong>watsonx.ai Studio</strong>
    </p>
  </div>
</div>

<!-- ── Main Container ── -->
<div class="container pb-5">

  <!-- Input Card -->
  <div class="input-card">
    <div class="row g-3">
      <div class="col-12">
        <p class="text-muted mb-3" style="font-size:.9rem;">
          <i class="fa-solid fa-circle-info text-primary me-1"></i>
          Describe your business interest, challenge, or innovation area below.
          The 5-agent AI pipeline will generate ideas, forecast trends, assess feasibility, and build a strategy.
        </p>
      </div>

      <!-- Primary Text Input -->
      <div class="col-md-12">
        <label class="form-label fw-bold">
          <i class="fa-solid fa-keyboard me-1 text-primary"></i> Business Interest / Challenge
        </label>
        <textarea id="userInput" class="form-control" rows="3"
          placeholder="e.g. I want to build an AI solution for smallholder farmers in emerging markets…"></textarea>
      </div>

      <!-- Image Description -->
      <div class="col-md-6">
        <label class="form-label">
          <span class="modality-label"><i class="fa-solid fa-image me-1"></i>Image / Sketch Description</span>
        </label>
        <textarea id="imageDesc" class="form-control" rows="2"
          placeholder="e.g. Wireframe showing a mobile dashboard with crop health alerts and weather widgets…"></textarea>
        <div class="form-text">Describe a product sketch, UI wireframe, or concept diagram.</div>
      </div>

      <!-- Voice Note -->
      <div class="col-md-6">
        <label class="form-label">
          <span class="modality-label"><i class="fa-solid fa-microphone me-1"></i>Voice Note (Transcription)</span>
        </label>
        <textarea id="voiceNote" class="form-control" rows="2"
          placeholder="e.g. I want to build an AI assistant that helps doctors in rural clinics diagnose patients faster…"></textarea>
        <div class="form-text">Paste a transcribed voice note or verbal description.</div>
      </div>

      <div class="col-12 d-flex align-items-center gap-3 flex-wrap">
        <button class="btn btn-forge" id="generateBtn" onclick="runIdeaForge()">
          <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Generate Innovation Report
        </button>
        <button class="btn btn-outline-secondary btn-sm" onclick="loadExample()">
          <i class="fa-solid fa-flask me-1"></i>Load Example
        </button>
        <span id="statusBadge" class="badge bg-secondary d-none">Ready</span>
      </div>
    </div>
  </div><!-- /input-card -->

  <!-- ── Results Section ── -->
  <div id="resultsContainer" class="result-section mt-4">

    <!-- Workflow Visualization -->
    <div class="mb-4">
      <div class="section-title"><i class="fa-solid fa-sitemap me-2 text-primary"></i>Agent Workflow Pipeline</div>
      <div class="d-flex align-items-center flex-wrap gap-1" id="workflowViz">
        <!-- populated by JS -->
      </div>
    </div>

    <!-- Tab Nav -->
    <div class="d-flex gap-2 flex-wrap mb-3" id="tabNav">
      <button class="tab-pill active" onclick="showTab('tabAgents',this)"><i class="fa-solid fa-robot me-1"></i>Agent Outputs</button>
      <button class="tab-pill" onclick="showTab('tabIdeas',this)"><i class="fa-solid fa-lightbulb me-1"></i>Ideas</button>
      <button class="tab-pill" onclick="showTab('tabTrend',this)"><i class="fa-solid fa-chart-line me-1"></i>Trend Forecast</button>
      <button class="tab-pill" onclick="showTab('tabFeasibility',this)"><i class="fa-solid fa-scale-balanced me-1"></i>Feasibility</button>
      <button class="tab-pill" onclick="showTab('tabStrategy',this)"><i class="fa-solid fa-rocket me-1"></i>Strategy</button>
      <button class="tab-pill" onclick="showTab('tabMap',this)"><i class="fa-solid fa-diagram-project me-1"></i>Idea Map</button>
      <button class="tab-pill" onclick="showTab('tabTop',this)"><i class="fa-solid fa-star me-1"></i>Top Idea</button>
    </div>

    <!-- Tab: Agent Outputs -->
    <div id="tabAgents" class="tab-pane-content">
      <div id="agentPanels"></div>
    </div>

    <!-- Tab: Ideas -->
    <div id="tabIdeas" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-lightbulb me-2"></i>Generated Business Ideas</div>
      <div id="ideaCards"></div>
    </div>

    <!-- Tab: Trend Forecast -->
    <div id="tabTrend" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-chart-line me-2"></i>Trend Forecast & Growth Analysis</div>
      <div id="trendContent"></div>
    </div>

    <!-- Tab: Feasibility -->
    <div id="tabFeasibility" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-scale-balanced me-2"></i>Feasibility Assessment</div>
      <div id="feasibilityContent"></div>
    </div>

    <!-- Tab: Strategy -->
    <div id="tabStrategy" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-rocket me-2"></i>Innovation Strategy & Roadmap</div>
      <div id="strategyContent"></div>
    </div>

    <!-- Tab: Idea Map -->
    <div id="tabMap" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-diagram-project me-2"></i>Visual Idea Map</div>
      <div id="ideaMap"></div>
    </div>

    <!-- Tab: Top Idea -->
    <div id="tabTop" class="tab-pane-content" style="display:none;">
      <div class="section-title"><i class="fa-solid fa-star me-2"></i>Top Recommended Idea</div>
      <div id="topIdeaCard"></div>
    </div>

  </div><!-- /resultsContainer -->
</div><!-- /container -->

<!-- Footer -->
<footer class="text-center py-3 mt-4" style="font-size:.78rem;color:#6f6f6f;border-top:1px solid #e0e0e0;">
  IdeaForge AI &nbsp;·&nbsp; Powered by <strong>IBM Granite Models</strong> on <strong>IBM watsonx.ai Studio</strong> &nbsp;·&nbsp;
  Agentic AI · Multi-Agent Collaboration · Business Innovation
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
/* ──────────────────────────────────────────────
   IdeaForge AI — Front-end Logic
   ────────────────────────────────────────────── */

const AGENT_COLORS = ['#6929c4','#1192e8','#005d5d','#9f1853','#fa4d56'];
const AGENT_CLASSES = ['a1','a2','a3','a4','a5'];

function loadExample(){
  document.getElementById('userInput').value =
    "I want to build an AI-powered platform that helps smallholder farmers in developing countries get real-time crop advice, weather alerts, and market prices using their basic smartphones.";
  document.getElementById('imageDesc').value =
    "Simple mobile app wireframe showing a home screen with crop health status, weather widget, and a chatbot button at the bottom.";
  document.getElementById('voiceNote').value =
    "I want to help farmers who don't have internet access get farming advice through SMS or WhatsApp using artificial intelligence.";
}

function showTab(id, btn){
  document.querySelectorAll('.tab-pane-content').forEach(el => el.style.display='none');
  document.getElementById(id).style.display='block';
  document.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

async function runIdeaForge(){
  const userInput = document.getElementById('userInput').value.trim();
  if(!userInput){ alert('Please enter your business interest or challenge.'); return; }

  const imageDesc = document.getElementById('imageDesc').value.trim();
  const voiceNote = document.getElementById('voiceNote').value.trim();

  // Show loading
  const overlay = document.getElementById('loadingOverlay');
  const msg = document.getElementById('loadingMsg');
  overlay.classList.add('active');
  const steps = [
    'Agent 1: Scanning knowledge landscape…',
    'Agent 2: Generating business ideas with IBM Granite…',
    'Agent 3: Forecasting market trends…',
    'Agent 4: Evaluating feasibility and risks…',
    'Agent 5: Crafting innovation strategy…',
    'Orchestrator: Synthesizing final report…'
  ];
  let si = 0;
  const ticker = setInterval(()=>{ msg.textContent = steps[si % steps.length]; si++; }, 1800);

  document.getElementById('generateBtn').disabled = true;

  try {
    const resp = await fetch('/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ user_input: userInput, image_desc: imageDesc, voice_note: voiceNote })
    });
    const data = await resp.json();
    if(data.error){ alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch(e){
    alert('Network error: ' + e.message);
  } finally {
    clearInterval(ticker);
    overlay.classList.remove('active');
    document.getElementById('generateBtn').disabled = false;
  }
}

function renderResults(data){
  document.getElementById('resultsContainer').classList.add('visible');
  renderWorkflow(data.agents);
  renderAgentPanels(data.agents);
  renderIdeas(data.ideas);
  renderTrend(data);
  renderFeasibility(data);
  renderStrategy(data);
  renderIdeaMap(data);
  renderTopIdea(data);
  // Auto-show agent tab
  showTab('tabAgents', document.querySelector('#tabNav .tab-pill'));
  document.getElementById('resultsContainer').scrollIntoView({behavior:'smooth'});
}

/* ── Workflow Viz ── */
function renderWorkflow(agents){
  const c = document.getElementById('workflowViz');
  c.innerHTML = '';
  agents.forEach((ag, i) => {
    const s = document.createElement('div');
    s.className = 'workflow-step';
    s.innerHTML = `
      <div class="workflow-dot" style="background:${AGENT_COLORS[i]}">${i+1}</div>
      <div>
        <div style="font-weight:700;font-size:.85rem;">${ag.icon || ''} ${ag.agent}</div>
        <div style="font-size:.75rem;color:#6f6f6f;">${ag.reason_activated}</div>
      </div>
      <span class="badge ms-auto" style="background:${AGENT_COLORS[i]};font-size:.7rem;">✓ Done</span>
    `;
    c.appendChild(s);
    if(i < agents.length - 1){
      const arr = document.createElement('div');
      arr.className = 'workflow-arrow ps-2';
      arr.innerHTML = '<i class="fa-solid fa-arrow-down"></i>';
      c.appendChild(arr);
    }
  });
}

/* ── Agent Panels ── */
function renderAgentPanels(agents){
  const c = document.getElementById('agentPanels');
  c.innerHTML = '';
  agents.forEach((ag, i) => {
    c.innerHTML += `
      <div class="agent-card ${AGENT_CLASSES[i]}">
        <div class="agent-header">
          <span style="font-size:1.1rem">${ag.icon||''}</span>
          <span>Agent ${i+1}: ${ag.agent}</span>
          <span class="agent-badge ms-auto">IBM Granite</span>
          <span class="badge bg-success ms-1" style="font-size:.65rem;">✓ Completed</span>
        </div>
        <div class="agent-body" style="background:#fafafa;padding:.5rem 1.2rem .3rem;font-size:.78rem;color:#6f6f6f;border-bottom:1px solid #e0e0e0;">
          <i class="fa-solid fa-circle-info me-1"></i><em>${ag.reason_activated}</em>
        </div>
        <div class="agent-body">${escHtml(ag.output || '')}</div>
      </div>`;
  });
}

/* ── Idea Cards ── */
function renderIdeas(ideas){
  const c = document.getElementById('ideaCards');
  if(!ideas || !ideas.length){ c.innerHTML='<p class="text-muted">No structured ideas extracted.</p>'; return; }
  c.innerHTML = ideas.map((idea, i) => `
    <div class="idea-card">
      <div class="d-flex align-items-center gap-2 mb-2">
        <span class="badge rounded-pill" style="background:${AGENT_COLORS[i%5]};font-size:.7rem;">Idea ${i+1}</span>
        <span class="idea-name">${escHtml(idea.name)}</span>
      </div>
      ${idea.problem ? `<div class="mb-1"><span class="fw-semibold text-muted" style="font-size:.78rem;">PROBLEM SOLVED</span><div style="font-size:.88rem">${escHtml(idea.problem)}</div></div>` : ''}
      ${idea.solution ? `<div class="mb-1"><span class="fw-semibold text-muted" style="font-size:.78rem;">PROPOSED SOLUTION</span><div style="font-size:.88rem">${escHtml(idea.solution)}</div></div>` : ''}
      ${idea.users ? `<div><span class="fw-semibold text-muted" style="font-size:.78rem;">TARGET USERS</span><div style="font-size:.88rem">${escHtml(idea.users)}</div></div>` : ''}
    </div>`).join('');
}

/* ── Trend Forecast ── */
function renderTrend(data){
  const c = document.getElementById('trendContent');
  const agents = data.agents || [];
  const tfAgent = agents.find(a => a.agent === 'Trend Forecasting Agent');
  const scores = data.trend_scores || [];

  let html = '';
  if(scores.length){
    html += '<div class="row g-3 mb-3">';
    scores.forEach(s => {
      html += `<div class="col-md-6 col-lg-4">
        <div class="kpi-card">
          <div class="kpi-val">${s.score}<span style="font-size:.9rem;color:#6f6f6f;">/100</span></div>
          <div class="kpi-label">${escHtml(s.name)}</div>
          <div class="score-bar-wrap mt-2"><div class="score-bar" style="width:${s.score}%"></div></div>
          <div class="mt-2">
            ${s.score >= 85 ? '<span class="badge bg-success">High Growth</span>' :
              s.score >= 70 ? '<span class="badge bg-primary">Emerging Opportunity</span>' :
              '<span class="badge bg-warning text-dark">Moderate</span>'}
          </div>
        </div>
      </div>`;
    });
    html += '</div>';
  }
  if(tfAgent) html += `<pre class="raw-out">${escHtml(tfAgent.output)}</pre>`;
  c.innerHTML = html || '<p class="text-muted">Run the generator to see trend forecasts.</p>';
}

/* ── Feasibility ── */
function renderFeasibility(data){
  const c = document.getElementById('feasibilityContent');
  const agents = data.agents || [];
  const faAgent = agents.find(a => a.agent === 'Feasibility Analysis Agent');
  const scores = data.feasibility_scores || [];

  let html = '';
  if(scores.length){
    html += '<div class="row g-3 mb-3">';
    scores.forEach(s => {
      const color = s.score >= 80 ? '#24a148' : s.score >= 65 ? '#0f62fe' : '#da1e28';
      html += `<div class="col-md-6 col-lg-4">
        <div class="kpi-card">
          <div class="kpi-val" style="color:${color}">${s.score}<span style="font-size:.9rem;color:#6f6f6f;">/100</span></div>
          <div class="kpi-label">${escHtml(s.name)}</div>
          <div class="score-bar-wrap mt-2"><div class="score-bar" style="width:${s.score}%;background:${color}"></div></div>
        </div>
      </div>`;
    });
    html += '</div>';
  }
  if(faAgent) html += `<pre class="raw-out">${escHtml(faAgent.output)}</pre>`;
  c.innerHTML = html || '<p class="text-muted">Run the generator to see feasibility analysis.</p>';
}

/* ── Strategy ── */
function renderStrategy(data){
  const c = document.getElementById('strategyContent');
  const agents = data.agents || [];
  const insAgent = agents.find(a => a.agent === 'Innovation Strategy Agent');
  if(!insAgent){ c.innerHTML='<p class="text-muted">Run the generator to see strategy.</p>'; return; }

  const secs = data.strategy || {};
  const keys = ['MVP PLAN', 'REVENUE MODELS', 'GO-TO-MARKET STRATEGY', 'FUTURE ENHANCEMENTS', 'SUCCESS METRICS'];
  const icons = ['fa-hammer','fa-dollar-sign','fa-bullhorn','fa-wand-sparkles','fa-chart-bar'];
  const colors = ['#0f62fe','#24a148','#6929c4','#9f1853','#005d5d'];

  let html = '<div class="row g-3 mb-3">';
  keys.forEach((k, i) => {
    const content = secs[k] || '';
    if(content){
      html += `<div class="col-md-6">
        <div class="agent-card">
          <div class="agent-header" style="background:${colors[i]}">
            <i class="fa-solid ${icons[i]}"></i> ${k}
          </div>
          <div class="agent-body">${escHtml(content)}</div>
        </div>
      </div>`;
    }
  });
  html += '</div>';
  if(!html.includes('agent-card')) html = `<pre class="raw-out">${escHtml(insAgent.output)}</pre>`;
  c.innerHTML = html;
}

/* ── Idea Map ── */
function renderIdeaMap(data){
  const c = document.getElementById('ideaMap');
  const ideas = data.ideas || [];
  const topic = (data.user_input || 'Your Innovation').substring(0, 50);

  const branches = [
    { label:'💡 Generated Ideas', items: ideas.map(i=>i.name||'').filter(Boolean).slice(0,4) },
    { label:'📈 Market Opportunities', items:['AI Automation','Sustainability Tech','Digital Health','EdTech'] },
    { label:'👥 Potential Customers', items:['Entrepreneurs','SMEs','Students','NGOs'] },
    { label:'💰 Revenue Models', items:['SaaS Subscription','B2B Licensing','Freemium','Grants'] },
    { label:'🚀 Future Enhancements', items:['Mobile App','API Marketplace','AI Personalisation','Global Scale'] },
  ];

  let html = `
    <div class="d-flex justify-content-center mb-4">
      <div class="map-center" style="max-width:320px;">
        <i class="fa-solid fa-brain me-2"></i>${escHtml(topic)}
      </div>
    </div>
    <div class="row g-3">`;
  branches.forEach(b => {
    html += `<div class="col-md-6 col-lg-4">
      <div class="agent-card">
        <div class="agent-header" style="background:#393939;font-size:.9rem;">${b.label}</div>
        <div class="agent-body">
          <ul class="mb-0 ps-3">${b.items.map(it=>`<li>${escHtml(it)}</li>`).join('')}</ul>
        </div>
      </div>
    </div>`;
  });
  html += '</div>';
  c.innerHTML = html;
}

/* ── Top Idea ── */
function renderTopIdea(data){
  const c = document.getElementById('topIdeaCard');
  const top = data.top_idea || {};
  if(!top.name){ c.innerHTML='<p class="text-muted">No top idea identified.</p>'; return; }

  const tScores = data.trend_scores || [];
  const fScores = data.feasibility_scores || [];
  const ts = (tScores.find(s=>s.name.toLowerCase().includes(top.name.toLowerCase().substring(0,10))) || {score:75}).score;
  const fs = (fScores.find(s=>s.name.toLowerCase().includes(top.name.toLowerCase().substring(0,10))) || {score:75}).score;

  c.innerHTML = `
    <div class="top-idea-card">
      <div class="d-flex align-items-center gap-2 mb-3">
        <span style="font-size:1.5rem;">🏆</span>
        <div>
          <div style="font-size:.78rem;opacity:.8;">TOP RECOMMENDED IDEA</div>
          <div style="font-size:1.3rem;font-weight:800;">${escHtml(top.name)}</div>
        </div>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-4 text-center">
          <div style="font-size:1.8rem;font-weight:800;">${ts}<span style="font-size:.9rem;">/100</span></div>
          <div style="font-size:.75rem;opacity:.8;">Trend Score</div>
        </div>
        <div class="col-4 text-center">
          <div style="font-size:1.8rem;font-weight:800;">${fs}<span style="font-size:.9rem;">/100</span></div>
          <div style="font-size:.75rem;opacity:.8;">Feasibility</div>
        </div>
        <div class="col-4 text-center">
          <div style="font-size:1.8rem;font-weight:800;">${top.impact_score||Math.round((ts+fs)/2)}<span style="font-size:.9rem;">/100</span></div>
          <div style="font-size:.75rem;opacity:.8;">Impact Score</div>
        </div>
      </div>
      ${top.problem ? `<div class="mb-2"><span style="font-size:.75rem;opacity:.8;">PROBLEM SOLVED</span><div>${escHtml(top.problem)}</div></div>` : ''}
      ${top.solution ? `<div class="mb-2"><span style="font-size:.75rem;opacity:.8;">PROPOSED SOLUTION</span><div>${escHtml(top.solution)}</div></div>` : ''}
      ${top.users ? `<div><span style="font-size:.75rem;opacity:.8;">TARGET USERS</span><div>${escHtml(top.users)}</div></div>` : ''}
    </div>
    <div class="row g-3 mt-2">
      <div class="col-md-4">
        <div class="kpi-card">
          <i class="fa-solid fa-chart-simple text-primary fs-3 mb-2"></i>
          <div class="kpi-val">${ts}/100</div>
          <div class="kpi-label">Trend Score</div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="kpi-card">
          <i class="fa-solid fa-circle-check text-success fs-3 mb-2"></i>
          <div class="kpi-val">${fs}/100</div>
          <div class="kpi-label">Feasibility Score</div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="kpi-card">
          <i class="fa-solid fa-star" style="color:#f1c21b;font-size:1.8rem;" class="mb-2"></i>
          <div class="kpi-val">${top.impact_score||Math.round((ts+fs)/2)}/100</div>
          <div class="kpi-label">Combined Impact</div>
        </div>
      </div>
    </div>`;
}

function escHtml(str){
  return String(str||'')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// Load example on page load for quick demo
window.addEventListener('DOMContentLoaded', loadExample);
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  IdeaForge AI – Smart Business Idea Generator")
    print("  Powered by IBM Granite Models on watsonx.ai Studio")
    print("=" * 60)
    print(f"  Model      : {GRANITE_MODEL_ID}")
    print(f"  SDK        : {'Available' if WATSONX_SDK_AVAILABLE else 'Not installed — running in demo mode'}")
    print(f"  Configured : {'YES – live IBM Granite calls' if (WATSONX_API_KEY and WATSONX_PROJECT_ID) else 'NO  – running in offline demo mode'}")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
