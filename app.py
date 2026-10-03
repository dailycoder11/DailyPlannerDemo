"""Daytrip: a small Streamlit day trip planner powered by Groq."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import date, datetime, time as day_time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from config.settings import ROOT_DIR, load_settings

load_dotenv(ROOT_DIR / ".env")
SETTINGS = load_settings()
LOG_PATH = (ROOT_DIR / SETTINGS.get("logFile", "logs/invocations.jsonl")).resolve()
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("daytrip.invocations")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def log_invocation(record: dict[str, Any]) -> None:
    """Append one structured invocation record to the configured JSONL file."""
    LOGGER.info(json.dumps(record, ensure_ascii=False, default=str))


def build_itinerary(
    *,
    destination: str,
    trip_date: date,
    people: int,
    budget_per_person: float,
    currency: str,
    start_time: day_time,
    return_time: day_time,
) -> tuple[str | None, str | None]:
    """Call Groq and log metadata for every attempted model invocation."""
    invocation_id = str(uuid.uuid4())
    started = time.monotonic()
    status = "error"
    error_message = None
    usage = None
    itinerary = None
    input_details = {
        "destination": destination,
        "date": trip_date.isoformat(),
        "people": people,
        "budgetPerPerson": budget_per_person,
        "currency": currency,
        "startTime": start_time.strftime("%H:%M"),
        "returnTime": return_time.strftime("%H:%M"),
    }

    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key":
            status = "configuration_error"
            error_message = "GROQ_API_KEY is missing."
            return None, "Planner service is not configured. Add GROQ_API_KEY to your .env file and restart Streamlit."

        client = Groq(api_key=api_key, timeout=45.0, max_retries=1)
        total_budget = budget_per_person * people
        completion = client.chat.completions.create(
            model=SETTINGS["model"],
            temperature=float(SETTINGS.get("temperature", 0.4)),
            max_tokens=int(SETTINGS.get("maxTokens", 1800)),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a practical local day-trip planner. Create a realistic itinerary that respects "
                        "the provided time window and per-person budget. Be concise and use Markdown headings, "
                        "times, clear activity descriptions, travel notes, and a budget estimate. Do not claim "
                        "live availability or exact prices; label costs as estimates. If local information is "
                        "uncertain, say so briefly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Plan a day trip to {destination} on {trip_date.isoformat()} for {people} "
                        f"{'person' if people == 1 else 'people'}. Trip window: {start_time:%H:%M} to "
                        f"{return_time:%H:%M} (local time). Maximum budget: {currency} {budget_per_person:.2f} "
                        f"per person, {currency} {total_budget:.2f} total. Suggest an itinerary and estimate "
                        "costs per person and for the group. Transport and interests can be inferred; call out "
                        "important assumptions."
                    ),
                },
            ],
        )
        itinerary = completion.choices[0].message.content
        if not itinerary or not itinerary.strip():
            status = "invalid_response"
            error_message = "Groq returned an empty response."
            return None, "The planner returned an empty itinerary. Please try again."

        usage_data = getattr(completion, "usage", None)
        if usage_data:
            usage = {
                "promptTokens": getattr(usage_data, "prompt_tokens", None),
                "completionTokens": getattr(usage_data, "completion_tokens", None),
                "totalTokens": getattr(usage_data, "total_tokens", None),
            }
        status = "success"
        return itinerary, None
    except Exception as error:  # Provider exceptions vary across SDK/API versions.
        status = "provider_error"
        error_message = f"{type(error).__name__}: {error}"
        return None, "We couldn’t reach the planner service. Please try again in a moment."
    finally:
        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "invocationId": invocation_id,
            "provider": SETTINGS.get("provider", "groq"),
            "model": SETTINGS.get("model"),
            "status": status,
            "durationMs": round((time.monotonic() - started) * 1000),
        }
        if usage:
            record["usage"] = usage
        if error_message:
            record["error"] = error_message
        if SETTINGS.get("logRequestAndResponse", False):
            record["input"] = input_details
            if itinerary:
                record["response"] = itinerary
        log_invocation(record)


st.set_page_config(
    page_title=f"{SETTINGS.get('appName', 'Daytrip')} — a day well planned",
    page_icon="✳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#242b27; --muted:#727b75; --green:#315a42; --lime:#dce9a4; --paper:#f7f7f4; }
      .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1020px; padding-top:1.3rem; padding-bottom:2rem; }
      .brand { font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-1px; font-size:19px; color:var(--ink); }
      .brand-mark { display:inline-grid; place-items:center; width:27px; height:27px; margin-right:8px; border-radius:50% 50% 50% 4px; background:var(--green); color:#fff; font:italic 19px Georgia,serif; transform:rotate(-7deg); }
      .topline { display:flex; justify-content:space-between; align-items:center; padding:0 0 17px; border-bottom:1px solid #e9eae5; color:#6f776e; font-size:12px; }
      .eyebrow { margin-top:4.2rem; color:#788278; font-size:9px; letter-spacing:.19em; font-weight:700; }
      .hero h1 { color:var(--ink); font:500 clamp(45px,5.4vw,68px)/1.08 'Manrope',sans-serif; letter-spacing:-.065em; margin:19px 0 15px; }
      .hero h1 em { color:var(--green); font-family:Georgia,serif; font-weight:400; letter-spacing:-.055em; }
      .hero p { max-width:355px; color:#707a73; font-size:14px; line-height:1.8; }
      .landscape { position:relative; height:210px; overflow:hidden; margin:31px 0 13px; border-radius:4px; background:#e9eee2; }
      .landscape .sun { position:absolute; top:30px; right:18%; width:56px; height:56px; border-radius:50%; background:#e9b07f; }
      .landscape .hill1,.landscape .hill2 { position:absolute; bottom:-65px; width:120%; height:145px; border-radius:50% 50% 0 0; }
      .landscape .hill1 { left:-20%; background:#c8d6b3; transform:rotate(-7deg); }
      .landscape .hill2 { left:15%; bottom:-92px; height:168px; background:#879f75; transform:rotate(7deg); }
      .art-note { position:absolute; z-index:1; top:15px; left:15px; padding:8px 11px; background:#f7f7f1df; color:#536354; font-size:10px; }
      .section-kicker { color:#788278; font-size:8px; letter-spacing:.19em; font-weight:700; }
      h2 { color:var(--ink); font-family:'Manrope',sans-serif; letter-spacing:-.045em; }
      [data-testid="stForm"] { padding:1.5rem 1.8rem 1.1rem; background:#fff; border:1px solid #edeee9; border-radius:5px; box-shadow:0 14px 45px #3645360a; }
      [data-testid="stForm"] label { color:#454e46; font-size:12px; font-weight:600; }
      [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stTimeInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"] { border-color:#e7e9e2; border-radius:3px; }
      [data-testid="stFormSubmitButton"] button { min-height:46px; margin-top:.5rem; border:0; border-radius:3px; background:var(--green); color:white; font-weight:600; }
      [data-testid="stFormSubmitButton"] button:hover { background:#254632; color:white; }
      .fineprint { padding-top:12px; color:#929b91; text-align:center; font-size:10px; }
      .result-panel { margin-top:1.5rem; padding:1.6rem 1.8rem; background:#fff; border:1px solid #edeee9; border-radius:5px; line-height:1.75; }
      .result-panel h1,.result-panel h2,.result-panel h3 { color:#2d3c30; font-family:'Manrope',sans-serif; }
      .footer { margin-top:3rem; padding-top:15px; border-top:1px solid #e9eae5; color:#8c948c; font-size:9px; letter-spacing:.1em; }
      @media(max-width:700px) { .eyebrow { margin-top:2rem; } .landscape { display:none; } [data-testid="stForm"] { padding:1.25rem 1rem .9rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="topline"><div class="brand"><span class="brand-mark">d</span>daytrip<span style="color:#d48b59">.</span></div><div>✳ &nbsp; Your next good day starts here</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.94, 1.06], gap="large", vertical_alignment="top")
with left:
    st.markdown(
        '<div class="hero"><div class="eyebrow">A LITTLE ESCAPE, WELL PLANNED</div>'
        '<h1>Make a day<br>of <em>somewhere.</em></h1>'
        '<p>Tell us where you’re headed. We’ll turn the details into a day worth looking forward to.</p>'
        '<div class="landscape"><span class="sun"></span><span class="hill1"></span><span class="hill2"></span><span class="art-note">✳ &nbsp; room for a little wonder</span></div>'
        '<div class="fineprint" style="text-align:left;letter-spacing:.14em">01 &nbsp; ─── &nbsp; YOUR DAY, YOUR PACE</div></div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-kicker">LET’S GET THE DETAILS</div>', unsafe_allow_html=True)
    st.subheader("Plan your day")
    with st.form("trip_details", clear_on_submit=False):
        destination = st.text_input("Where are you going?", placeholder="A city, town or somewhere nearby", max_chars=100)
        first_row, second_row = st.columns(2)
        with first_row:
            trip_date = st.date_input("When’s the day?", value=date.today(), min_value=date.today())
        with second_row:
            people = st.number_input("Who’s coming?", min_value=1, max_value=50, value=2, step=1)
        budget_col, currency_col = st.columns([1.55, 1])
        with budget_col:
            budget_per_person = st.number_input("Budget per person", min_value=0.0, value=2500.0, step=500.0, format="%.2f", help="A comfortable ceiling; we’ll do our best to stay under it.")
        with currency_col:
            currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP", "CAD", "AUD", "JPY"], index=0)
        start_col, return_col = st.columns(2)
        with start_col:
            start_time = st.time_input("Start around", value=day_time(9, 0), step=60)
        with return_col:
            return_time = st.time_input("Back by", value=day_time(19, 0), step=60)
        submitted = st.form_submit_button("Build my day  ↗", use_container_width=True)

    st.markdown('<div class="fineprint">✳ &nbsp; Thoughtfully made for your crew. No two days alike.</div>', unsafe_allow_html=True)

if submitted:
    if not destination.strip():
        st.error("Enter a destination to start planning your day.")
    elif start_time >= return_time:
        st.error("Your return time needs to be later than your start time.")
    elif budget_per_person <= 0:
        st.error("Enter a budget greater than zero per person.")
    else:
        with st.spinner("Putting your day together…"):
            itinerary, error = build_itinerary(
                destination=destination.strip(),
                trip_date=trip_date,
                people=int(people),
                budget_per_person=float(budget_per_person),
                currency=currency,
                start_time=start_time,
                return_time=return_time,
            )
        if error:
            st.error(error)
        elif itinerary:
            st.session_state["itinerary"] = itinerary
            st.session_state["trip_summary"] = (
                f"{destination.strip()} · {trip_date.strftime('%A, %B %-d, %Y')} · "
                f"{int(people)} {'person' if int(people) == 1 else 'people'} · "
                f"{currency} {budget_per_person:,.2f} per person"
            )

if st.session_state.get("itinerary"):
    st.markdown('<div class="result-panel"><div class="section-kicker">YOUR DAY, THOUGHTFULLY PLANNED</div><h2>Your itinerary</h2>', unsafe_allow_html=True)
    st.caption(st.session_state.get("trip_summary", ""))
    st.markdown(st.session_state["itinerary"])
    st.caption("Costs and local suggestions are estimates. Please confirm timings and prices before you go.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">DAYTRIP ✳ &nbsp; MADE FOR THE IN-BETWEEN DAYS</div>', unsafe_allow_html=True)
