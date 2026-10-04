"""Daytrip: a small Streamlit day trip planner powered by Groq."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import date, datetime, time as day_time
from pathlib import Path
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from evals.trip_evaluator import MAX_REVISIONS, evaluate_itinerary
from tools.harness import ToolCallHarness, TOOL_LIMIT_MESSAGE, WEATHER_DATE_REJECTION
from tools.registry import TOOL_DEFINITIONS

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
MAX_TOOL_STEPS = 3


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
    on_rate_limit_wait: Callable[[int], None] | None = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Call Groq with native function tools until it returns a final itinerary."""
    invocation_id = str(uuid.uuid4())
    started = time.monotonic()
    status = "error"
    error_message = None
    total_usage = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}
    revision_count = 0
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
            return None, None, "Planner service is not configured. Add GROQ_API_KEY to your .env file and restart Streamlit."

        client = Groq(api_key=api_key, timeout=45.0, max_retries=0)
        total_budget = budget_per_person * people
        system_prompt = (
            "You are a trip-planning assistant.\n\n"
            "Use the available tools whenever factual information about attractions, weather or travel time is needed.\n"
            "Do not invent facts that can be obtained from a tool.\n"
            "Gather enough information before creating the final itinerary.\n"
            "You may call each available tool at most once per trip request. Do not repeat a tool call.\n"
            "Avoid unnecessary tool calls.\n"
            "When sufficient information is available, produce a clear chronological trip plan that respects the user's "
            "start time, end time, group size and budget. Label prices as estimates and do not claim live availability.\n"
            "Return the final itinerary as ONLY a valid JSON object with this shape: "
            "{\"cost_per_person\": number, \"activities\": [{\"date\": \"YYYY-MM-DD\", "
            "\"start_time\": \"HH:MM\", \"end_time\": \"HH:MM\", \"title\": string, "
            "\"category\": \"activity or meal\", \"description\": string}]}. "
            "Include at least two activities and a lunch or meal activity; order activities chronologically."
        )
        user_prompt = (
            f"Plan a day trip to {destination} on {trip_date.isoformat()} for {people} "
            f"{'person' if people == 1 else 'people'}. Trip window: {start_time:%H:%M} to "
            f"{return_time:%H:%M} (local time). Maximum budget: {currency} {budget_per_person:.2f} "
            f"per person, {currency} {total_budget:.2f} total. Suggest an itinerary and estimate "
            "costs per person and for the group. Transport and interests can be inferred; call out "
            "important assumptions."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_tool_steps = min(MAX_TOOL_STEPS, max(0, int(SETTINGS.get("maxToolSteps", MAX_TOOL_STEPS))))
        harness = ToolCallHarness(
            trip_start_date=trip_date,
            trip_end_date=trip_date,
            max_successful_calls=max_tool_steps,
        )
        tools_enabled = True
        max_llm_calls = min(50, max(1, int(SETTINGS.get("maxLlmCallsPerRequest", 50))))
        retry_delay = max(0, int(SETTINGS.get("rateLimitRetryDelaySeconds", 30)))
        max_rate_limit_retries = max(0, int(SETTINGS.get("maxRateLimitRetries", 1)))
        llm_call_index = 0

        while True:
            rate_limit_retries = 0
            while True:
                if llm_call_index >= max_llm_calls:
                    status = "llm_call_limit_reached"
                    error_message = f"Maximum of {max_llm_calls} LLM calls reached for this request."
                    print(f"[LLM] request call limit reached ({max_llm_calls})")
                    return None, None, (
                        f"The planner reached its limit of {max_llm_calls} LLM calls for this request. "
                        "Please submit a new request to start a fresh plan."
                    )
                llm_call_index += 1
                call_started = time.monotonic()
                request_messages = json.loads(json.dumps(messages, ensure_ascii=False))
                completion = None
                call_error = None
                call_status = "response_received"
                should_retry = False
                try:
                    completion = client.chat.completions.create(
                        model=SETTINGS["model"],
                        temperature=float(SETTINGS.get("temperature", 0.4)),
                        max_tokens=int(SETTINGS.get("maxTokens", 1800)),
                        messages=messages,
                        tools=TOOL_DEFINITIONS if tools_enabled else None,
                        tool_choice="auto" if tools_enabled else "none",
                    )
                except RateLimitError as exc:
                    call_error = f"{type(exc).__name__}: {exc}"
                    error_text = (str(exc) + " " + json.dumps(getattr(exc, "body", None), default=str)).lower()
                    is_tpm_limit = "tokens per minute" in error_text or "tpm" in error_text
                    should_retry = is_tpm_limit and rate_limit_retries < max_rate_limit_retries
                    call_status = "rate_limit_retry" if should_retry else "rate_limit_error"
                    if not should_retry:
                        raise
                except Exception as exc:
                    call_error = f"{type(exc).__name__}: {exc}"
                    call_status = "error"
                    raise
                finally:
                    response_record: dict[str, Any] = {"error": call_error} if call_error else {}
                    if completion is not None:
                        response_record = completion.model_dump(mode="json")
                        usage_data = getattr(completion, "usage", None)
                        if usage_data:
                            total_usage["promptTokens"] += getattr(usage_data, "prompt_tokens", 0) or 0
                            total_usage["completionTokens"] += getattr(usage_data, "completion_tokens", 0) or 0
                            total_usage["totalTokens"] += getattr(usage_data, "total_tokens", 0) or 0
                    log_invocation({
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "invocationId": invocation_id,
                        "callIndex": llm_call_index,
                        "provider": SETTINGS.get("provider", "groq"),
                        "model": SETTINGS.get("model"),
                        "status": call_status,
                        "durationMs": round((time.monotonic() - call_started) * 1000),
                        "prompt": {
                            "messages": request_messages,
                            "tools": TOOL_DEFINITIONS if tools_enabled else None,
                            "toolChoice": "auto" if tools_enabled else "none",
                            "temperature": float(SETTINGS.get("temperature", 0.4)),
                            "maxTokens": int(SETTINGS.get("maxTokens", 1800)),
                        },
                        "response": response_record,
                    })

                if not should_retry:
                    break
                rate_limit_retries += 1
                print(f"[LLM] token-per-minute limit reached; retrying in {retry_delay} seconds")
                for remaining in range(retry_delay, 0, -1):
                    if on_rate_limit_wait:
                        on_rate_limit_wait(remaining)
                    time.sleep(1)

            choice = completion.choices[0]
            assistant_message = choice.message
            tool_calls = assistant_message.tool_calls or []
            if not tool_calls:
                itinerary = assistant_message.content
                print("[LLM] returned final answer")
                evaluation = evaluate_itinerary(
                    itinerary or "",
                    budget_per_person=budget_per_person,
                    requested_start_time=start_time,
                    requested_end_time=return_time,
                    trip_start_date=trip_date,
                    trip_end_date=trip_date,
                )
                for check_name, check_status in evaluation["checks"].items():
                    print(f"[EVAL] {check_name}: {check_status}")
                print(f"[EVAL] overall: {evaluation['status']}")
                if evaluation["status"] == "FAIL" and revision_count < MAX_REVISIONS:
                    revision_count += 1
                    print(f"[REVISION {revision_count}/{MAX_REVISIONS}]")
                    messages.append({"role": "assistant", "content": itinerary or ""})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Revise the itinerary to address these deterministic evaluation failures: "
                            f"{json.dumps(evaluation['failures'], ensure_ascii=False)}. Use information already "
                            "collected in this conversation and do not invent factual details. Return ONLY a valid "
                            "JSON object with cost_per_person and an activities array; each activity must have date, "
                            "start_time, end_time, title, category, and description. Keep the activities chronological, "
                            "within the requested trip dates and times, within the per-person budget, and include "
                            "at least two activities and a lunch or meal."
                        ),
                    })
                    continue
                status = "success" if evaluation["status"] == "PASS" else "evaluation_failed"
                return itinerary or "", evaluation, None

            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [call.model_dump(mode="json") for call in tool_calls],
            })
            limit_reached_this_turn = False
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                try:
                    parsed_for_trace = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (TypeError, ValueError):
                    parsed_for_trace = {}
                requested_date = parsed_for_trace.get("trip_date") if isinstance(parsed_for_trace, dict) else None
                suffix = f": {requested_date}" if name == "get_weather" and requested_date else ""
                print(f"[LLM] requested {name}{suffix}")

                if not tools_enabled:
                    tool_result = {"error": TOOL_LIMIT_MESSAGE}
                    print("[HARNESS] BLOCKED - tool-call limit reached")
                    limit_reached_this_turn = True
                else:
                    parsed, signature, rejection = harness.validate(name, arguments)
                    if rejection:
                        if rejection == "duplicate request":
                            tool_result = {"error": "This identical tool request has already completed. Choose a different request or continue with the information available."}
                            print("[HARNESS] BLOCKED - duplicate request")
                        elif rejection == "tool-call limit reached":
                            tool_result = {"error": TOOL_LIMIT_MESSAGE}
                            print("[HARNESS] BLOCKED - tool-call limit reached")
                            tools_enabled = False
                            limit_reached_this_turn = True
                        else:
                            tool_result = {"error": rejection}
                            if rejection == WEATHER_DATE_REJECTION:
                                print("[HARNESS] BLOCKED - date outside trip range")
                            else:
                                print(f"[HARNESS] BLOCKED - {rejection}")
                    else:
                        print("[HARNESS] ALLOWED")
                        tool_result, succeeded = harness.execute_validated(name, parsed or {}, signature or "")
                        if succeeded:
                            print(f"[TOOL {harness.successful_call_count}/{max_tool_steps}] completed")
                            if harness.successful_call_count >= max_tool_steps:
                                tools_enabled = False
                                limit_reached_this_turn = True
                        else:
                            print(f"[TOOL] {name} failed")

                result_content = json.dumps(tool_result, ensure_ascii=False, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result_content,
                })

            if limit_reached_this_turn:
                messages.append({
                    "role": "user",
                    "content": TOOL_LIMIT_MESSAGE,
                })
    except RateLimitError as error:
        status = "rate_limit_error"
        error_message = f"{type(error).__name__}: {error}"
        return None, None, "Groq’s token limit is still exhausted after the retry. Please wait a little longer and try again."
    except Exception as error:  # Provider exceptions vary across SDK/API versions.
        error_message = f"{type(error).__name__}: {error}"
        return None, None, "We couldn’t reach the planner service. Please try again in a moment."
    finally:
        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "invocationId": invocation_id,
            "provider": SETTINGS.get("provider", "groq"),
            "model": SETTINGS.get("model"),
            "status": status,
            "durationMs": round((time.monotonic() - started) * 1000),
        }
        record["usage"] = total_usage
        if error_message:
            record["error"] = error_message
        # Each LLM request/response pair is logged in the loop above, including
        # intermediate tool calls. This summary ties them together by invocationId.
        LOGGER.info(json.dumps(record, ensure_ascii=False, default=str))


def format_itinerary_for_display(content: str, currency: str) -> str:
    """Render the model's structured itinerary inside the existing result panel."""
    try:
        plan = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content
    if not isinstance(plan, dict) or not isinstance(plan.get("activities"), list):
        return content

    lines = []
    cost = plan.get("cost_per_person")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        lines.append(f"**Estimated cost per person:** {currency} {cost:,.2f}")
        lines.append("")
    for activity in plan["activities"]:
        if not isinstance(activity, dict):
            continue
        date_text = str(activity.get("date", ""))
        start_text = str(activity.get("start_time", ""))
        end_text = str(activity.get("end_time", ""))
        title = str(activity.get("title", "Activity"))
        category = str(activity.get("category", "")).strip()
        description = str(activity.get("description", "")).strip()
        timing = " – ".join(part for part in (start_text, end_text) if part)
        heading = " · ".join(part for part in (timing, date_text, category) if part)
        lines.append(f"### {title}")
        if heading:
            lines.append(f"_{heading}_")
        if description:
            lines.append(description)
        lines.append("")
    return "\n".join(lines).strip() or content


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
        rate_limit_notice = st.empty()

        def show_rate_limit_wait(seconds_remaining: int) -> None:
            rate_limit_notice.warning(
                "Groq’s token-per-minute limit was reached. We’ll automatically try again "
                f"in {seconds_remaining} seconds."
            )

        with st.spinner("Putting your day together…"):
            itinerary, evaluation, error = build_itinerary(
                destination=destination.strip(),
                trip_date=trip_date,
                people=int(people),
                budget_per_person=float(budget_per_person),
                currency=currency,
                start_time=start_time,
                return_time=return_time,
                on_rate_limit_wait=show_rate_limit_wait,
            )
        rate_limit_notice.empty()
        if error:
            st.error(error)
        elif itinerary:
            st.session_state["itinerary"] = itinerary
            st.session_state["evaluation"] = evaluation
            st.session_state["trip_summary"] = (
                f"{destination.strip()} · {trip_date.strftime('%A, %B %-d, %Y')} · "
                f"{int(people)} {'person' if int(people) == 1 else 'people'} · "
                f"{currency} {budget_per_person:,.2f} per person"
            )

if st.session_state.get("itinerary"):
    st.markdown('<div class="result-panel"><div class="section-kicker">YOUR DAY, THOUGHTFULLY PLANNED</div><h2>Your itinerary</h2>', unsafe_allow_html=True)
    st.caption(st.session_state.get("trip_summary", ""))
    evaluation = st.session_state.get("evaluation")
    if evaluation:
        if evaluation.get("status") == "PASS":
            st.success("Itinerary checks passed.")
        else:
            failures = evaluation.get("failures", [])
            st.warning("Itinerary needs review: " + "; ".join(failures))
    st.markdown(format_itinerary_for_display(st.session_state["itinerary"], currency))
    st.caption("Costs and local suggestions are estimates. Please confirm timings and prices before you go.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">DAYTRIP ✳ &nbsp; MADE FOR THE IN-BETWEEN DAYS</div>', unsafe_allow_html=True)
