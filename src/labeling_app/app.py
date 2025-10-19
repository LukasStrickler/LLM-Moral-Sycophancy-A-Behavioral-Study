"""Streamlit entrypoint for the moral sycophancy labeling platform."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Final, TypeVar

import markdown
import streamlit as st
import streamlit_cookies_manager as cookies_manager
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Ensure the project source directory is importable when running via Streamlit
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from labeling_app.core.assignment import AssignmentService  # noqa: E402
from labeling_app.core.models import Dataset  # noqa: E402
from labeling_app.db import create_client, ensure_schema  # noqa: E402
from labeling_app.settings import AppSettings, get_settings  # noqa: E402

st.set_page_config(
    page_title="LLM Moral Sycophancy - Labeling Portal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root { color-scheme: dark; }

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0f172a 55%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

main .block-container {
    max-width: 1000px;
    padding: 0.75rem 0.9rem 2.25rem;
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.92);
    padding-top: 0.5rem;
}

.stButton > button,
.stForm button {
    border-radius: 10px;
}

.stCustomComponentV1 iframe[src*="streamlit_cookies_manager"],
.stCustomComponentV1 iframe[title*="streamlit_cookies_manager"],
.stCustomComponentV1 iframe[src*="cookie_manager"],
.stCustomComponentV1 iframe[title*="cookie_manager"],
iframe[src*="streamlit_cookies_manager"],
iframe[title*="streamlit_cookies_manager"],
iframe[src*="cookie_manager"],
iframe[title*="cookie_manager"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    overflow: hidden !important;
}

.stContainer {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

div[data-testid="stMetric"] {
    background: rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    padding: 1.2rem 0.75rem;
}

div[data-testid="stMetric"] svg {
    display: none;
}

div[data-testid="stMetricValue"] {
    font-weight: 600;
}

/* Add 32px top padding specifically to the Submit button */
.submit-button-spacer {
    height: 28px;
}

/* Hide the "Press Enter to submit this form" text */
div[data-testid="InputInstructions"] {
    visibility: hidden !important;
    display: none !important;
}

/* Disable bottom padding for main block container */
div[data-testid="stMainBlockContainer"],
.stMainBlockContainer.block-container {
    padding-bottom: 1em !important;
    padding-top: 2em !important;
}

/* Lock viewport - no page scrolling */
html, body, .stApp, div[data-testid="stMain"] {
    height: 100vh !important;
    overflow: hidden !important;
}

/* Allow main block to scroll only if needed for layout */
div[data-testid="stMainBlockContainer"] {
    height: 100% !important;
    overflow-y: auto !important;
}

@media (max-width: 900px) {
    main .block-container {
        padding: 0.5rem 0.75rem 2.25rem;
    }
    
    div[data-testid="stMainBlockContainer"] {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    
    /* Force rating and submit to stay in one row on mobile - only for form */
    @media (max-width: 768px) {
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
        
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div {
            flex: 1 !important;
            min-width: 0 !important;
        }
        
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div:first-child {
            flex: 2 !important;
        }
        
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div:last-child {
            flex: 1 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        /* Limit prompt and response areas to 20vh on mobile */
        div[style*="max-height: 50vh"] {
            max-height: 25vh !important;
        }
    }
}
</style>
"""

SESSION_REVIEWER_CODE: Final = "reviewer_code"
SESSION_ACTIVE_DATASET: Final = "active_dataset"
SESSION_CURRENT_PAYLOAD: Final = "current_payload"
SESSION_PREFETCHED_PAYLOAD: Final = "prefetched_payload"
SESSION_RATING_VALUE: Final = "rating_value"
SESSION_COMMENT: Final = "comment_text"
SESSION_ACTIVE_PAGE: Final = "active_page"
SESSION_LAST_SUBMIT_JOB: Final = "__last_submit_job__"
SESSION_PROGRESS_STORE: Final = "__progress_store__"

REVIEWER_COOKIE_KEY: Final = "reviewer_code"


@dataclass(frozen=True)
class DatasetSummary:
    dataset: Dataset
    title: str
    description: str
    icon: str
    total: int
    completed: int
    available: bool


@dataclass
class ProgressSnapshot:
    completed: int = 0
    total: int = 0
    pending: int = 0
    last_synced: datetime | None = None

    @property
    def display_completed(self) -> int:
        return self.completed + self.pending

    @property
    def completion_ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(self.display_completed / self.total, 1.0)


class BackgroundJobManager:
    """Run lightweight background jobs for non-blocking DB operations."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self._results: dict[str, dict] = {}
        self._results_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            add_script_run_ctx(self._worker, get_script_run_ctx())
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            job_id, job_type, payload = job
            try:
                result = self._execute(job_type, payload)
                with self._results_lock:
                    self._results[job_id] = {"status": "completed", "result": result}
            except Exception as exc:  # pragma: no cover - log only
                message = str(exc) if str(exc) else repr(exc)
                with self._results_lock:
                    self._results[job_id] = {"status": "failed", "error": message}
            finally:
                self._queue.task_done()

    def _execute(self, job_type: str, payload: dict) -> dict:
        settings = get_settings()
        client = create_client(settings)
        service = AssignmentService(client)

        if job_type == "submit_review":
            return service.submit_review(
                payload["response_id"],
                payload["reviewer_code"],
                payload["score"],
                payload["notes"],
            )

        raise ValueError(f"Unknown job type: {job_type}")

    def submit_job(self, job_type: str, payload: dict) -> str:
        job_id = f"{job_type}_{uuid.uuid4().hex}"
        self._queue.put((job_id, job_type, payload))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        with self._results_lock:
            return self._results.pop(job_id, None)

    def cleanup_old_results(self, max_age_seconds: float = 300) -> None:
        now = time.time()
        with self._results_lock:
            self._results = {
                job_id: result
                for job_id, result in self._results.items()
                if now - (int(job_id.split("_")[-1]) / 1000) < max_age_seconds
            }


@st.cache_resource
def get_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


DATASET_META: dict[Dataset, dict[str, str]] = {
    Dataset.AITA: {
        "title": "AITA Dataset",
        "description": "Reddit AITA judgments",
        "icon": "📝",
    },
    Dataset.SCENARIO: {
        "title": "Scenario Dataset",
        "description": "Rent scenario prompts",
        "icon": "🏠",
    },
}


@st.cache_resource
def bootstrap() -> AppSettings:
    """Return cached settings. Schema migrations run via scripts ahead of time."""
    return get_settings()


@st.cache_resource
def get_assignment_service_cached() -> AssignmentService:
    settings = get_settings()
    client = create_client(settings)
    ensure_schema(client)
    return AssignmentService(client)


T = TypeVar("T")


def _with_service_retry(
    operation: Callable[[AssignmentService], T], attempts: int = 3, delay: float = 0.25
) -> T:
    for attempt in range(attempts):
        try:
            service = get_assignment_service_cached()
            return operation(service)
        except Exception:
            get_assignment_service_cached.clear()
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


SESSION_DEFAULTS: dict[str, object] = {
    SESSION_REVIEWER_CODE: None,
    SESSION_ACTIVE_DATASET: None,
    SESSION_CURRENT_PAYLOAD: None,
    SESSION_PREFETCHED_PAYLOAD: None,
    SESSION_RATING_VALUE: 0.0,
    SESSION_COMMENT: "",
    SESSION_ACTIVE_PAGE: "overview",
    SESSION_LAST_SUBMIT_JOB: None,
}


def apply_theme() -> None:
    st.markdown(THEME_STYLES, unsafe_allow_html=True)


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault(SESSION_PROGRESS_STORE, {})


def reset_assignments() -> None:
    st.session_state[SESSION_CURRENT_PAYLOAD] = None
    st.session_state[SESSION_PREFETCHED_PAYLOAD] = None
    st.session_state.pop(SESSION_RATING_VALUE, None)
    st.session_state.pop(SESSION_COMMENT, None)
    st.session_state.setdefault(SESSION_RATING_VALUE, 0.0)
    st.session_state.setdefault(SESSION_COMMENT, "")
    st.session_state[SESSION_LAST_SUBMIT_JOB] = None


def get_cookie_manager() -> cookies_manager.EncryptedCookieManager:
    password = os.environ.get("COOKIE_ENCRYPTION_PASSWORD")
    if not password:
        # Generate ephemeral key for development environments
        import secrets

        password = secrets.token_urlsafe(32)
        st.warning(
            "⚠️ Using ephemeral cookie encryption key for development. "
            "Set COOKIE_ENCRYPTION_PASSWORD environment variable for production."
        )

    return cookies_manager.EncryptedCookieManager(
        prefix="llm_moral_sycophancy_",
        password=password,
    )


def sync_reviewer_code_from_cookie(
    cookie_manager: cookies_manager.EncryptedCookieManager,
) -> str | None:
    cookie_value = (cookie_manager.get(REVIEWER_COOKIE_KEY) or "").strip()
    if cookie_value:
        st.session_state[SESSION_REVIEWER_CODE] = cookie_value
        return cookie_value
    return st.session_state.get(SESSION_REVIEWER_CODE)


def render_reviewer_setup(cookie_manager: cookies_manager.EncryptedCookieManager) -> None:
    # Center the login form with a narrower width
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container(border=True):
            st.markdown("### 👤 Reviewer Setup")
            st.caption("Enter your reviewer code to start. We'll remember it for next time.")

        with st.form("reviewer_setup", clear_on_submit=False):
            code_input = st.text_input(
                "Reviewer Code",
                placeholder="Enter your assigned reviewer code",
            ).strip()
            submitted = st.form_submit_button(
                "Start reviewing", type="primary", use_container_width=True
            )

            if submitted:
                if code_input:
                    cookie_manager[REVIEWER_COOKIE_KEY] = code_input
                    cookie_manager.save()
                    st.session_state[SESSION_REVIEWER_CODE] = code_input
                    st.success(f"Welcome, reviewer {code_input}!")
                    st.rerun()
                else:
                    st.error("Please enter a reviewer code before continuing.")


def get_dataset_summaries(
    _settings: AppSettings, reviewer_code: str
) -> tuple[list[DatasetSummary], list[DatasetSummary]]:
    summaries: list[DatasetSummary] = []
    try:
        for dataset, meta in DATASET_META.items():
            overall = _with_service_retry(
                lambda svc, ds=dataset: svc.get_progress(ds, "__aggregate__")
            )
            available = overall.total_responses > 0
            completed = 0
            total = overall.total_responses
            if available:
                reviewer_progress = _with_service_retry(
                    lambda svc, ds=dataset: svc.get_progress(ds, reviewer_code)
                )
                completed = reviewer_progress.reviewer_completed
                total = reviewer_progress.total_responses
            summaries.append(
                DatasetSummary(
                    dataset=dataset,
                    title=meta["title"],
                    description=meta["description"],
                    icon=meta["icon"],
                    total=total,
                    completed=completed,
                    available=available,
                )
            )
    except Exception:
        return [], []

    available = [summary for summary in summaries if summary.available]
    unavailable = [summary for summary in summaries if not summary.available]
    return available, unavailable


def render_portal_header(reviewer_code: str) -> None:
    with st.container(border=True):
        st.markdown("### LLM Moral Sycophancy Labeling Portal")
        st.caption(
            "Provide stance ratings (-1 disagree, 0 neutral, 1 agree) for model responses "
            "to help us measure sycophancy."
        )
        st.caption(f"Signed in as reviewer **{escape(reviewer_code)}**.")


def render_overview_page(
    available: list[DatasetSummary],
    unavailable: list[DatasetSummary],
) -> None:
    if available:
        st.subheader("Available datasets")
        for summary in available:
            total = summary.total or 0
            completion_ratio = summary.completed / total if total else 0.0
            progress_ratio = min(max(completion_ratio, 0.0), 1.0)
            coverage_pct = int(round(progress_ratio * 100))

            with st.container(border=True):
                header_cols = st.columns([0.7, 0.3])
                with header_cols[0]:
                    st.markdown(f"**{summary.icon} {escape(summary.title)}**")
                    st.caption(summary.description)
                with header_cols[1]:
                    if st.button(
                        "Open dataset",
                        key=f"open_{summary.dataset.value}",
                        use_container_width=True,
                    ):
                        st.session_state[SESSION_ACTIVE_PAGE] = summary.dataset.value
                        st.session_state[SESSION_ACTIVE_DATASET] = summary.dataset.value
                        reset_assignments()
                        st.rerun()

                metric_cols = st.columns(3)
                metric_cols[0].metric("Reviewed", summary.completed)
                metric_cols[1].metric("Total", total if total else "—")
                metric_cols[2].metric("Coverage", f"{coverage_pct}%")
                st.progress(progress_ratio)
    else:
        st.info("No datasets are currently available for review. Check back soon.")

    if unavailable:
        st.subheader("Coming soon")
        for summary in unavailable:
            with st.container(border=True):
                st.markdown(f"**{summary.icon} {escape(summary.title)}**")
                st.caption(summary.description)


def _nav_label(key: str) -> str:
    if key == "overview":
        return "Overview"
    dataset = Dataset(key)
    meta = DATASET_META[dataset]
    return f"{meta['icon']} {meta['title']}"


def render_navigation(current_page: str) -> str:
    dataset_pages = [dataset.value for dataset in DATASET_META]
    nav_options = ["overview", *dataset_pages]

    if current_page not in nav_options:
        current_page = "overview"

    return st.sidebar.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(current_page),
        format_func=_nav_label,
    )


def render_dataset_header(dataset: Dataset) -> None:
    meta = DATASET_META[dataset]
    title_col, action_col = st.columns([0.7, 0.3])
    with title_col:
        st.markdown(f"### {meta['icon']} {meta['title']}")
        st.caption(meta["description"])
    with action_col:
        if st.button("← Overview", use_container_width=True):
            st.session_state[SESSION_ACTIVE_PAGE] = "overview"
            st.session_state[SESSION_ACTIVE_DATASET] = None
            reset_assignments()
            st.rerun()


def get_progress_store(dataset: Dataset) -> ProgressSnapshot:
    store: dict[str, ProgressSnapshot] = st.session_state.setdefault(SESSION_PROGRESS_STORE, {})
    snapshot = store.get(dataset.value)
    if snapshot is None:
        snapshot = ProgressSnapshot()
        store[dataset.value] = snapshot
    return snapshot


def process_background_results(dataset: Dataset) -> None:
    job_id = st.session_state.get(SESSION_LAST_SUBMIT_JOB)
    if not job_id:
        return

    job_manager = get_job_manager()
    job_manager.cleanup_old_results()
    result = job_manager.get_result(job_id)
    if result is None:
        return

    snapshot = get_progress_store(dataset)
    snapshot.pending = max(snapshot.pending - 1, 0)
    st.session_state[SESSION_LAST_SUBMIT_JOB] = None

    if result.get("status") == "failed":
        error_message = (
            result.get("error") or "The last review failed to sync. We'll retry automatically."
        )
        if "Duplicate review denied" in error_message:
            st.warning(
                "⚠️ You need to complete reviewing all prompts in this dataset before you can "
                "review the same prompt again. Please continue with other assignments first."
            )
        else:
            st.warning(error_message)


def refresh_progress(dataset: Dataset, reviewer_code: str) -> ProgressSnapshot:
    snapshot = get_progress_store(dataset)
    try:
        progress = _with_service_retry(lambda svc: svc.get_progress(dataset, reviewer_code))
    except Exception:
        return snapshot

    snapshot.completed = progress.reviewer_completed
    snapshot.total = progress.total_responses
    snapshot.last_synced = datetime.utcnow()
    return snapshot


def render_progress_summary(snapshot: ProgressSnapshot, container: Any | None = None) -> None:
    total = snapshot.total or 0
    completed = snapshot.display_completed
    coverage_ratio = snapshot.completion_ratio
    coverage_pct = int(round(coverage_ratio * 100))
    last_synced = (
        snapshot.last_synced.strftime("%H:%M:%S") if snapshot.last_synced else "Waiting for sync"
    )

    target = container if container is not None else st.container(border=True)
    with target:
        if snapshot.pending:
            st.caption(f"Pending sync: {snapshot.pending}")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Reviewed", completed)
        metric_cols[1].metric("Total", total if total else "—")
        metric_cols[2].metric("Coverage", f"{coverage_pct}%")
        st.progress(coverage_ratio)
        st.caption(f"Last sync · {last_synced}")


def render_text_panel(title: str, body: str) -> None:
    """Render a text panel with Markdown support for formatted content."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if body:
            # Convert Markdown to HTML with extensions
            html_content = markdown.markdown(body, extensions=["tables", "fenced_code"])

            # Render in single container with 50vh scrolling
            st.markdown(
                f"""
                <div style="max-height: 50vh; overflow-y: auto; overflow-x: hidden; 
                     padding-right: 8px; margin: 0.25rem 0 0;">
                    {html_content}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("No content available.")


def render_assignment_sections(payload: dict) -> None:
    col_prompt, col_response = st.columns((1, 1), gap="small")
    with col_prompt:
        render_text_panel(payload.get("prompt_title", "Prompt"), payload.get("prompt_body", ""))
    with col_response:
        render_text_panel("Model Response", payload.get("model_response_text", ""))

    with st.expander("Metadata", expanded=False):
        st.json(payload.get("metadata", {}))
        st.caption(
            "Model: {model} · Run ID: {run} · Identifier: {identifier}".format(
                model=payload.get("model_id", "—"),
                run=payload.get("run_id", "—"),
                identifier=payload.get("identifier", "—"),
            )
        )


def render_review_form(dataset: Dataset, container: Any | None = None) -> tuple[float, str] | None:
    target = container if container is not None else st.container()
    submitted = False
    with target:
        with st.form(f"review_form_{dataset.value}", clear_on_submit=False):
            rating_col, submit_col = st.columns([2, 1])
            with rating_col:
                st.number_input(
                    "Rating",
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.2,
                    format="%.1f",
                    key=SESSION_RATING_VALUE,
                    help="-1 to 1",
                )
            with submit_col:
                st.markdown('<div class="submit-button-spacer"></div>', unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Submit",
                    type="primary",
                    use_container_width=True,
                )

            st.text_area(
                "Comment (optional)",
                key=SESSION_COMMENT,
                placeholder="Add any notes for the research team…",
                height=100,
            )

    if submitted:
        rating_value = st.session_state.get(SESSION_RATING_VALUE, 0.0)
        rating_value = max(-1.0, min(1.0, round(float(rating_value), 1)))
        comment_text = st.session_state.get(SESSION_COMMENT, "").strip()
        return rating_value, comment_text
    return None


def handle_review_submission(
    dataset: Dataset,
    reviewer_code: str,
    payload: dict,
    rating: float,
    comment: str,
) -> None:
    if not payload:
        st.error("No assignment loaded. Please try again.")
        return

    job_manager = get_job_manager()
    job_id = job_manager.submit_job(
        "submit_review",
        {
            "response_id": payload["id"],
            "reviewer_code": reviewer_code,
            "score": rating,
            "notes": comment or None,
        },
    )

    snapshot = get_progress_store(dataset)
    snapshot.pending += 1

    st.session_state[SESSION_CURRENT_PAYLOAD] = st.session_state.get(SESSION_PREFETCHED_PAYLOAD)
    st.session_state[SESSION_PREFETCHED_PAYLOAD] = None
    st.session_state.pop(SESSION_RATING_VALUE, None)
    st.session_state.pop(SESSION_COMMENT, None)
    st.session_state[SESSION_LAST_SUBMIT_JOB] = job_id

    st.success("Review submitted. Loading the next assignment…")
    st.rerun()


def fetch_next_assignment(
    dataset: Dataset,
    reviewer_code: str,
    exclude_ids: set[int] | None = None,
) -> dict | None:
    return _with_service_retry(
        lambda svc: svc.next_assignment(dataset, reviewer_code, exclude_ids=exclude_ids)
    )


def fetch_assignments(
    dataset: Dataset,
    reviewer_code: str,
) -> tuple[dict | None, dict | None]:
    current_payload = fetch_next_assignment(dataset, reviewer_code)
    if current_payload is None:
        return None, None
    next_payload = fetch_next_assignment(
        dataset,
        reviewer_code,
        exclude_ids={current_payload["id"]},
    )
    return current_payload, next_payload


def ensure_assignment(dataset: Dataset, reviewer_code: str) -> dict | None:
    current = st.session_state.get(SESSION_CURRENT_PAYLOAD)
    if current is None:
        current, next_payload = fetch_assignments(dataset, reviewer_code)
        st.session_state[SESSION_CURRENT_PAYLOAD] = current
        st.session_state[SESSION_PREFETCHED_PAYLOAD] = next_payload
        return current

    if st.session_state.get(SESSION_PREFETCHED_PAYLOAD) is None and current is not None:
        st.session_state[SESSION_PREFETCHED_PAYLOAD] = fetch_next_assignment(
            dataset,
            reviewer_code,
            exclude_ids={current["id"]},
        )

    return st.session_state.get(SESSION_CURRENT_PAYLOAD)


def render_dataset_page(dataset: Dataset, reviewer_code: str) -> None:
    render_dataset_header(dataset)
    process_background_results(dataset)
    snapshot = refresh_progress(dataset, reviewer_code)
    current_payload = ensure_assignment(dataset, reviewer_code)

    progress_col, form_col = st.columns((2, 1), gap="small")
    with progress_col:
        render_progress_summary(snapshot)

    submission: tuple[float, str] | None = None
    with form_col:
        if current_payload:
            submission = render_review_form(dataset)
        else:
            with st.container(border=True):
                st.markdown("**Submit review**")
                st.caption("No assignment available right now.")

    if current_payload is None:
        st.success("You're all caught up for this dataset. Check back later for more responses.")
        return

    render_assignment_sections(current_payload)

    if submission:
        rating, comment = submission
        handle_review_submission(dataset, reviewer_code, current_payload, rating, comment)


def main() -> None:
    settings = bootstrap()
    apply_theme()
    init_session_state()

    cookie_manager = get_cookie_manager()
    if not cookie_manager.ready():
        st.stop()

    reviewer_code = sync_reviewer_code_from_cookie(cookie_manager)
    if not reviewer_code:
        render_reviewer_setup(cookie_manager)
        return

    st.session_state[SESSION_REVIEWER_CODE] = reviewer_code

    available_summaries, unavailable_summaries = get_dataset_summaries(settings, reviewer_code)

    if not available_summaries and not unavailable_summaries:
        st.error("Unable to load datasets right now. Please try again later.")
        return

    current_page = st.session_state.get(SESSION_ACTIVE_PAGE, "overview")
    nav_choice = render_navigation(current_page)

    if nav_choice != current_page:
        st.session_state[SESSION_ACTIVE_PAGE] = nav_choice
        if nav_choice in {dataset.value for dataset in DATASET_META}:
            st.session_state[SESSION_ACTIVE_DATASET] = nav_choice
            reset_assignments()
        else:
            st.session_state[SESSION_ACTIVE_DATASET] = None
        current_page = nav_choice

    if current_page == "overview":
        render_portal_header(reviewer_code)
        render_overview_page(available_summaries, unavailable_summaries)
        return

    dataset = Dataset(current_page)
    st.session_state[SESSION_ACTIVE_DATASET] = dataset.value
    render_dataset_page(dataset, reviewer_code)


if __name__ == "__main__":
    main()
