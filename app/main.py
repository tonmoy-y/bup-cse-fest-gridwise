import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.directives.normalizer import normalize_directives
from app.llm.failover import run_interpretation
from app.llm.provider import LLMProviderError
from app.optimizer.solver import OptimizationError, solve_schedule
from app.schemas import OptimizeRequest
from app.validation.schedule_validator import validate_schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM-Assisted Energy Optimization API")


@app.get("/health")
def health():
    return {"status": "ok"}


def _redact(message: str) -> str:
    """Strip anything that could resemble a key/token before it reaches a response."""
    import re

    message = re.sub(r"key=[^&\s\"]+", "key=REDACTED", message)
    message = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "REDACTED", message)
    return message[:400]


def _build_plan_summary(interpretations, hourly) -> str:
    applied = [i for i in interpretations if i.applies]
    if not applied:
        return "No operator notes affected the schedule; optimized purely on cost."
    kinds = ", ".join(sorted({i.directive_type for i in applied}))
    return f"Applied {len(applied)} operator directive(s) ({kinds}) and minimized grid cost."


@app.post("/optimize-energy")
async def optimize_energy(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Malformed JSON body"})

    try:
        req = OptimizeRequest.model_validate(body)
    except ValidationError as exc:
        # exc.errors() embeds raw exception objects in "ctx" for custom validators,
        # which JSONResponse cannot serialize; exc.errors(include_url=False) still
        # contains them, so route through the safely-serialized json() form instead.
        details = json.loads(exc.json())
        return JSONResponse(
            status_code=400,
            content={"error": "Request does not match the required schema", "details": details},
        )

    try:
        failover_result = run_interpretation(req.operator_notes, req.battery.capacity_kwh)
    except LLMProviderError as exc:
        logger.error("All LLM providers failed: %s", exc)
        reason = _redact(str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "The interpretation service is temporarily unavailable",
                "reason": reason,
            },
        )
    except Exception:
        logger.exception("Unexpected interpreter failure")
        return JSONResponse(status_code=500, content={"error": "Internal interpretation error"})

    validated = failover_result.validated
    constraints = normalize_directives(validated, req.battery.minimum_energy_kwh)

    try:
        result = solve_schedule(req.hours, req.battery, constraints)
    except OptimizationError as exc:
        logger.error("Optimization failure: %s", exc)
        return JSONResponse(status_code=500, content={"error": "No feasible schedule could be computed"})
    except Exception:
        logger.exception("Unexpected optimizer failure")
        return JSONResponse(status_code=500, content={"error": "Internal optimization error"})

    outcome = validate_schedule(result.hourly, req.hours, req.battery, constraints)
    if not outcome.valid:
        logger.error("Final schedule failed validation: %s", outcome.errors)
        return JSONResponse(status_code=500, content={"error": "Computed schedule failed validation"})

    directive_interpretation = [
        {
            "note_index": v.note_index,
            "applies": v.applies,
            "directive_type": v.directive_type,
            "structured_adjustment": v.structured_adjustment,
            "explanation": v.explanation,
        }
        for v in validated
    ]

    hourly_plan = [
        {
            "hour": h.hour,
            "grid_kwh": h.grid_kwh,
            "solar_used_kwh": h.solar_used_kwh,
            "battery_action": h.battery_action,
            "battery_kwh": h.battery_kwh,
            "battery_energy_after_kwh": h.battery_energy_after_kwh,
        }
        for h in result.hourly
    ]

    response_body = {
        "scenario_id": req.scenario_id,
        "directive_interpretation": directive_interpretation,
        "hourly_plan": hourly_plan,
        "total_grid_kwh": result.total_grid_kwh,
        "total_cost_bdt": result.total_cost_bdt,
        "peak_grid_kwh": result.peak_grid_kwh,
        "plan_summary": _build_plan_summary(validated, hourly_plan),
    }

    return JSONResponse(status_code=200, content=response_body)
