"""Strategy factory API: run factory, check job status."""
from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import FactoryJob
from app.services.factory_service import run_factory
from app.services.strategy_templates import generate_all_candidates

router = APIRouter(prefix="/api/strategy-factory", tags=["factory"])


class RunFactoryRequest(BaseModel):
    ts_code: str
    cutoff_date: date
    position_ratios: list[int] = [40, 30, 30]


async def _run_factory_bg(ts_code: str, cutoff_date: date, position_ratios: list[int], job_id: str):
    """Background wrapper for factory execution."""
    from app.db import async_session

    async with async_session() as db:
        try:
            await run_factory(db, ts_code, cutoff_date, position_ratios, job_id=job_id)
        except Exception as e:
            job = await db.get(FactoryJob, job_id)
            if job:
                job.status = "failed"
                job.error = str(e)
                await db.commit()


@router.post("/run")
async def run(
    body: RunFactoryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    candidates = generate_all_candidates()

    job = FactoryJob(
        id=job_id,
        ts_code=body.ts_code,
        status="pending",
        total_candidates=len(candidates),
        config={
            "cutoff_date": body.cutoff_date.isoformat(),
            "position_ratios": body.position_ratios,
        },
    )
    db.add(job)
    await db.commit()

    background_tasks.add_task(
        _run_factory_bg, body.ts_code, body.cutoff_date, body.position_ratios, job_id
    )
    return {"job_id": job_id, "status": "pending", "total_candidates": len(candidates)}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(FactoryJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "total_candidates": job.total_candidates,
        "evaluated": job.evaluated,
        "passed": job.passed,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
