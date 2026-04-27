"""Strategy pool router — controls which templates are visible & in what order."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.strategy_pool_service import (
    delete_entry, list_concepts, list_pool, reorder, update_entry,
)

router = APIRouter(prefix="/api/strategy-pool", tags=["strategy-pool"])


class UpdateRequest(BaseModel):
    display_name: str | None = None
    concept: str | None = None
    description: str | None = None
    is_active: bool | None = None


class ReorderRequest(BaseModel):
    order: list[str]   # ordered template_ids


@router.get("")
def list_all(
    active_only: bool = False,
    concept: str | None = None,
):
    return list_pool(active_only=active_only, concept=concept)


@router.get("/concepts")
def concepts():
    return list_concepts()


@router.patch("/{template_id}")
def update(template_id: str, body: UpdateRequest):
    patch = body.model_dump(exclude_unset=True)
    e = update_entry(template_id, patch)
    if not e:
        raise HTTPException(404, "Strategy not in pool")
    return e


@router.delete("/{template_id}")
def delete(template_id: str):
    ok = delete_entry(template_id)
    if not ok:
        raise HTTPException(404, "Strategy not in pool")
    return {"ok": True}


@router.post("/reorder")
def reorder_pool(body: ReorderRequest):
    return reorder(body.order)
