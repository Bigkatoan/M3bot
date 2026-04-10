"""Utilities for vision-based M3bot environments.

Key helper: set_vis_markers_guide_purpose
  Sets all /Visuals USD prims to UsdGeom "guide" render purpose.
  Effect:
    - Viewport (interactive 3D view, LiveStream) — STILL shows markers  ✓
    - CameraSensor / Replicator (gym observation images) — does NOT render them  ✓
  Use as a startup EventTerm in vision env cfgs so the obs camera never sees
  debug helpers (target spheres, frame axes) that don't exist on real hardware.
"""

from __future__ import annotations

import torch


def set_vis_markers_guide_purpose(
    env,  # ManagerBasedRLEnv — not typed to avoid circular import
    env_ids: torch.Tensor,
) -> None:
    """Set all /Visuals prims to USD 'guide' purpose (one-time, at startup).

    USD render purposes:
      - "default" / "render"  → rendered by cameras AND visible in viewport
      - "guide"               → visible in viewport ONLY, skipped by Replicator cameras

    Called as EventTerm(mode="startup") so it runs once after the first reset,
    by which time all VisualizationMarkers have already been created.
    """
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        changed = 0
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith("/Visuals"):
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue
            imageable = UsdGeom.Imageable(prim)
            attr = imageable.GetPurposeAttr()
            if attr.IsValid():
                attr.Set(UsdGeom.Tokens.guide)
            else:
                imageable.CreatePurposeAttr(UsdGeom.Tokens.guide)
            changed += 1

        if changed:
            print(f"[vision_utils] Set {changed} /Visuals prims to 'guide' purpose "
                  "(visible in viewport, hidden from CameraSensor).")
    except Exception as exc:
        # Non-fatal: if omni.usd is unavailable (e.g. unit tests) just skip.
        print(f"[vision_utils] set_vis_markers_guide_purpose skipped: {exc}")
