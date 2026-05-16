from __future__ import annotations

from attack.pipeline.runs import run_pts_continuous_init_diagnostic as _impl


globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("__")
    }
)

main = _impl.main
__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    raise SystemExit(main())
