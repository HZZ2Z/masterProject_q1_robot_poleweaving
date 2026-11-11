"""Train one isolated SuperMimic profile: soccer or pole-weaving."""

from __future__ import annotations

import argparse
from pathlib import Path

from q1_soccer.profiles import resolve_profile


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("soccer", "pole-weaving"), default="soccer")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    profile = resolve_profile(args.profile, args.config)
    forwarded = ["--config", str(profile.config_path)]
    if args.resume is not None:
        if args.profile == "soccer":
            forwarded.extend(("--resume", str(args.resume)))
        else:
            forwarded.extend(("--resume-checkpoint", str(args.resume)))
    if args.check:
        forwarded.append("--check" if args.profile == "soccer" else "--preflight-only")

    if args.profile == "soccer":
        from q1_soccer.train import main as soccer_main

        return soccer_main(forwarded)

    from pole_task.training import main as pole_main

    return pole_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
