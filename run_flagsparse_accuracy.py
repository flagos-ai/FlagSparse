#!/usr/bin/env python3
"""Run FlagSparse accuracy tests per operator."""

from run_flagsparse_pytest import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_phase="accuracy",
            expose_phase_arg=False,
            description=__doc__,
            include_accuracy_args=True,
            include_performance_args=False,
        )
    )
