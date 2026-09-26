#!/usr/bin/env python3
"""Generate all main-text publication figures from saved experiment outputs."""
from __future__ import annotations

from question1.visualization.paper import generate_all as q1
from question2.visualization.paper import generate_all as q2
from question3.visualization.paper import generate_all as q3


def main() -> None:
    q1()
    q2()
    q3()
    print("All publication figures generated under outputs/q*/paper_figures.")


if __name__ == "__main__":
    main()
