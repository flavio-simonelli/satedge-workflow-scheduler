"""Writing of a figure to a file, with the matplotlib canvas of the image formats."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save(figure: Figure, path: str | Path, *, dpi: int = 200) -> Path:
    """Write ``figure`` to ``path``; the format follows the extension. Returns the path."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(file, dpi=dpi, bbox_inches="tight")
    return file
