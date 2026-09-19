import logging

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()

def get_logger(name: str = "pipeline") -> logging.Logger:
    """Logger under the `pipeline` namespace; the rich handler is attached once, to `pipeline` itself."""
    root = logging.getLogger("pipeline")
    if not root.handlers:
        root.addHandler(RichHandler(show_path=False, markup=True, log_time_format="%Y-%m-%d \n%H:%M:%S"))
        root.setLevel(logging.INFO)
    return logging.getLogger(name)


def log_table(title: str, columns: list[str], rows: list[list], caption: str | None = None, styles: list[str | None] | None = None, left: bool = False) -> None:
    """Print a table: the first column is left aligned, the others right aligned (all left with `left`).
    `styles` colors each column."""
    table = Table(title=title, caption=caption, box=box.SIMPLE_HEAVY, header_style="bold cyan", title_style="bold")
    for i, column in enumerate(columns):
        table.add_column(column, justify="left" if left or i == 0 else "right", style=styles[i] if styles else None)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def log_cfg(cfg: dict[str, any]) -> None:
    """Print the configuration as a table of section / key / value."""
    rows = []
    for section, values in cfg.items():
        if section.startswith("_"):
            continue
        items = values.items() if isinstance(values, dict) else [("", values)]
        for i, (key, value) in enumerate(items):
            rows.append([section if i == 0 else "", key, repr(value)])
    log_table("Configuration", ["section", "key", "value"], rows, styles=["bold yellow", "green", "cyan"], left=True)


def log_dataset(summary: dict, splits: list[dict], title: str = "Dataset") -> None:
    """Totals of the dataset, then one row per split."""
    log_table(title, ["", "value"], [[k, f"{v:,}" if isinstance(v, int) else f"{v:.4f}"] for k, v in summary.items()])
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    log_table(
        "Split",
        ["split", "interactions", "share", "users", "items", "target seen"],
        [[r["split"], f"{r['interactions']:,}", pct(r["share"]), f"{r['users']:,}", f"{r['items']:,}", pct(r["target seen"])] for r in splits],
        caption="target seen: share of val/test items already in the history the model sees",
    )


def log_metrics(results: dict[str, dict[str, float]], metrics: list[str], ks: list[int], title: str = "Metrics", caption: str | None = None) -> None:
    """One row per metric, one column per split and cutoff, with a divider between the name and each split:
    results = {"val": {"ndcg@10": ...}, "test": {...}}."""
    colors = ["cyan", "magenta", "yellow", "green"]
    table = Table(title=title, caption=caption, box=box.SIMPLE_HEAVY, header_style="bold", title_style="bold")
    table.add_column("metric", style="bold")
    for color, split in zip(colors, results):
        table.add_column("│", justify="center", style="dim", width=1)
        for k in ks:
            table.add_column(f"{split}@{k}", justify="right", style=color, header_style=f"bold {color}")
    for m in metrics:
        row = [m]
        for split in results:
            row += ["│"] + [f"{results[split][f'{m}@{k}']:.4f}" for k in ks]
        table.add_row(*row)
    console.print(table)
