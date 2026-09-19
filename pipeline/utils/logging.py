import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

def get_logger(name: str = "pipeline") -> logging.Logger:
    """Logger under the `pipeline` namespace; the rich handler is attached once, to `pipeline` itself."""
    root = logging.getLogger("pipeline")
    if not root.handlers:
        root.addHandler(RichHandler(show_path=False, markup=True, log_time_format="%Y-%m-%d \n%H:%M:%S"))
        root.setLevel(logging.INFO)
    return logging.getLogger(name)
    
def log_cfg(cfg: dict[str, any]) -> None:
    """Print the configuration as a compact, colorized block."""
    console.print("[bold cyan]Configuration:[/]")

    for section, values in cfg.items():
        if section.startswith("_"):
            continue

        console.print(f"  [bold yellow]{section}:[/]")

        if isinstance(values, dict):
            for key, value in values.items():
                console.print(
                    f"    [bold green]{key}[/] = {value!r}"
                )
        else:
            console.print(
                f"  [bold green]{section}[/] = {values!r}"
            )