import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.traceback import install


class UIHandler:
    """Handles Rich terminal output and progress tracking.

    Ensures that the interface remains premium and visually consistent with
    modern development tools.
    """

    def __init__(self) -> None:
        self.console = Console()
        # Install rich traceback for better error reporting
        install(show_locals=True, console=self.console)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configures standard logging to use RichHandler."""
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, console=self.console)],
        )

    def print_panel(self, message: str, title: str, style: str = "white") -> None:
        """Prints a themed panel message."""
        self.console.print(Panel(message, title=title, style=style, expand=False))

    def print_stats_table(self, title: str, stats: dict[str, int]) -> None:
        """Prints a table of dataset statistics."""
        table = Table(title=title, box=None)
        table.add_column("Split", style="cyan", header_style="bold cyan")
        table.add_column("Count", style="magenta", header_style="bold magenta")
        for key, value in stats.items():
            table.add_row(key, str(value))
        self.console.print(table)

    def print_classification_report(self, title: str, report_dict: dict[str, Any]) -> None:
        """Prints a formatted classification report table."""
        table = Table(title=title)
        table.add_column("Class", style="cyan", header_style="bold cyan")
        table.add_column("Precision", style="magenta")
        table.add_column("Recall", style="magenta")
        table.add_column("F1-Score", style="magenta")
        table.add_column("Support", style="magenta")

        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                table.add_row(
                    label,
                    f"{metrics.get('precision', 0.0):.2f}",
                    f"{metrics.get('recall', 0.0):.2f}",
                    f"{metrics.get('f1-score', 0.0):.2f}",
                    str(int(metrics.get("support", 0.0))),
                )
            elif label == "accuracy":
                acc = float(metrics)
                table.add_row("accuracy", "", "", f"{acc:.2f}", "")

        self.console.print(table)

    def get_progress_bar(self) -> Progress:
        """Returns a configured Progress bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            console=self.console,
        )

    def log(self, message: str) -> None:
        """Logs a message to the console using standard logging."""
        logging.info(message)
