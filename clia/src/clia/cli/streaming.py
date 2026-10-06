from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()
class ResponseStreamer:
    """
    Manages live rendering of LLM responses as tokens stream in.
    """
    def __init__(self, console_instance: Optional[Console] = None):
        """
        Args:
            console_instance: Optional Rich Console instance for output.
        """
        self.console = console_instance or console
        self._buffer = ""
        self._live: Optional[Live] = None

    def __enter__(self) -> "ResponseStreamer":
        """
        Starts the Rich Live rendering context.
        """
        self._buffer = ""
        self._live = Live(
            Markdown(""),
            console=self.console,
            refresh_per_second=15,
            vertial_overflow="visible"
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stops the Rich Live rendering context
        """
        if self._live:
            if self._buffer:
                self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None

    def on_token(self, token: str) -> None:
        """
        Callback handler to append new tokens and update the live markdown view.

        Args:
            token: A chunk or token recieved from the LLM provider stream.
        """
        self._buffer += token
        if self._live:
                self._live.update(Markdown(self._buffer))

    def get_accumulated_text(self) -> str:
         """
         Returns the complete response string accumulated during streaming.
         """
         return self._buffer