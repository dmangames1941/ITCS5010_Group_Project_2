from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

_history = InMemoryHistory()

_prompt_style = Style.from_dict({
    "prompt_prefix": "#cyan bold#",
    "prompt_symbol": "#white bold#"
})

def get_user_prompt(prompt_label: str = "clia> ") -> str:
    """
    Prompts the user for their next message or command,supporting readline-style history.
    """
    try:
        user_input = prompt(
            [("class:prompt_prefix", "clia"), ("class:prompt_symbol", ">")],
            style=_prompt_style,
            history=_history
        )
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        return "exit"

def ask_confirmation(message: str = "Execute tool?", default: bool = True) -> bool:
    """
    Prompts the user for a [y/N] or [Y/n] confirmation in 'confirm' execution mode.
    """
    hint = "[Y/n]" if default else "[y/N]"
    try:
        response = prompt(f"{message}{hint}: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False