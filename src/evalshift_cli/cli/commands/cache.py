"""Implementation of ``evalshift cache clear``.

Tiny utility that deletes every row from this project's SQLite cache
(``.evalshift/cache.db`` in the current folder). We
group cache subcommands under a Typer sub-app so future operations
(``cache info``, ``cache prune``, …) slot in naturally without
polluting the top-level command list.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from evalshift_cli.cache import schema
from evalshift_cli.cache.store import CacheStore

cache_app = typer.Typer(
    name="cache",
    help="Manage this project's LLM-response cache.",
    no_args_is_help=True,
    add_completion=False,
)


@cache_app.command(name="clear")
def cache_clear() -> None:
    """Remove every entry from this project's cache (.evalshift/cache.db)."""
    console = Console()
    removed = asyncio.run(_clear())
    console.print(
        f"[green]✓[/green] cleared {removed} cached call(s) from {schema.DEFAULT_CACHE_PATH}"
    )
    legacy = schema.legacy_cache_path()
    if legacy.is_file():
        console.print(
            f"[dim]{legacy} is the old machine-wide cache and is no longer used; "
            "delete it to free the space.[/dim]"
        )


async def _clear() -> int:
    store = await CacheStore.open()
    try:
        return await store.clear()
    finally:
        await store.close()


__all__ = ["cache_app"]
