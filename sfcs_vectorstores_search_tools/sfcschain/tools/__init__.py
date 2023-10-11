from typing import Any
from langchain.tools.base import BaseTool, StructuredTool, Tool, tool


def _import_sfcs_duty_search() -> Any:
    from sfcschain.tools.sfcs_duty_search.tool import SfcsDutySearch

    return SfcsDutySearch


def _import_sfcs_sitemap_search() -> Any:
    from sfcschain.tools.sfcs_sitemap_search.tool import SfcsSiteMapSearch

    return SfcsSiteMapSearch


def __getattr__(name: str) -> Any:
    if name == "SfcsDutySearch":
        return _import_sfcs_duty_search()
    elif name == "SfcsSiteMapSearch":
        return _import_sfcs_sitemap_search()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["SfcsDutySearch", "SfcsSiteMapSearch"]
