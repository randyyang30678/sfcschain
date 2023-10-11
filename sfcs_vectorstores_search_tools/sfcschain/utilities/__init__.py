from typing import Any
from langchain.utilities.requests import Requests, RequestsWrapper, TextRequestsWrapper


def _import_sfcs_duty_search() -> Any:
    from sfcschain.utilities.sfcs_duty_search import SfcsDutySearchWrapper

    return SfcsDutySearchWrapper


def _import_sfcs_sitemap_search() -> Any:
    from sfcschain.utilities.sfcs_sitemap_search import SfcsSiteMapSearchWrapper

    return SfcsSiteMapSearchWrapper


def __getattr__(name: str) -> Any:
    if name == "SfcsDutySearchWrapper":
        return _import_sfcs_duty_search()
    elif name == "SfcsSiteMapSearchWrapper":
        return _import_sfcs_sitemap_search()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "SfcsDutySearchWrapper",
    "SfcsSiteMapSearchWrapper",
]
