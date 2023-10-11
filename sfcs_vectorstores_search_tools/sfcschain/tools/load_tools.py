"""Load tools."""
import warnings
from typing import Any, Dict, List, Optional, Callable, Tuple
from mypy_extensions import Arg, KwArg
from langchain.agents.tools import Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks

from sfcschain.tools.base import BaseTool, StructuredTool, tool
from sfcschain.tools.sfcs_duty_search.tool import SfcsDutySearch
from sfcschain.tools.sfcs_sitemap_search.tool import SfcsSiteMapSearch

from langchain.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)

from sfcschain.utilities.sfcs_duty_search import SfcsDutySearchWrapper
from sfcschain.utilities.sfcs_sitemap_search import SfcsSiteMapSearchWrapper


def _get_sfcs_duty_search(**kwargs: Any) -> BaseTool:
    return SfcsDutySearch(search_wrapper=SfcsDutySearchWrapper(**kwargs))


def _get_sfcs_sitemap_search(**kwargs: Any) -> BaseTool:
    return SfcsSiteMapSearch(search_wrapper=SfcsSiteMapSearchWrapper(**kwargs))


_LLM_TOOLS: Dict[str, Tuple[Callable[[KwArg(Any)], BaseTool], List[str]]] = {
    "sfcs_duty_search": (_get_sfcs_duty_search, ["embedding_url"]),
    "sfcs_sitemap_search": (_get_sfcs_sitemap_search, ["embedding_url"]),
}


def _handle_callbacks(
    callback_manager: Optional[BaseCallbackManager], callbacks: Callbacks
) -> Callbacks:
    if callback_manager is not None:
        warnings.warn(
            "callback_manager is deprecated. Please use callbacks instead.",
            DeprecationWarning,
        )
        if callbacks is not None:
            raise ValueError(
                "Cannot specify both callback_manager and callbacks arguments."
            )
        return callback_manager
    return callbacks


def load_tools(
    tool_names: List[str],
    llm: Optional[BaseLanguageModel] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> List[BaseTool]:
    """Load tools based on their name.

    Args:
        tool_names: name of the tool to load.
        llm: language model to use.
        callbacks: Optional callback manager or list of callback handlers.
            If not provided, default global callback manager will be used.
    Returns:
        List of tools.
    """
    tools = []
    callbacks = _handle_callbacks(
        callback_manager=kwargs.get("callback_manager"), callbacks=callbacks
    )
    for name in tool_names:
        if name == "requests":
            warnings.warn(
                "tool name `requests` is deprecated - "
                "please use `requests_all` or specify the requests method"
            )

        if name == "requests_all":
            # expand requests into various methods
            requests_method_tools = [_tool for _tool in _LLM_TOOLS]
            tool_names.extend(requests_method_tools)
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            tool = _LLM_TOOLS[name](llm)
            tools.append(tool)
        else:
            raise ValueError(f"Got unknown tool {name}")
    if callbacks is not None:
        for tool in tools:
            tool.callbacks = callbacks
    return tools


def get_all_tool_names() -> List[str]:
    """Get all tool names."""
    return list(_LLM_TOOLS)
