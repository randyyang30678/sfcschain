from typing import Any, Optional
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from sfcschain.utilities.sfcs_sitemap_search import SfcsSiteMapSearchWrapper


class SfcsSiteMapSearch(BaseTool):
    """Tool that queries the SfcsSiteMapSearch for similar documents."""

    name: str = "sfcs_sitemap_search"
    description: str = "Tool provides a search interface for the SFCS SiteMap."
    search_wrapper: SfcsSiteMapSearchWrapper

    @classmethod
    def from_base_url(
        cls, base_url: str, search_kwargs: Optional[dict] = None, **kwargs: Any
    ) -> "SfcsSiteMapSearch":
        """Create a tool from a embedding base url.
        Returns:
            A tool.
        """
        wrapper = SfcsSiteMapSearchWrapper(
            embedding_url=base_url, search_kwargs=search_kwargs or {}
        )
        return cls(search_wrapper=wrapper, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool."""
        return self.search_wrapper.run(query=query)
