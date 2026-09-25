"""TEST-4 agent behavior with deterministic model responses."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from paper_digest import agent as agent_mod
from paper_digest.store import Chunk, PaperStore

models.ALLOW_MODEL_REQUESTS = False


def _seed(store: PaperStore) -> None:
    store.add_chunks(
        [
            Chunk("old.1", "Old Paper", "2026-08-01", 1, "old paper abstract about needle retrieval"),
            Chunk("old.1", "Old Paper", "2026-08-01", 2, "old paper method section"),
        ],
        pages=2,
    )
    store.add_chunks(
        [
            Chunk("new.1", "New Paper", "2026-09-20", 1, "new paper abstract about needle retrieval"),
            Chunk("new.1", "New Paper", "2026-09-20", 3, "new paper main result on page three"),
        ],
        pages=3,
    )


def _tool_returns(messages: list[ModelMessage]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in messages:
        for part in getattr(m, "parts", []):
            if isinstance(part, ToolReturnPart):
                out.setdefault(part.tool_name, []).append(str(part.content))
    return out


class ScriptedModel:
    """Calls list_papers, search_chunks, read_pages in turn, then emits `final` as the structured output."""

    def __init__(self, final: dict, read_pages_args: dict | None = None):
        self.final = final
        self.read_pages_args = read_pages_args or {"arxiv_id": "new.1", "pages": [1, 3]}
        self.seen: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.seen.append(messages)
        step = len(self.seen)
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("list_papers", {})])
        if step == 2:
            return ModelResponse(parts=[ToolCallPart("search_chunks", {"query": "needle retrieval"})])
        if step == 3:
            return ModelResponse(parts=[ToolCallPart("read_pages", self.read_pages_args)])
        assert info.output_tools, "structured output expected"
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, self.final)])

    @property
    def tool_returns(self) -> dict[str, list[str]]:
        return _tool_returns(self.seen[-1])


ANSWER = {
    "summary": "New Paper reports the result.",
    "findings": [{"claim": "The main result is on page three.", "citations": [{"arxiv_id": "new.1", "title": "New Paper", "pages": [1, 3]}]}],
    "gaps": ["nothing about cooking"],
}


def test_ask_without_since_sees_all_papers(store: PaperStore):
    _seed(store)
    model = ScriptedModel(ANSWER)
    with agent_mod.ask_agent.override(model=FunctionModel(model)):
        a = agent_mod.ask("what is the result?", store)

    assert a.summary == ANSWER["summary"]
    assert a.findings[0].citations[0].arxiv_id == "new.1"
    assert a.findings[0].citations[0].pages == [1, 3]
    assert a.gaps == ["nothing about cooking"]

    ret = model.tool_returns
    assert "old.1 | Old Paper | 2026-08-01 | 2 pages" in ret["list_papers"][0]
    assert "new.1 | New Paper | 2026-09-20 | 3 pages" in ret["list_papers"][0]
    assert "[old.1 |" in ret["search_chunks"][0] and "[new.1 |" in ret["search_chunks"][0]
    read = ret["read_pages"][0]
    assert "[new.1 | New Paper | 2026-09-20 | page 1]" in read
    assert "[new.1 | New Paper | 2026-09-20 | page 3]" in read
    assert "page 2]" not in read


def test_ask_with_since_restricts_tools_to_period(store: PaperStore):
    _seed(store)
    model = ScriptedModel(ANSWER)
    with agent_mod.ask_agent.override(model=FunctionModel(model)):
        agent_mod.ask("what is the result?", store, since=date(2026, 9, 1))

    ret = model.tool_returns
    assert "new.1" in ret["list_papers"][0] and "old.1" not in ret["list_papers"][0]
    assert "[new.1 |" in ret["search_chunks"][0] and "old.1" not in ret["search_chunks"][0]


def test_ask_since_boundary_is_inclusive(store: PaperStore):
    _seed(store)
    model = ScriptedModel(ANSWER)
    with agent_mod.ask_agent.override(model=FunctionModel(model)):
        agent_mod.ask("q", store, since=date(2026, 9, 20))
    assert "new.1" in model.tool_returns["list_papers"][0]


def test_ask_on_empty_library_reports_nothing(store: PaperStore):
    empty = {"summary": "The library is empty.", "findings": [], "gaps": ["no papers indexed"]}
    model = ScriptedModel(empty, read_pages_args={"arxiv_id": "nope", "pages": [1]})
    with agent_mod.ask_agent.override(model=FunctionModel(model)):
        a = agent_mod.ask("anything?", store)

    assert a.findings == []
    assert a.gaps == ["no papers indexed"]
    ret = model.tool_returns
    assert ret["list_papers"] == ["No papers indexed for this period."]
    assert ret["search_chunks"] == ["No results."]
    assert ret["read_pages"] == ["No results."]


def test_finding_requires_citation(store: PaperStore):
    bad = {"summary": "x", "findings": [{"claim": "uncited", "citations": []}]}
    model = ScriptedModel(bad)
    with agent_mod.ask_agent.override(model=FunctionModel(model)):
        with pytest.raises(Exception):  # validation retries exhausted
            agent_mod.ask("q", store)


DIGEST = {
    "period": "since 2026-09-01",
    "themes": ["retrieval", "evaluation"],
    "items": [{
        "arxiv_id": "new.1", "title": "New Paper", "published": "2026-09-20",
        "one_liner": "Does a thing.", "why_it_matters": "It matters.", "key_pages": [1, 3],
    }],
}


def test_digest_passes_since_into_prompt_and_tools(store: PaperStore):
    _seed(store)
    model = ScriptedModel(DIGEST)
    with agent_mod.digest_agent.override(model=FunctionModel(model)):
        d = agent_mod.digest(store, since=date(2026, 9, 1))

    assert d.period == "since 2026-09-01"
    assert [i.arxiv_id for i in d.items] == ["new.1"]
    assert d.items[0].key_pages == [1, 3]

    first_request = model.seen[0][0]
    user_text = " ".join(str(p.content) for p in first_request.parts if hasattr(p, "content"))
    assert "on or after 2026-09-01" in user_text
    assert "Set period to 'since 2026-09-01'" in user_text
    ret = model.tool_returns
    assert "new.1" in ret["list_papers"][0] and "old.1" not in ret["list_papers"][0]


def test_digest_on_empty_library(store: PaperStore):
    empty = {"period": "since 2026-09-01", "themes": [], "items": []}
    model = ScriptedModel(empty)
    with agent_mod.digest_agent.override(model=FunctionModel(model)):
        d = agent_mod.digest(store, since=date(2026, 9, 1))
    assert d.items == [] and d.themes == []
    assert model.tool_returns["list_papers"] == ["No papers indexed for this period."]


def test_fmt_formats_chunks_and_empty():
    assert agent_mod._fmt([]) == "No results."
    out = agent_mod._fmt([Chunk("a", "T", "2026-01-01", 4, "body")])
    assert out == "[a | T | 2026-01-01 | page 4]\nbody"
