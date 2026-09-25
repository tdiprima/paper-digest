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
    a = agent_mod.ask("what is the result?", store, model=FunctionModel(model))

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
    agent_mod.ask("what is the result?", store, since=date(2026, 9, 1), model=FunctionModel(model))

    ret = model.tool_returns
    assert "new.1" in ret["list_papers"][0] and "old.1" not in ret["list_papers"][0]
    assert "[new.1 |" in ret["search_chunks"][0] and "old.1" not in ret["search_chunks"][0]


def test_ask_since_boundary_is_inclusive(store: PaperStore):
    _seed(store)
    model = ScriptedModel(ANSWER)
    agent_mod.ask("q", store, since=date(2026, 9, 20), model=FunctionModel(model))
    assert "new.1" in model.tool_returns["list_papers"][0]


def test_ask_on_empty_library_reports_nothing(store: PaperStore):
    empty = {"summary": "The library is empty.", "findings": [], "gaps": ["no papers indexed"]}
    model = ScriptedModel(empty, read_pages_args={"arxiv_id": "nope", "pages": [1]})
    a = agent_mod.ask("anything?", store, model=FunctionModel(model))

    assert a.findings == []
    assert a.gaps == ["no papers indexed"]
    ret = model.tool_returns
    assert ret["list_papers"] == ["No papers indexed for this period."]
    assert ret["search_chunks"] == ["No results."]
    assert ret["read_pages"] == ["No results."]


def test_finding_requires_citation(store: PaperStore):
    bad = {"summary": "x", "findings": [{"claim": "uncited", "citations": []}]}
    model = ScriptedModel(bad)
    with pytest.raises(Exception):  # validation retries exhausted
        agent_mod.ask("q", store, model=FunctionModel(model))


def test_citation_to_unread_page_is_rejected(store: PaperStore):
    _seed(store)
    bad = {**ANSWER, "findings": [{"claim": "c", "citations": [{"arxiv_id": "new.1", "title": "New Paper", "pages": [2]}]}]}
    model = ScriptedModel(bad)  # reads new.1 pages 1,3 only
    with pytest.raises(Exception):
        agent_mod.ask("q", store, model=FunctionModel(model))
    retry_text = str(model.seen[-1][-1].parts[0].content)
    assert "new.1 p.2" in retry_text


def test_citation_to_unread_paper_is_rejected(store: PaperStore):
    _seed(store)
    bad = {**ANSWER, "findings": [{"claim": "c", "citations": [{"arxiv_id": "ghost.9", "title": "Ghost", "pages": [1]}]}]}
    model = ScriptedModel(bad)
    with pytest.raises(Exception):
        agent_mod.ask("q", store, model=FunctionModel(model))
    assert "ghost.9 p.1" in str(model.seen[-1][-1].parts[0].content)


def test_model_can_fix_citation_after_retry(store: PaperStore):
    _seed(store)
    bad = {**ANSWER, "findings": [{"claim": "c", "citations": [{"arxiv_id": "new.1", "title": "New Paper", "pages": [2]}]}]}
    calls = 0

    def fn(messages, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_pages", {"arxiv_id": "new.1", "pages": [1, 3]})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, bad)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, ANSWER)])

    a = agent_mod.ask("q", store, model=FunctionModel(fn))
    assert calls == 3
    assert a.findings[0].citations[0].pages == [1, 3]


ITEM = {
    "arxiv_id": "new.1", "title": "New Paper", "published": "2026-09-20",
    "one_liner": "Does a thing.", "why_it_matters": "It matters.", "key_pages": [1, 3],
}


class DigestModel:
    """Per-paper runs: read_pages then emit `item`. Themes run: emit `themes` directly."""

    def __init__(self, item: dict, themes: list[str], read_pages: list[int] = (1, 3)):
        self.item, self.themes, self.read_pages = item, themes, list(read_pages)
        self.runs: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        out = info.output_tools[0].name
        first = messages[0].parts[-1].content
        if first.startswith("Here are one-line summaries"):
            self.runs.append(messages)
            return ModelResponse(parts=[ToolCallPart(out, {"themes": self.themes})])
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("read_pages", {"arxiv_id": self.item["arxiv_id"], "pages": self.read_pages})])
        self.runs.append(messages)
        return ModelResponse(parts=[ToolCallPart(out, self.item)])


def test_digest_runs_per_paper_then_synthesizes_themes(store: PaperStore):
    _seed(store)
    model = DigestModel(ITEM, ["retrieval", "evaluation"])
    d = agent_mod.digest(store, since=date(2026, 9, 1), model=FunctionModel(model))

    assert d.period == "since 2026-09-01"
    assert d.themes == ["retrieval", "evaluation"]
    assert [i.arxiv_id for i in d.items] == ["new.1"]  # old.1 excluded by since
    assert d.items[0].key_pages == [1, 3]

    item_run, themes_run = model.runs
    assert "paper new.1" in item_run[0].parts[-1].content
    assert "page 1]" in _tool_returns(item_run)["read_pages"][0]
    assert "new.1 | New Paper: Does a thing." in themes_run[0].parts[-1].content


def test_digest_item_with_unread_key_pages_is_rejected(store: PaperStore):
    _seed(store)
    model = DigestModel({**ITEM, "key_pages": [2]}, ["t"])
    with pytest.raises(Exception):
        agent_mod.digest(store, since=date(2026, 9, 1), model=FunctionModel(model))


def test_digest_on_empty_library_makes_no_model_calls(store: PaperStore):
    def fn(messages, info):
        raise AssertionError("model should not be called")

    d = agent_mod.digest(store, since=date(2026, 9, 1), model=FunctionModel(fn))
    assert d == agent_mod.Digest(period="since 2026-09-01", themes=[], items=[])


def test_fmt_formats_chunks_and_empty():
    assert agent_mod._fmt([]) == "No results."
    out = agent_mod._fmt([Chunk("a", "T", "2026-01-01", 4, "body")])
    assert out == "[a | T | 2026-01-01 | page 4]\nbody"
