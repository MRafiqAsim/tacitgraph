"""Prompt loading, domain-context injection and safe formatting."""

from tacitgraph import prompt_loader as pl
from tacitgraph.prompt_loader import format_prompt, get_prompt


def test_system_prompts_resolve_domain_context():
    prompt = get_prompt("retrieval", "generation", "system_prompt")
    assert "{domain_context}" not in prompt
    assert pl.get_domain_context() in prompt


def test_domain_context_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv(pl.DOMAIN_CONTEXT_ENV, "You analyse a law firm's case archive.")
    prompt = get_prompt("retrieval", "generation", "system_prompt")
    assert prompt.startswith("You analyse a law firm's case archive.")


def test_non_string_values_pass_through():
    assert isinstance(get_prompt("silver", "pii_detection", "temperature"), (int, float))


def test_missing_prompt_returns_default():
    assert get_prompt("silver", "does_not_exist", "system_prompt", default="x") == "x"


def test_get_section_resolves_every_value():
    section = pl.get_section("retrieval", "generation")
    assert "{domain_context}" not in section["system_prompt"]


def test_format_prompt_keeps_literal_json_braces():
    template = 'Return {"entities": []} for {query}'
    assert format_prompt(template, query="Q") == 'Return {"entities": []} for Q'


def test_format_prompt_ignores_unknown_placeholders():
    assert format_prompt("{known} and {unknown}", known="a") == "a and {unknown}"


def test_every_system_prompt_has_resolved_context():
    data = pl._load()
    for layer, sections in data.items():
        if not isinstance(sections, dict):
            continue
        for name, section in sections.items():
            if isinstance(section, dict) and "system_prompt" in section:
                resolved = get_prompt(layer, name, "system_prompt")
                assert "{domain_context}" not in resolved, f"{layer}.{name}"
