# temp-strands — MP speech profiles

A [Strands Agents](https://strandsagents.com) system that takes the name of an Australian federal
(House of Representatives, Senate) or Victorian (Legislative Assembly, Legislative Council) member and
summarises **what they said in Parliament across their whole career** on four topics: housing,
healthcare, economy and education. Every claim carries a verbatim quote, a Hansard citation, the date,
chamber and the capacity the member spoke in (personal, ministerial, shadow, committee, presiding).

It reports what the member said. It does not infer their beliefs or score their positions.

## Architecture

```
offline ingestion                                  request path (Strands Graph)
-----------------                                  ----------------------------
OpenAustralia XML (Cth Hansard) ─┐                 name ─► resolve (deterministic, may ask to disambiguate)
APH Parliamentary Handbook ──────┤                        │
Vic Parliament search + Hansard ─┼─► parse ─► attribute   plan: career timeline → periods per parliament
                                 │   ─► topic-tag         ├─► housing ┐
manual crosswalk (Cth ↔ Vic) ────┘   ─► SQLite + FTS5     ├─► healthcare ├─ per-period map (cached) → career reduce
                                                         ├─► economy  │
                                                         └─► education┘
                                                                 └─► verify: quote ⊂ own words, speaker,
                                                                     membership date, topic → cited profile
```

| Piece | Module |
|---|---|
| Domain and output models (`MPProfile`, `Claim`, `HansardRef`, `Disambiguation`) | `src/mp_profile/models.py` |
| Source adapters: OpenAustralia, Handbook, Victorian Hansard | `src/mp_profile/sources/` |
| Resumable ingestion with coverage records | `src/mp_profile/ingest.py` |
| Speech store (SQLite + FTS5, topic tags, coverage, summary cache) | `src/mp_profile/store.py` |
| Person resolution and federal↔Victorian crosswalk | `src/mp_profile/resolve.py`, `crosswalk.py` |
| Strands `GraphBuilder` workflow: plan → 4 parallel analysts → verify | `src/mp_profile/pipeline/` |
| Summarisers: extractive (no model) or Strands/Bedrock structured output | `src/mp_profile/pipeline/summarizers.py` |
| Parliament MCP server (MCP SDK v2 `MCPServer`) | `src/mp_profile/mcp_server.py` |
| Conversational follow-up agent over the MCP server | `src/mp_profile/chat.py` |
| Telemetry: Strands OTel traces/metrics + domain metrics + hooks | `src/mp_profile/telemetry.py` |
| Evals: deterministic checks, golden set via `strands-agents-evals`, LLM judges, online scoring | `src/mp_profile/evals/` |

Design rules:

- Fetching, parsing, attribution, career timelines, citations and quote checks are plain code. LLMs only
  write period summaries and career summaries, from passages they are given.
- Interjections are dropped at parse time. Material the member reads out or quotes (motions, quotes of
  other members) is kept but prefixed with `> ` and never counts as their own words.
- Federal and Victorian records are only merged through an explicit crosswalk CSV with evidence
  (`mp-profile crosswalk suggest` proposes candidates; a human confirms).
- A failed topic analyst gives a `partial` profile with a gap note rather than failing the request.
- Thin evidence produces gaps and low confidence, not filler.

## Quick start (no credentials needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
mp-profile load-sample                 # fictional members, parsed by the real adapters
mp-profile profile "Alex Morgan"        # served federally and in Victoria
mp-profile profile "Taylor"             # ambiguous → candidate list
mp-profile profile "Sam Taylor" --json
```

The bundled sample data is fictional (see `src/mp_profile/sample_data/README.md`) so tests never put
words in a real member's mouth.

## Real data

```bash
mp-profile ingest people --handbook                           # OpenAustralia + Vic members, APH PHIDs
mp-profile ingest federal  --start 2022-07-01 --end 2022-12-31
mp-profile ingest victoria --start 2022-07-01 --end 2022-12-31
mp-profile crosswalk suggest > crosswalk-candidates.csv       # review, then:
mp-profile crosswalk apply crosswalk.csv
mp-profile coverage
```

Ingestion is resumable (sittings already stored are skipped unless `--force`), rate-limited, retries on
429/5xx and caches raw responses under `data/raw/`. Coverage notes in each profile say which parts of a
career are not in the store.

Known limits of this first version:

- Victorian Hansard is parsed from the HTML published since late 2018. Earlier Victorian Hansard is PDF
  only and needs an OCR adapter.
- The Victorian member search does not publish membership dates, so Victorian periods are derived from
  speech dates.
- Senate and committee Hansard are ingested via OpenAustralia where available; committee transcripts are
  not yet ingested.
- **Licensing:** Commonwealth Hansard is published under CC BY-NC-ND; Victorian reuse terms need
  confirming. Get a licensing review before publishing generated summaries.

## Using a model

```bash
export MP_PROFILE_MODEL=global.anthropic.claude-sonnet-4-6   # any Bedrock model id; "none" = extractive
mp-profile profile "..."                                     # LLM period + career summaries, still verified
mp-profile chat                                              # agent over the Parliament MCP server
```

Model output goes through the same deterministic verifier: claims whose quote is not an exact substring of
the member's own words in the cited passage, or that cite another speaker, a date outside their
membership, or an off-topic passage, are removed and listed in `verification_issues`. Period summaries are
cached by passage set, summariser and prompt version, so re-running a profile only pays for new periods.

## MCP server

```bash
mp-profile mcp          # stdio
```

Tools: `search_members`, `get_member`, `search_speeches`, `get_passage`, `coverage`, `mp_profile`.
Example client config:

```json
{"mcpServers": {"parliament-au": {"command": "mp-profile", "args": ["mcp"],
  "env": {"MP_PROFILE_DB": "/path/to/speeches.sqlite"}}}}
```

## Observability

`setup_telemetry()` wires Strands' OpenTelemetry tracing and metrics. Each profile request is one trace:
`mp_profile.request` → Strands graph → node spans → agent/model/tool spans, with person, topic and prompt
version as attributes. Domain metrics (`mp_profile.*`) include request latency, passages considered,
claims emitted, verifier rejections by check, cache hits, tool calls/latency, tokens, ingestion counts,
unattributed passages and eval scores.

| Variable | Effect |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | export traces and metrics over OTLP/HTTP (Langfuse, Jaeger, CloudWatch via ADOT, …) |
| `MP_PROFILE_CONSOLE_TELEMETRY=1` | print spans/metrics to the console |
| `MP_PROFILE_RUN_LOG=runs.jsonl` | log every response with its trace id for online evals |

## Evals

```bash
mp-profile eval topics               # topic tagger precision/recall/F1 vs labelled passages
mp-profile eval offline              # golden set via strands-agents-evals (deterministic gates)
mp-profile eval offline --judge      # + LLM-as-judge neutrality and faithfulness (Bedrock)
mp-profile eval online --log runs.jsonl   # re-score production runs; prints failing trace ids
```

Deterministic checks (CI gates): quote fidelity, speaker attribution, topic relevance, membership-date
alignment, exactly four categories, gap honesty (no claims without evidence), neutrality lexicon,
person resolution, disambiguation, jurisdiction coverage for dual-career members, context fidelity
(quoted opponents, interjections and motions never quoted as the member), and graph trajectory.

CI (`.github/workflows/ci.yml`) runs ruff, mypy, pytest and the eval gates, and uploads the eval report.
