"""Adapters layer for aumos-testing-harness.

Contains external service integrations:
  - repositories.py   — SQLAlchemy BaseRepository implementations
  - kafka.py          — Kafka event publisher for test lifecycle events
  - llm_evaluator.py  — deepeval LLM metric evaluator
  - rag_evaluator.py  — RAGAS RAG pipeline evaluator
  - agent_evaluator.py — Agent capability evaluator
  - red_team_runner.py — Garak + Giskard OWASP red-team probe runner
"""
