"""Phase 1 data audit. Offline tooling: reads ground truth (labels, patterns) by design.

Nothing under scripts/audit may be imported by the agent runtime (src/agents, src/tools,
src/rag, src/app); tests/test_eval_isolation.py enforces this.
"""
