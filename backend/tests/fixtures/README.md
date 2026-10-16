# Synthetic test fixtures

Everything under `tests/fixtures/markdown/` is **invented** for automated tests.
None of it is a real ePolicies document, and it is never ingested into the real
knowledge base unless you explicitly point `MARKDOWN_DIR` here.

The fixtures cover nested directories, same-named files in different folders
(`hr/remote-work-policy.md` vs `it/remote-work-policy.md`), tables (including a
60-row table that must be split with repeated headers), lists, URLs, policy
identifiers, a deliberate policy conflict (3 vs 2 remote days), a duplicate
file, an empty file, a cp1252-encoded file and a prompt-injection attempt.
