# Task Forge Self-Consistency Dry-Run

Tester models read each `curated_task.json` plus the visible source and propose an execution plan.
Independent judge `gpt-5.4-mini` scores them `pass / partial / fail`.

- Tester models: `kimi_k2_5, glm_4_7_flash, deepseek_v4_flash, mimo_v2_5_pro`
- Drafter-family testers (reference-only column): `mimo_v2_5_pro`
- Cross-vendor roll-up labels: `kimi_k2_5, glm_4_7_flash, deepseek_v4_flash`

## Bucket Summary

### `anchor_candidate`

- task_count: `2`
- avg_score_spread (all testers): `6`
- avg_cross_vendor_score_spread: `6`
- avg_pass_count_per_task (all testers): `2.5`
- avg_cross_vendor_pass_count_per_task: `1.5`
- cross_vendor_pass_rate: `0.5`
- avg_total_score_by_model: `{'kimi_k2_5': 10, 'glm_4_7_flash': 4, 'deepseek_v4_flash': 6.5, 'mimo_v2_5_pro': 9}`
- pass_rate_by_model: `{'kimi_k2_5': 1.0, 'glm_4_7_flash': 0.5, 'deepseek_v4_flash': 0.0, 'mimo_v2_5_pro': 1.0}`
- outcome_votes: `{'pass': 5, 'partial': 2, 'fail': 1}`

## Task Summary

- `task_forge_20260519_path_small_v1_full_review_01` / `anchor_candidate` / `PP_move_finished.py`: scores={'glm_4_7_flash': 8, 'kimi_k2_5': 10, 'deepseek_v4_flash': 8, 'mimo_v2_5_pro': 8} cross_vendor_scores={'kimi_k2_5': 10, 'glm_4_7_flash': 8, 'deepseek_v4_flash': 8} outcomes={'pass': 3, 'partial': 1}
- `task_forge_20260519_path_small_v1_full_review_02` / `anchor_candidate` / `compress.py`: scores={'glm_4_7_flash': 0, 'kimi_k2_5': 10, 'deepseek_v4_flash': 5, 'mimo_v2_5_pro': 10} cross_vendor_scores={'kimi_k2_5': 10, 'glm_4_7_flash': 0, 'deepseek_v4_flash': 5} outcomes={'fail': 1, 'pass': 2, 'partial': 1}

## Notes

- MiMo overlaps with the drafting model family; its column is kept for reference and is excluded from the cross-vendor roll-up.
- This dry-run is a self-consistency check on task design, not a production-pipeline step.
