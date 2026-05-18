# Benchmark Rules

Task Forge should author benchmark tasks, not product requirements or implementation plans.

Use these rules:

1. Keep one core boundary per task.
2. Keep one planning unit per task.
3. Separate recoverable workspace facts from user-only policy.
4. Do not leak the answer through the prompt, harness, or rubric.
5. Prefer grounded, inspectable workflow slices over broad system design asks.
6. If a detail is not visible in the source or workspace, mark it as unresolved instead of inventing it.
7. Harnesses should observe the target behavior, not solve the task.
