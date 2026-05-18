# Task Families

## Archive Cleanup

- Good boundary: whether cleanup of the original archive is safe after split output is verified.
- Recoverable facts: chunk names, emitted files, archive presence, visible output layout.
- User-only policy: whether ambiguous or incomplete split output still permits cleanup.
- Common failure mode: collapsing into static code review instead of an execution-grounded verification task.

## Mutation Move

- Good boundary: whether a source item or pair is eligible to be moved.
- Recoverable facts: current inventory, pairing evidence, destination conflicts.
- User-only policy: skip-vs-fail behavior for partial or ambiguous pairs.
- Common failure mode: broadening into a whole migration plan instead of one bounded mutation decision.

## Symlink Layout

- Good boundary: whether one split member should link to one source target under visible constraints.
- Recoverable facts: split membership files, source existence, target layout.
- User-only policy: how missing sources should be reported or blocked.

## Schema Generation

- Good boundary: whether one dataset entry can be emitted without inventing defaults.
- Recoverable facts: filenames, directory structure, code constants, visible conventions.
- User-only policy: whether unstated defaults like labels or modality may be filled in.
