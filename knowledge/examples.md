# Example Cohorts

## Strong Anchor Example

- Title: `Promote Complete Image-Mask Pairs Safely`
- Boundary: move only verified complete pairs and leave partial pairs untouched
- Why it works: execution-grounded, observable, one planning unit, one clear mutation decision

## Supporting Example

- Title: `Delete Original Archive Only After Verified Split Output`
- Boundary: decide whether cleanup is safe after verifying expected split parts
- Why it is supporting: still workflow-grounded, but cleanup eligibility may remain underdefined unless verification rules are made explicit

## Reject Example

- Pattern: linked paths exist, but executable behavior is not visible
- Why reject: the pipeline should not invent a benchmark boundary from filenames or path literals alone

## Static Review Warning

- A task that only asks for code-quality review or unsafe deletion analysis is usually a supporting or reject shape, not a strong execution benchmark anchor.
