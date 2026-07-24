# Kimi Task D-R Parser Human Review

Open `review_app.html` in a browser. No server or package installation is
required. The page stores progress in browser local storage; export CSV or JSON
before changing browser, machine, or profile.

## Review Objective

Determine the verdict expressed by the model response. This is a parser audit,
not a task-accuracy audit. Do not decide whether the trajectory truly succeeds,
and do not use the gold label to infer what the model intended.

Gold is hidden by default for this reason.

## Required Review Population

1. Two reviewers independently review all 49 priority rows where the legacy and
   corrected parsers disagree.
2. Both reviewers also review the remaining 83 recoverable rows.
3. Both reviewers review the one truncated/unparsable row.
4. Strict one-token responses do not require manual parser review.

Reviewers must work independently and export separate files before discussing
disagreements.

The HTML includes gold only for post-review auditing. Keep gold hidden while
assigning the human verdict.

## Human Verdict Rules

Choose exactly one:

- `Success`: the response ultimately concludes that the query succeeds.
- `Fail`: the response ultimately concludes that the query fails.
- `Ambiguous`: the response contains conflicting conclusions without a clear
  final resolution.
- `Unparsable`: the response is truncated, empty, refused, or contains no
  identifiable conclusion.

Apply evidence in this order:

1. A standalone final `Success` or `Fail` is decisive.
2. Otherwise, use an unambiguous concluding sentence about the query.
3. Statements describing the successful reference trajectory are context, not
   the query verdict.
4. Intermediate reasoning is not the verdict when a later conclusion resolves
   it.
5. Do not infer a missing conclusion from the gold label or from your own
   trajectory interpretation.
6. If two incompatible conclusions remain and neither is clearly final, choose
   `Ambiguous`.
7. If generation stops before a conclusion, choose `Unparsable`.

## Evidence Category

Choose the strongest applicable category:

- `Standalone final verdict`
- `Unambiguous concluding sentence`
- `Conflicting conclusions`
- `Truncated / no conclusion`

Quote the decisive ending or describe the conflict in Notes whenever confidence
is not High.

## Parser Agreement

The corrected parser is considered correct only when:

- human `Success` equals parser `Success`;
- human `Fail` equals parser `Fail`; or
- human `Unparsable` equals parser `Unparsable`.

Human `Ambiguous` never counts as parser agreement with a binary prediction.

## Acceptance Rule

Before freezing corrected Kimi metrics:

1. Adjudicate every disagreement between the two reviewers.
2. Manually resolve every item where the corrected parser disagrees with the
   adjudicated verdict.
3. Require at least 95% corrected-parser agreement over all 133 reviewed rows.
4. Require at least 95% agreement over the 49 parser-changed priority rows.
5. Check agreement separately by environment and variant; investigate any
   stratum below 90%.
6. Re-run scoring if adjudication changes any parsed verdict.

Report counts for Strict, Recoverable, Ambiguous, and Unparsable outputs, along
with inter-reviewer agreement and adjudicated parser agreement.
