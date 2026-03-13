# Key Findings

- Dataset coverage: 1956 comments across 5 videos, with a spam rate of 51.4%.
- Best held-out F1: Random Forest at 0.947; deployed model: Logistic Regression for strong performance plus easier reviewer-facing explanations.
- Default queue policy (`review_threshold=0.55`, `auto_remove_threshold=0.85`) auto-removes 24.9% of comments and sends 21.7% to human review.
- Under that policy, the prototype catches 88.4% of spam while keeping wrongful auto-removals to 0 comments on the held-out set.
- Error analysis found 8 false positives and 23 false negatives at the default 0.50 threshold; borderline promotional language and short ambiguous comments remain the hardest cases.
