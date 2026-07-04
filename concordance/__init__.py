"""Concordance — extract interesting vocabulary from books.

See the requirements & architecture spec for the design. The pipeline is
frequency-floor -> validity-gate -> LLM interestingness judge -> review -> CSV,
and it is deliberately *keep-biased*: a genuine rarity should survive to review
even at the cost of a little noise, never the reverse.
"""

__version__ = "0.1.0"
