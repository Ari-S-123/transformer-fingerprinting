# Presentation Summary

## Recommended storyline

1. Scale-up from `fma_small` to `fma_medium` changed the retrieval floor more than the average case.
2. Multi-segment aggregation is the cheapest inference-time intervention because it reuses historical checkpoints.
3. Hard-negative retraining is the training-time intervention that matters when acoustically similar false matches dominate.
4. The final recommendation should prioritize the best historical checkpoint, then add aggregation by default, and reserve retraining for the most failure-sensitive deployment path.
