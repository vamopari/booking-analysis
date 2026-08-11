
## Discussion: Precision/Recall Trade-off (Logistic Regression)

Default Logistic Regression (threshold = 0.5): Accuracy 0.809, Precision 0.839, Recall 0.598, F1 0.699, MCC 0.582.

Using `class_weight="balanced"`: Accuracy 0.787, Precision 0.709, Recall 0.721, F1 0.715, MCC 0.545.

AUC stayed identical (0.852) in both cases — `class_weight` shifts the decision
threshold's effective trade-off point but doesn't change the model's underlying
ranking ability. We kept the default (unweighted) model for consistency across
all 5 models, but in a real deployment, the choice would depend on which error
is costlier for the hotel: a missed cancellation (lost resale opportunity) vs.
a false alarm (unnecessary overbooking prep). If missed cancellations are
costlier, the balanced version's higher recall (0.721 vs 0.598) would likely
be the better production choice despite the drop in precision.
