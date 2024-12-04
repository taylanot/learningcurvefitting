import numpy as np

data = np.genfromtxt("regression_train.csv",delimiter=",")


weights = (arange(2,101))[:, None]

# We have the following shape [new dataset, meta dataset, learner, (point on curve),... extra dims]
scalar = np.sum(weights * data[None] * data[:, None], axis=3) / np.sum(weights * (data ** 2), axis=2)[None]

adapted_curves = scalar[:, :, :, None] * data
adapted_target = scalar[..., None] * self.target
distance = np.sum((data - data[:, None]) ** 2, axis=3)
distance_adapted = np.sum((adapted_curves - data) ** 2, axis=3)

# extra dim for moving target
distance = np.repeat(distance[..., None], len(self.target_anchors), axis=-1)
distance_adapted = np.repeat(distance_adapted[..., None], len(self.target_anchors), axis=-1)

# Remove curves that can't predict at target
ind = np.isnan(self.label).nonzero()
distance[:, ind[0], ind[1], ..., ind[2]] = np.nan
distance_adapted[:, ind[0], ind[1], ..., ind[2]] = np.nan

# So that it doesn't pick itself
np.einsum('ii...->i...', distance)[...] = np.nan
np.einsum('ii...->i...', distance_adapted)[...] = np.nan

# Take k closest
part_scale_after = np.argpartition(distance, k, axis=1)[:, :k]
part_scale_before = np.argpartition(distance_adapted, k, axis=1)[:, :k]
k_closest_curves_scale_after = np.take_along_axis(adapted_target, part_scale_after, axis=1)
k_closest_curves_scale_before = np.take_along_axis(adapted_target, part_scale_before, axis=1)

# Predicted target is just the mean of the k closest curves at the target point
prediction_scale_after = np.mean(k_closest_curves_scale_after, axis=1)
prediction_scale_before = np.mean(k_closest_curves_scale_before, axis=1)


print(prediction_scale_after)
