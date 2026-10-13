export function predict(model, features, scaler) {
  if (model.constant !== undefined) return model.constant;
  const x = features.map((value, i) => (value - scaler.mean[i]) / scaler.scale[i]);
  let decision = model.intercept;
  for (let i = 0; i < model.vectors.length; i++) {
    let squaredDistance = 0;
    for (let j = 0; j < x.length; j++) squaredDistance += (x[j] - model.vectors[i][j]) ** 2;
    decision += model.dual[i] * Math.exp(-model.gamma * squaredDistance);
  }
  return decision > 0 ? 1 : 0;
}
