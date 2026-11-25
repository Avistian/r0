"use strict";
let curve = null;
function clearCurve() {
  curve = null;
  $("response-plot").replaceChildren(); $("curve-values").textContent = "";
  $("curve-note").textContent = current?.view.recipe.recipe.version === "exact-gp-v1"
    ? "A direct GP draw needs explicitly identified conditional values. This analytic-curve view does not interpolate or invent an extension."
    : "The curve uses the full observed feature range. Select a row and evaluate to see the response.";
}
function drawCurve() {
  if (!curve) return;
  const channel = Math.min(number("channel"), curve.scores[0].length - 1);
  const values = curve.scores.map(row => row[channel]);
  const {svg, x, y} = plotFrame("response-plot", "Saved function response", `x${curve.feature}`, `Latent signal ${channel}`, ...curve.bounds, Math.min(...values), Math.max(...values));
  const points = curve.values.map((value, i) => `${x(value)},${y(values[i])}`).join(" ");
  svg.append(element("polyline", {points, fill: "none", stroke: "#17826b", "stroke-width": 2}));
  $("curve-note").textContent = `${curve.points} evaluated points across the observed range of x${curve.feature} (${curve.feature_role}). All other inputs are fixed to row ${curve.anchor_row}. Lines connect these points; behavior between them is not resolved.`;
  $("curve-values").textContent = pretty(curve);
}
$("curve-row").onchange = clearCurve;
$("curve-points").onchange = clearCurve;
$("curve").onclick = () => act("Evaluating saved function…", async () => {
  clearCurve();
  const result = await request("curve", {feature: number("feature"), anchor_row: number("curve-row"), points: number("curve-points")});
  if (result.curve.feature === number("feature") && result.curve.anchor_row === number("curve-row") && result.curve.points === number("curve-points")) {
    curve = result.curve; drawCurve();
  }
});
