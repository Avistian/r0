"""One set of controls and charts, with local HTTP or Colab callback transport."""

import json
from pathlib import Path


def explorer_page(*, token: str = "", callback_name: str | None = None) -> str:
    assets = Path(__file__).parent / "web"
    if callback_name is None:
        transport = "window.r0token = " + json.dumps(token) + ";"
    else:
        transport = """
window.r0request = async (action, data) => {
  const result = await google.colab.kernel.invokeFunction(CALLBACK_NAME, [action, data], {});
  const payload = result.data["application/json"];
  if (payload.error) throw new Error(payload.error);
  return payload;
};
""".replace("CALLBACK_NAME", json.dumps(callback_name))
    return (
        (assets / "index.html")
        .read_text()
        .replace("/* STYLES */", (assets / "style.css").read_text())
        .replace(
            "/* APPLICATION */",
            (assets / "task_controls.js").read_text() + "\n" + (assets / "app.js").read_text(),
        )
        .replace(
            "/* CURVES */",
            (assets / "curves.js").read_text()
            + "\n"
            + (assets / "gp_column_curves.js").read_text()
            + "\n"
            + (assets / "gp_target_edits.js").read_text(),
        )
        .replace("<!-- GP TARGET CONTROLS -->", (assets / "gp_column_curves.html").read_text())
        .replace(
            "/* GRAPHS */",
            (assets / "graphs.js").read_text() + (assets / "gp_covariance.js").read_text(),
        )
        .replace("<!-- GP POPULATION MATRICES -->", (assets / "gp_covariance.html").read_text())
        .replace("/* PAIRS */", (assets / "pairs.js").read_text())
        .replace(
            "<!-- INTERVENTIONS -->",
            (assets / "gp_target_edits.html").read_text()
            + (assets / "interventions.html")
            .read_text()
            .replace("<!-- TYPED_POLICY_FIELDS -->", (assets / "typed-policies.html").read_text()),
        )
        .replace(
            "/* INTERVENTIONS */",
            (assets / "interventions.js").read_text() + (assets / "typed-policies.js").read_text(),
        )
        .replace(
            "<!-- NETWORK CONTROLS -->",
            (assets / "network_controls.html").read_text()
            + (assets / "observations.html").read_text()
            + (assets / "convolutions.html").read_text()
            + (assets / "population_dags.html").read_text()
            + (assets / "gp_roots.html").read_text()
            + (assets / "gp_calibration.html").read_text(),
        )
        .replace("<!-- NETWORK WORKBENCH -->", (assets / "network_workbench.html").read_text())
        .replace(
            "/* NETWORKS */",
            (assets / "networks.js").read_text()
            + "\n"
            + (assets / "source_roots.js").read_text()
            + "\n"
            + (assets / "prototype_roots.js").read_text()
            + "\n"
            + (assets / "observations.js").read_text()
            + "\n"
            + (assets / "convolutions.js").read_text(),
        )
        .replace("<!-- FITTED CONTROLS -->", (assets / "fitted_controls.html").read_text())
        .replace("/* WORKBENCH */", (assets / "workbench.js").read_text())
        .replace("/* NATIVE */", (assets / "native.js").read_text())
        .replace("/* INNOVATIONS */", (assets / "innovations.js").read_text())
        .replace("<!-- INNOVATIONS -->", (assets / "innovations.html").read_text())
        .replace(
            "/* CONVERTERS */",
            (assets / "converters.js").read_text()
            + "\n"
            + (assets / "population_dags.js").read_text()
            + (assets / "gp_roots.js").read_text()
            + (assets / "gp_calibration.js").read_text(),
        )
        .replace("<!-- CONVERTERS -->", (assets / "converters.html").read_text())
        .replace("/* TRANSPORT */", transport)
        .replace("<!-- RUNTIME -->", "Colab generation" if callback_name else "local generation")
    )
