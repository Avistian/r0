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
        .replace("/* APPLICATION */", (assets / "app.js").read_text())
        .replace("/* CURVES */", (assets / "curves.js").read_text())
        .replace("/* GRAPHS */", (assets / "graphs.js").read_text())
        .replace("/* PAIRS */", (assets / "pairs.js").read_text())
        .replace("<!-- INTERVENTIONS -->", (assets / "interventions.html").read_text())
        .replace("/* INTERVENTIONS */", (assets / "interventions.js").read_text())
        .replace("/* WORKBENCH */", (assets / "workbench.js").read_text())
        .replace("/* NATIVE */", (assets / "native.js").read_text())
        .replace("/* TRANSPORT */", transport)
        .replace("<!-- RUNTIME -->", "Colab generation" if callback_name else "local generation")
    )
