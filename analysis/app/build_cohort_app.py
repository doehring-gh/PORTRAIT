"""Build a self-contained offline cohort Passport app by injecting a precomputed
cohort bundle into an HTML template. No statistics are computed in the browser."""
import json, os, sys

HERE = os.path.dirname(__file__)

def build(bundle_path, template_path=None, out_path=None):
    """
    Embed a cohort bundle into an HTML template to create a standalone app.
    
    Args:
        bundle_path: Path to precomputed cohort bundle (JSON).
        template_path: Path to HTML template with __BUNDLE__ placeholder.
                      Defaults to cohort_app_template.html in this directory.
        out_path: Output path for the generated HTML file.
                 Defaults to results/app/portrait_passport_app.html.
    
    Returns:
        Tuple of (output_path, html_size_bytes, patient_count).
    """
    template_path = template_path or os.path.join(HERE, "cohort_app_template.html")
    out_path = out_path or os.path.join(HERE, "..", "..", "results", "app", "portrait_passport_app.html")
    with open(bundle_path) as fh:
        bundle = json.load(fh)
    with open(template_path) as fh:
        tpl = fh.read()
    # Embed as compact JSON; escape </ to avoid breaking the <script> block
    payload = json.dumps(bundle, separators=(",", ":"), default=float).replace("</", "<\\/")
    html = tpl.replace("__BUNDLE__", payload)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path, len(html), bundle["n_patients"]

if __name__ == "__main__":
    bp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cohort_bundle_full.json")
    out, n, npat = build(bp)
    print("wrote", out, "bytes:", n, "patients:", npat)
