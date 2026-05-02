"""
app.py — Predictive Maintenance Gradio Dashboard
================================================
Entry point for Hugging Face Spaces (and local dev).
Imports preprocessing from pipeline.py and prediction logic from router.py.
Contains zero ML logic — only UI wiring.

Compatible with Gradio 6.x
"""

import os
import tempfile
import json
import gradio as gr
import pandas as pd
from pipeline import preprocess, build_single_input
from router import predict

# ── Paths ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV_PATH = os.path.join(_BASE, "sample_batch.csv")
METRICS_PATH = os.path.join(_BASE, "model_metrics.json")

# ── Shared UI options ────────────────────────────────────────────────
PRIORITY_OPTIONS = [
    "Minimize missed failures",
    "Minimize unnecessary maintenance",
]
DETAIL_OPTIONS = [
    "Primary cause only",
    "All contributing causes",
]

# ── Custom CSS for a polished look ───────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Header ──────────────────────────────────────────────────────── */
.header-bar {
    background: linear-gradient(135deg, #2e0f12 0%, #5c1e24 50%, #3d1418 100%);
    padding: 36px 40px 32px;
    border-radius: 20px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(46, 15, 18, 0.45);
    position: relative;
    overflow: hidden;
}
.header-bar::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(217, 186, 140, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(163, 42, 57, 0.08) 0%, transparent 50%);
    pointer-events: none;
}
.header-bar h1 {
    color: #fff;
    font-size: 2.2em;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.8px;
    position: relative;
    z-index: 1;
}
.header-bar p {
    color: #d1bba4;
    margin: 0;
    font-size: 1em;
    font-weight: 400;
    letter-spacing: 0.2px;
    position: relative;
    z-index: 1;
}
.header-bar .badge {
    display: inline-block;
    background: rgba(217, 186, 140, 0.25);
    border: 1px solid rgba(217, 186, 140, 0.4);
    color: #f5e6d3;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    margin-top: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}

/* ── Section headers ─────────────────────────────────────────────── */
.section-header {
    background: linear-gradient(135deg, rgba(163, 42, 57, 0.08), rgba(217, 186, 140, 0.05));
    border: 1px solid rgba(163, 42, 57, 0.15);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.section-header h3 {
    margin: 0;
    font-size: 1em;
    font-weight: 700;
    color: #e8cca4;
    letter-spacing: 0.3px;
}
.section-header p {
    margin: 4px 0 0;
    font-size: 0.82em;
    color: #c4af9d;
}

/* ── Result cards ────────────────────────────────────────────────── */
.result-card {
    padding: 28px;
    border-radius: 16px;
    margin-top: 14px;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeSlideIn 0.4s ease-out;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-ok {
    background: linear-gradient(135deg, #0a2e1c, #145233);
    border-left: 5px solid #34d399;
}
.result-fail {
    background: linear-gradient(135deg, #2d0a0a, #541a1a);
    border-left: 5px solid #f87171;
}
.result-card .icon {
    font-size: 2em;
    margin-bottom: 4px;
}
.result-card .status {
    font-size: 1.4em;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.3px;
}
.result-ok  .status { color: #34d399; }
.result-fail .status { color: #f87171; }
.result-card .reason {
    font-size: 1.05em;
    color: #e2e8f0;
    margin-bottom: 8px;
    line-height: 1.5;
}
.result-card .reason strong {
    color: #fff;
}
.result-card .meta {
    font-size: 0.8em;
    color: #c4af9d;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 10px;
}

/* ── Summary stats row (batch tab) ───────────────────────────────── */
.stats-row {
    display: flex;
    gap: 14px;
    margin: 16px 0;
    animation: fadeSlideIn 0.4s ease-out;
}
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, rgba(46, 15, 18, 0.8), rgba(61, 20, 24, 0.6));
    border: 1px solid rgba(163, 42, 57, 0.15);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.stat-card .stat-value {
    font-size: 2em;
    font-weight: 800;
    margin-bottom: 2px;
}
.stat-card .stat-label {
    font-size: 0.78em;
    color: #c4af9d;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.stat-total .stat-value { color: #f5e6d3; }
.stat-ok    .stat-value { color: #34d399; }
.stat-fail  .stat-value { color: #f87171; }
.stat-rate  .stat-value { color: #fbbf24; }

/* ── Info box ────────────────────────────────────────────────────── */
.info-box {
    background: linear-gradient(135deg, rgba(163, 42, 57, 0.08), rgba(217, 186, 140, 0.05));
    border: 1px solid rgba(163, 42, 57, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.88em;
    color: #e8cca4;
    line-height: 1.6;
}
.info-box code {
    background: rgba(163, 42, 57, 0.15);
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.9em;
    color: #f5e6d3;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 20px 0 8px;
    margin-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #64748b;
    font-size: 0.8em;
    letter-spacing: 0.3px;
}
.footer a {
    color: #d6a87c;
    text-decoration: none;
}

/* ── Misc polish ─────────────────────────────────────────────────── */
.gradio-container { max-width: 1100px !important; }
button.primary {
    background: linear-gradient(135deg, #8a2533, #b03041) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(163, 42, 57, 0.35) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    box-shadow: 0 6px 25px rgba(163, 42, 57, 0.5) !important;
    transform: translateY(-1px) !important;
}
"""


# ── Load model metrics ───────────────────────────────────────────────
def load_model_metrics():
    """Load model metrics from JSON file."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def render_metrics_table():
    """Render metrics as HTML table."""
    metrics = load_model_metrics()
    
    if not metrics:
        return """
        <div class='info-box'>
            <strong>Note:</strong> Model metrics not yet generated. Run <code>python generate_metrics.py</code> to populate this tab.
        </div>
        """
    
    html = """
    <div style='overflow-x: auto; margin-top: 20px;'>
    """
    
    for model_name, model_data in metrics.items():
        test_set = model_data.get("test_set", {})
        cv_5fold = model_data.get("cv_5fold", {})
        train_size = model_data.get("train_size", 0)
        test_size = model_data.get("test_size", 0)
        
        html += f"""
        <div class='stat-card' style='margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, rgba(46, 15, 18, 0.6), rgba(61, 20, 24, 0.4)); border-radius: 12px; border-left: 4px solid #d6a87c;'>
            <h3 style='color: #e8cca4; margin-top: 0;'>{model_name}</h3>
            <p style='color: #c4af9d; font-size: 0.9em; margin: 5px 0;'>Train: {train_size} samples | Test: {test_size} samples</p>
            
            <table style='width: 100%; margin-top: 15px; border-collapse: collapse; color: #e2e8f0;'>
                <thead>
                    <tr style='background: rgba(163, 42, 57, 0.2); border-bottom: 2px solid rgba(163, 42, 57, 0.3);'>
                        <th style='padding: 10px; text-align: left;'>Threshold</th>
                        <th style='padding: 10px; text-align: center;'>Metric</th>
                        <th style='padding: 10px; text-align: center;'>Test Set</th>
                        <th style='padding: 10px; text-align: center;'>CV Mean ± Std</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for threshold in sorted(test_set.keys()):
            test_metrics = test_set.get(threshold, {})
            cv_metrics = cv_5fold.get(threshold, {})
            
            # Precision row
            html += f"""
                <tr style='border-bottom: 1px solid rgba(163, 42, 57, 0.1);'>
                    <td style='padding: 10px; color: #f5e6d3;'><strong>{threshold}</strong></td>
                    <td style='padding: 10px; text-align: center; color: #fbbf24;'>Precision</td>
                    <td style='padding: 10px; text-align: center;'><strong>{test_metrics.get("precision", "N/A")}</strong></td>
                    <td style='padding: 10px; text-align: center;'>{cv_metrics.get("mean", {}).get("precision", "N/A")} ± {cv_metrics.get("std", {}).get("precision", "N/A")}</td>
                </tr>
            """
            
            # Recall row
            html += f"""
                <tr style='border-bottom: 1px solid rgba(163, 42, 57, 0.1);'>
                    <td style='padding: 10px; color: #f5e6d3;'></td>
                    <td style='padding: 10px; text-align: center; color: #fbbf24;'>Recall</td>
                    <td style='padding: 10px; text-align: center;'><strong>{test_metrics.get("recall", "N/A")}</strong></td>
                    <td style='padding: 10px; text-align: center;'>{cv_metrics.get("mean", {}).get("recall", "N/A")} ± {cv_metrics.get("std", {}).get("recall", "N/A")}</td>
                </tr>
            """
            
            # F1 row
            html += f"""
                <tr style='border-bottom: 1px solid rgba(163, 42, 57, 0.1);'>
                    <td style='padding: 10px; color: #f5e6d3;'></td>
                    <td style='padding: 10px; text-align: center; color: #fbbf24;'>F1 Score</td>
                    <td style='padding: 10px; text-align: center;'><strong>{test_metrics.get("f1", "N/A")}</strong></td>
                    <td style='padding: 10px; text-align: center;'>{cv_metrics.get("mean", {}).get("f1", "N/A")} ± {cv_metrics.get("std", {}).get("f1", "N/A")}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
    
    html += "</div>"
    return html


# ── Tab 1: Manual single prediction ─────────────────────────────────
def predict_single(
    air_temp, process_temp, rpm, torque, tool_wear,
    machine_type, business_priority,
):
    try:
        df = build_single_input(
            air_temp, process_temp, rpm, torque, tool_wear, machine_type
        )
        X = preprocess(df)
        # diagnostic_detail defaults to 'All contributing causes' in router.predict
        res = predict(X, business_priority)[0]
    except Exception as exc:
        return f"""
        <div class='result-card result-fail'>
            <div class='icon'>⚠️</div>
            <div class='status'>ERROR</div>
            <div class='reason'>{exc}</div>
        </div>"""

    confidence_pct = res.get("confidence", 0.0) * 100
    failure_model = res.get("failure_model_name", "Unknown")
    failure_threshold = res.get("failure_threshold", 0.5)
    cause_model = res.get("cause_model_name", "Unknown")
    cause_threshold = res.get("threshold", 0.5)

    if res["failure_predicted"]:
        return f"""
        <div class='result-card result-fail'>
            <div class='icon'>🚨</div>
            <div class='status'>FAILURE DETECTED</div>
            <div class='reason'>Predicted cause: <strong>{res['failure_reason']}</strong></div>
            <div class='meta'>
                📊 Confidence: <strong>{confidence_pct:.1f}%</strong> &nbsp;·&nbsp;
                🎯 Priority: {business_priority}<br/>
                🔬 Detail: All contributing causes<br/>
                <strong>Model Pipeline:</strong><br/>
                1️⃣ Failure Detection: {failure_model} (threshold: {failure_threshold:.2f})<br/>
                2️⃣ Cause Detection: {cause_model} (threshold: {cause_threshold:.2f})
            </div>
        </div>"""
    else:
        return f"""
        <div class='result-card result-ok'>
            <div class='icon'>✅</div>
            <div class='status'>SYSTEM NORMAL</div>
            <div class='reason'>No failure predicted — machine operating within safe parameters.</div>
            <div class='meta'>
                📊 Confidence: <strong>{confidence_pct:.1f}%</strong> &nbsp;·&nbsp;
                🎯 Priority: {business_priority}<br/>
                🔬 Detail: All contributing causes<br/>
                <strong>Model Pipeline:</strong><br/>
                1️⃣ Failure Detection: {failure_model} (threshold: {failure_threshold:.2f})<br/>
                2️⃣ Cause Detection: {cause_model} (threshold: {cause_threshold:.2f})
            </div>
        </div>"""


# ── Tab 2: Batch CSV prediction ──────────────────────────────────────
def predict_batch(csv_file, business_priority):
    if csv_file is None:
        gr.Warning("Please upload a CSV file first.")
        return None, None, ""

    try:
        filepath = csv_file.name if hasattr(csv_file, "name") else csv_file
        df_raw = pd.read_csv(filepath)
    except Exception as exc:
        gr.Warning(f"Failed to read CSV: {exc}")
        return None, None, ""

    if "Type" not in df_raw.columns:
        gr.Warning(
            "CSV is missing the 'Type' column (expected values: L, M, H). "
            "Cannot proceed."
        )
        return None, None, ""

    try:
        X = preprocess(df_raw.copy())
        results = predict(X, business_priority)
    except Exception as exc:
        gr.Warning(f"Prediction failed: {exc}")
        return None, None, ""

    df_out = df_raw.copy()
    df_out["Predicted_Failure"] = [int(r["failure_predicted"]) for r in results]
    df_out["Failure_Reason"]    = [r["failure_reason"] for r in results]
    df_out["Confidence_%"]      = [f"{r.get('confidence', 0.0) * 100:.1f}" for r in results]
    df_out["Failure_Detection_Model"]      = [r.get("failure_model_name", "Unknown") for r in results]
    df_out["Failure_Detection_Threshold"]  = [f"{r.get('failure_threshold', 0.5):.2f}" for r in results]
    df_out["Cause_Detection_Model"]        = [r.get("cause_model_name", "Unknown") for r in results]
    df_out["Cause_Detection_Threshold"]    = [f"{r.get('threshold', 0.5):.2f}" for r in results]

    output_path = os.path.join(tempfile.gettempdir(), "predictions_output.csv")
    df_out.to_csv(output_path, index=False)

    # Build summary stats
    total = len(results)
    n_fail = sum(1 for r in results if r["failure_predicted"])
    n_ok = total - n_fail
    rate = (n_fail / total * 100) if total > 0 else 0
    avg_confidence = (sum(r.get("confidence", 0.0) for r in results) / total * 100) if total > 0 else 0

    summary_html = f"""
    <div class='stats-row'>
        <div class='stat-card stat-total'>
            <div class='stat-value'>{total}</div>
            <div class='stat-label'>Total Rows</div>
        </div>
        <div class='stat-card stat-ok'>
            <div class='stat-value'>{n_ok}</div>
            <div class='stat-label'>Normal</div>
        </div>
        <div class='stat-card stat-fail'>
            <div class='stat-value'>{n_fail}</div>
            <div class='stat-label'>Failures</div>
        </div>
        <div class='stat-card stat-rate'>
            <div class='stat-value'>{rate:.1f}%</div>
            <div class='stat-label'>Failure Rate</div>
        </div>
    </div>
    <div class='stats-row'>
        <div class='stat-card stat-ok'>
            <div class='stat-value'>{avg_confidence:.1f}%</div>
            <div class='stat-label'>Avg Confidence</div>
        </div>
        <div class='stat-card stat-total'>
            <div class='stat-value'>{business_priority[:15]}...</div>
            <div class='stat-label'>Priority Mode</div>
        </div>
    </div>"""

    preview = df_out.head(15)
    return preview, output_path, summary_html


# ── Build Gradio interface (Gradio 6.x compatible) ───────────────────
with gr.Blocks(
    title="Predictive Maintenance Dashboard",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    # ── Header ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class='header-bar'>
        <h1>⚙️ Predictive Maintenance Dashboard</h1>
        <p>Real-time failure prediction from industrial sensor data &mdash;
        powered by Decision Tree ensemble models</p>
        <span class='badge'>ML Lab &middot; Machine Learning Project</span>
    </div>
    """)

    # ── Tab 1 - Manual Input ─────────────────────────────────────────
    with gr.Tab("🔧 Manual Input"):
        gr.HTML("""
        <div class='info-box'>
            💡 Enter sensor readings manually to get an instant prediction.
            Adjust <strong>Business Priority</strong> to switch between cost-sensitive
            and baseline models, and <strong>Diagnostic Detail</strong> to choose
            single vs. multi-cause diagnosis.
        </div>
        """)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML("<div class='section-header'><h3>📊 Sensor Readings</h3><p>Input real-time machine telemetry</p></div>")
                air_temp     = gr.Number(label="Air Temperature (K)",     value=298.1, minimum=290, maximum=310)
                process_temp = gr.Number(label="Process Temperature (K)", value=308.6, minimum=300, maximum=320)
                rpm          = gr.Number(label="Rotational Speed (RPM)",  value=1551,  minimum=0,   maximum=3000)
                torque       = gr.Number(label="Torque (Nm)",             value=42.8,  minimum=0,   maximum=100)
                tool_wear    = gr.Number(label="Tool Wear (min)",         value=0,     minimum=0,   maximum=300)
                mtype        = gr.Dropdown(
                    label="Machine Type",
                    choices=["L", "M", "H"],
                    value="M",
                    info="L = Low quality, M = Medium, H = High",
                )

            with gr.Column(scale=1):
                gr.HTML("<div class='section-header'><h3>🎛️ Model Configuration</h3><p>Select decision strategy and diagnostic depth</p></div>")
                priority = gr.Radio(
                    label="Business Priority",
                    choices=PRIORITY_OPTIONS,
                    value=PRIORITY_OPTIONS[1],
                    info="Cost-sensitive model penalises missed failures more heavily",
                )
                # Diagnostic detail is fixed to 'All contributing causes' by default
                gr.HTML("<div class='info-box'>Diagnostic detail is set to <strong>All contributing causes</strong>.</div>")
                run_btn = gr.Button("⚡ Run Prediction", variant="primary", size="lg")

                gr.HTML("<div class='section-header'><h3>📋 Prediction Result</h3><p>Model output will appear below</p></div>")
                output = gr.HTML()

        run_btn.click(
            fn=predict_single,
            inputs=[air_temp, process_temp, rpm, torque, tool_wear, mtype, priority],
            outputs=output,
        )

    # ── Tab 2 - Batch CSV Upload ─────────────────────────────────────
    with gr.Tab("📁 Batch CSV Upload"):
        gr.HTML("""
        <div class='info-box'>
            📤 Upload a CSV file with sensor columns
            (<code>Type</code>, <code>Air temperature [K]</code>,
            <code>Process temperature [K]</code>, <code>Rotational speed [rpm]</code>,
            <code>Torque [Nm]</code>, <code>Tool wear [min]</code>).
            The model will append <code>Predicted_Failure</code> and
            <code>Failure_Reason</code> columns to the results.
        </div>
        """)

        # Sample CSV download
        if os.path.exists(SAMPLE_CSV_PATH):
            gr.HTML("<div class='section-header'><h3>📥 Sample File</h3><p>Download this 20-row demo CSV to test the batch pipeline</p></div>")
            sample_download = gr.File(
                label="sample_batch.csv — 20 rows with mixed failure scenarios",
                value=SAMPLE_CSV_PATH,
                interactive=False,
            )

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<div class='section-header'><h3>📂 Upload Your CSV</h3><p>Drag & drop or click to browse</p></div>")
                csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            with gr.Column(scale=1):
                gr.HTML("<div class='section-header'><h3>🎛️ Configuration</h3></div>")
                priority_b = gr.Radio(
                    label="Business Priority",
                    choices=PRIORITY_OPTIONS,
                    value=PRIORITY_OPTIONS[0],
                )
                gr.HTML("<div class='info-box'>Diagnostic detail for batch runs defaults to <strong>All contributing causes</strong>.</div>")

        batch_btn = gr.Button("⚡ Run Batch Prediction", variant="primary", size="lg")

        # Summary stats (populated after prediction)
        batch_summary = gr.HTML()

        preview_tbl  = gr.Dataframe(label="Preview (first 15 rows)", wrap=True)
        download_btn = gr.File(label="📥 Download Full Results CSV")

        batch_btn.click(
            fn=predict_batch,
            inputs=[csv_input, priority_b],
            outputs=[preview_tbl, download_btn, batch_summary],
        )

    # ── Tab 3 - Model Statistics ────────────────────────────────────
    with gr.Tab("📊 Model Statistics"):
        gr.HTML("""
        <div class='info-box'>
            <strong>Model Performance Metrics</strong><br/>
            Test Set Performance vs 5-Fold Cross-Validation results
        </div>
        """)
        
        gr.HTML(render_metrics_table())

    # ── Footer ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class='footer'>
        Predictive Maintenance Project &middot; Decision Tree Models &middot;
        Built with <a href='https://gradio.app' target='_blank'>Gradio</a>
    </div>
    """)

# ── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        ssr_mode=False,
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
