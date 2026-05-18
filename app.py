import gradio as gr
from engine.input_layer import InputProcessor
from engine.retrieval_layer import get_retriever
from engine.reasoning_layer import BiasReasoningEngine
from engine.utils import format_bias_report_html, format_warnings_html, pretty_print_json

print("Initializing Evidence Retriever...")
retriever = get_retriever()

print("Initializing Reasoning Engine...")
reasoning_engine = BiasReasoningEngine()


def process_analysis(text: str, url: str, image) -> tuple[str, str]:
    try:
        context = InputProcessor.build_unified_context(text, url, image)
        warnings_html = format_warnings_html(context.get("warnings", []))

        if not context["has_text"] and not context["has_image"]:
            if context.get("warnings"):
                error_html = warnings_html + format_bias_report_html(
                    {"error": "Could not extract usable content from the provided inputs. See warnings above."}
                )
            else:
                error_html = format_bias_report_html(
                    {"error": "Please provide at least some text, a URL, or an image."}
                )
            return error_html, "{}"

        query_text = context["text"][:500]
        evidence = retriever.retrieve(query_text, top_k=2)
        report = reasoning_engine.analyze(context, evidence)

        html_output = warnings_html + format_bias_report_html(report)
        json_output = pretty_print_json(report)

        return html_output, json_output

    except Exception as e:
        return format_bias_report_html({"error": f"Pipeline failure: {str(e)}"}), "{}"


css = """
/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Page background ────────────────────────────────────── */
body, html {
    background: #eef2ff !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    width: 100% !important;
    overflow-x: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    background: #eef2ff !important;
    padding: 0 clamp(12px, 3vw, 40px) clamp(40px, 6vh, 80px) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif !important;
    min-height: 100vh !important;
}

/* Force Gradio's inner wrapper to also stretch full-width */
.gradio-container > .main,
.gradio-container > .main > .wrap {
    max-width: 100% !important;
    width: 100% !important;
}

footer { display: none !important; }

/* ── Two-column row fills full width ────────────────────── */
.gradio-container [class*="row"],
.gradio-container [class*="Row"] {
    width: 100% !important;
    display: flex !important;
    gap: clamp(12px, 2vw, 28px) !important;
}

/* ── Input card ─────────────────────────────────────────── */
#input-col {
    background: #ffffff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 18px !important;
    padding: clamp(16px, 2.5vw, 32px) !important;
    box-shadow: 0 4px 24px rgba(79, 70, 229, 0.08) !important;
    transition: box-shadow 0.2s !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
#input-col:focus-within {
    box-shadow: 0 4px 32px rgba(79, 70, 229, 0.14) !important;
}

/* ── Output card ────────────────────────────────────────── */
#output-col {
    background: #ffffff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 18px !important;
    padding: clamp(16px, 2.5vw, 32px) !important;
    box-shadow: 0 4px 24px rgba(79, 70, 229, 0.08) !important;
    min-height: 320px !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid #e0e7ff !important;
    margin-bottom: 16px !important;
    gap: 0 !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #a5b4fc !important;
    padding: 9px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: color 0.15s, border-color 0.15s !important;
    cursor: pointer !important;
}
.tab-nav button.selected {
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
    font-weight: 600 !important;
}
.tab-nav button:hover:not(.selected) {
    color: #6366f1 !important;
}

/* ── Textarea ───────────────────────────────────────────── */
textarea {
    background: #f5f3ff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 13px 15px !important;
    resize: vertical !important;
    transition: border-color 0.15s, box-shadow 0.15s, background 0.15s !important;
    width: 100% !important;
}
textarea:focus {
    background: #ffffff !important;
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14) !important;
    outline: none !important;
}
textarea::placeholder { color: #a5b4fc !important; font-style: italic !important; }

/* ── URL input ──────────────────────────────────────────── */
input[type="text"] {
    background: #f5f3ff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-size: 14px !important;
    padding: 11px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s, background 0.15s !important;
    width: 100% !important;
}
input[type="text"]:focus {
    background: #ffffff !important;
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14) !important;
    outline: none !important;
}
input[type="text"]::placeholder { color: #a5b4fc !important; font-style: italic !important; }

/* ── Image upload area ──────────────────────────────────── */
.gr-image-upload {
    border: 2px dashed #c7d2fe !important;
    border-radius: 12px !important;
    background: #f5f3ff !important;
}

/* ── Analyze button ─────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border-radius: 10px !important;
    height: 48px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.38) !important;
    transition: opacity 0.15s, transform 0.12s, box-shadow 0.15s !important;
    cursor: pointer !important;
    margin-top: 8px !important;
}
#analyze-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.46) !important;
}
#analyze-btn:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.24) !important;
}

/* ── Examples section ───────────────────────────────────── */
.examples-holder {
    margin-top: 20px !important;
}
.examples-holder > .label,
.examples-holder label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #6366f1 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 10px !important;
    display: block !important;
}
.examples-holder table {
    border-collapse: collapse !important;
    width: 100% !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    font-size: 13px !important;
}
.examples-holder table th {
    background: #f5f3ff !important;
    color: #6366f1 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid #e0e7ff !important;
    text-align: left !important;
}
.examples-holder table td {
    padding: 10px 14px !important;
    color: #3730a3 !important;
    border-bottom: 1px solid #f0f4ff !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
    line-height: 1.45 !important;
}
.examples-holder table tr:last-child td { border-bottom: none !important; }
.examples-holder table tr:hover td { background: #eef2ff !important; }

/* ── Accordion (JSON output) ────────────────────────────── */
.gr-accordion,
details {
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    background: #f5f3ff !important;
    margin-top: 16px !important;
    overflow: hidden !important;
}
.gr-accordion > div:first-child,
details > summary {
    padding: 11px 16px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #6366f1 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    cursor: pointer !important;
    user-select: none !important;
    list-style: none !important;
}

/* ── Mobile: stack columns ──────────────────────────────── */
@media (max-width: 860px) {
    /* Make the flex row wrap so columns stack */
    .gradio-container [class*="row"],
    .gradio-container [class*="Row"] {
        flex-wrap: wrap !important;
    }

    #input-col, #output-col {
        flex: 0 0 100% !important;
        min-width: 100% !important;
        width: 100% !important;
        border-radius: 14px !important;
    }

    #output-col { margin-top: 4px !important; }

    #analyze-btn {
        height: 52px !important;
        font-size: 15px !important;
    }
}

@media (max-width: 540px) {
    #input-col, #output-col {
        border-radius: 12px !important;
    }

    /* Prevent iOS font-size zoom on focus */
    textarea, input[type="text"] {
        font-size: 16px !important;
    }

    .tab-nav button {
        padding: 9px 14px !important;
        font-size: 12px !important;
    }
}
"""

PLACEHOLDER_HTML = """
<div style="
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:260px; gap:14px;
    font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;">
    <div style="
        width:56px; height:56px; border-radius:16px;
        background:linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        display:flex; align-items:center; justify-content:center;
        font-size:26px; border:1px solid #c7d2fe;">
        🔍
    </div>
    <div style="text-align:center;">
        <p style="margin:0 0 6px; font-size:15px; font-weight:600; color:#6366f1;">
            Ready to analyze
        </p>
        <p style="margin:0; font-size:13px; color:#a5b4fc; line-height:1.6; max-width:240px;">
            Paste text, enter a URL, or upload an image and press <strong style="color:#6366f1;">Analyze</strong>
        </p>
    </div>
</div>
"""

with gr.Blocks(theme=gr.themes.Base(), css=css, title="BiasLens") as demo:

    gr.HTML("""
    <div style="
        padding: 44px 0 32px;
        border-bottom: 1px solid #c7d2fe;
        margin-bottom: 28px;
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;">

        <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:12px;">
            <div style="
                width:40px; height:40px; border-radius:12px; flex-shrink:0;
                background:linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                display:flex; align-items:center; justify-content:center;
                font-size:20px; box-shadow:0 4px 12px rgba(79,70,229,0.35);">
                🔍
            </div>
            <span style="font-size:24px; font-weight:700; color:#1e1b4b; letter-spacing:-0.03em;">
                BiasLens
            </span>
            <div style="display:flex; gap:7px; flex-wrap:wrap; margin-left:2px;">
                <span style="
                    font-size:11px; font-weight:600; color:#4f46e5;
                    background:#eef2ff; padding:3px 11px; border-radius:20px;
                    border:1px solid #c7d2fe; letter-spacing:0.02em;">
                    Gemma 4 E4B
                </span>
                <span style="
                    font-size:11px; font-weight:600; color:#4f46e5;
                    background:#eef2ff; padding:3px 11px; border-radius:20px;
                    border:1px solid #c7d2fe; letter-spacing:0.02em;">
                    FAISS RAG
                </span>
                <span style="
                    font-size:11px; font-weight:600; color:#4f46e5;
                    background:#eef2ff; padding:3px 11px; border-radius:20px;
                    border:1px solid #c7d2fe; letter-spacing:0.02em;">
                    Offline-first
                </span>
            </div>
        </div>

        <p style="margin:0; font-size:14px; color:#6366f1; line-height:1.6; max-width:560px;">
            Detect cognitive, linguistic, and structural bias in any text, URL, or image.
            Powered by Gemma 4 &middot; FAISS retrieval &middot; No API key needed.
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="input-col"):
            with gr.Tabs():
                with gr.Tab("Text"):
                    input_text = gr.Textbox(
                        label="",
                        lines=6,
                        placeholder="Paste an article excerpt, headline, social media post, or any text…",
                        show_label=False,
                    )
                    input_url = gr.Textbox(
                        label="",
                        placeholder="Or fetch from a URL — https://example.com/article",
                        show_label=False,
                    )

                with gr.Tab("Image"):
                    input_image = gr.Image(
                        type="pil",
                        label="",
                        sources=["upload"],
                        show_label=False,
                    )

            analyze_btn = gr.Button("Analyze", elem_id="analyze-btn", size="lg")

            gr.Examples(
                examples=[
                    ["Women are naturally better at taking care of children than men.", "", None],
                    ["The invading forces from that country are always barbaric and uncivilized.", "", None],
                ],
                inputs=[input_text, input_url, input_image],
                label="Examples",
            )

        with gr.Column(scale=1, elem_id="output-col"):
            output_html = gr.HTML(value=PLACEHOLDER_HTML)

            with gr.Accordion("JSON Output", open=False):
                output_json = gr.Code(language="json", show_label=False)

    analyze_btn.click(
        fn=process_analysis,
        inputs=[input_text, input_url, input_image],
        outputs=[output_html, output_json],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
