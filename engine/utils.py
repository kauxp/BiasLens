import json

_FONT = "-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif"


def format_warnings_html(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li style='margin-bottom:4px;'>{w}</li>" for w in warnings)
    return f"""
<div style="
    padding:12px 16px; margin-bottom:14px;
    border:1px solid #fde68a; border-radius:10px;
    background:#fffbeb; font-family:{_FONT};">
    <p style="margin:0 0 6px; font-size:10px; font-weight:700; color:#92400e;
              text-transform:uppercase; letter-spacing:.08em;">Input Warning</p>
    <ul style="margin:0; padding-left:16px; font-size:13px; color:#78350f; line-height:1.6;">{items}</ul>
</div>"""


def format_bias_report_html(report: dict) -> str:
    if "error" in report:
        return f"""
<div style="
    padding:16px 18px; border:1px solid #fecaca; border-radius:12px;
    background:#fef2f2; font-family:{_FONT};">
    <p style="margin:0 0 5px; font-size:10px; font-weight:700; color:#991b1b;
              text-transform:uppercase; letter-spacing:.08em;">Error</p>
    <p style="margin:0; font-size:13px; color:#7f1d1d; line-height:1.6;">{report.get('error')}</p>
    {f'<pre style="margin:10px 0 0; font-size:11px; color:#991b1b; overflow-x:auto; white-space:pre-wrap; background:#fff5f5; padding:10px; border-radius:6px;">{report.get("raw","")}</pre>' if report.get('raw') else ''}
</div>"""

    score = report.get("bias_score", 0.0)
    confidence = report.get("confidence", 0.0)

    if score >= 0.6:
        status_label, status_color, status_bg = "Strong Bias",   "#dc2626", "#fef2f2"
    elif score >= 0.4:
        status_label, status_color, status_bg = "Moderate Bias", "#d97706", "#fffbeb"
    elif score >= 0.2:
        status_label, status_color, status_bg = "Mild Framing",  "#ca8a04", "#fefce8"
    else:
        status_label, status_color, status_bg = "No Significant Bias", "#059669", "#f0fdf4"

    bar_pct = int(score * 100)

    # ── Bias dimensions ────────────────────────────────────────
    bias_dims  = report.get("bias_dimensions", {})
    bias_types = report.get("bias_types", [])
    dims_html  = ""

    if bias_dims and isinstance(bias_dims, dict):
        active = sorted(
            ((k, v) for k, v in bias_dims.items() if v > 0),
            key=lambda x: x[1], reverse=True
        )
        if active:
            rows = ""
            for k, v in active:
                label = k.replace("_bias", "").replace("_", " ").title()
                rows += f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
    <span style="width:96px; font-size:12px; color:#6366f1; flex-shrink:0; font-weight:500;">{label}</span>
    <div style="flex:1; height:4px; background:#e0e7ff; border-radius:3px;">
        <div style="width:{int(v*100)}%; height:100%; background:{status_color};
                    border-radius:3px; transition:width 0.4s ease;"></div>
    </div>
    <span style="font-size:12px; color:#3730a3; width:28px; text-align:right; font-weight:600;">{v:.1f}</span>
</div>"""
            dims_html = f"""
<div style="margin-bottom:20px;">
    <p style="margin:0 0 10px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#a5b4fc;">Bias Dimensions</p>
    {rows}
</div>"""

    elif bias_types:
        tags = "".join(
            f'<span style="font-size:12px; color:#4f46e5; background:#eef2ff; font-weight:500;'
            f'padding:3px 10px; border-radius:20px; border:1px solid #c7d2fe;'
            f'margin:0 5px 5px 0; display:inline-block;">{b}</span>'
            for b in bias_types
        )
        dims_html = f"""
<div style="margin-bottom:20px;">
    <p style="margin:0 0 9px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#a5b4fc;">Bias Types</p>
    <div style="display:flex; flex-wrap:wrap;">{tags}</div>
</div>"""

    # ── Flagged phrases ────────────────────────────────────────
    phrases = report.get("biased_phrases", [])
    phrases_html = ""
    if phrases:
        items = "".join(
            f'<div style="font-size:13px; font-style:italic; color:#3730a3; padding:6px 0;'
            f'border-bottom:1px solid #eef2ff;">&ldquo;{p}&rdquo;</div>'
            for p in phrases
        )
        phrases_html = f"""
<div style="margin-bottom:20px;">
    <p style="margin:0 0 8px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#a5b4fc;">Flagged Phrases</p>
    {items}
</div>"""

    # ── Reasoning ──────────────────────────────────────────────
    reasoning = report.get("reasoning", "")
    reasoning_html = ""
    if reasoning:
        reasoning_html = f"""
<div style="padding:13px 16px; background:#f5f3ff; border-left:3px solid #a5b4fc;
            border-radius:0 8px 8px 0; margin-bottom:20px;">
    <p style="margin:0 0 5px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#a5b4fc;">Reasoning</p>
    <p style="margin:0; font-size:14px; color:#3730a3; line-height:1.7;">{reasoning}</p>
</div>"""

    # ── Neutral rewrite ────────────────────────────────────────
    rewrite = report.get("neutral_rewrite", "")
    rewrite_html = ""
    if rewrite:
        rewrite_html = f"""
<div style="padding:13px 16px; background:#f0fdf4; border-left:3px solid #6ee7b7;
            border-radius:0 8px 8px 0; margin-bottom:20px;">
    <p style="margin:0 0 5px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#059669;">Neutral Rewrite</p>
    <p style="margin:0; font-size:14px; color:#065f46; line-height:1.7; font-style:italic;">{rewrite}</p>
</div>"""

    # ── Evidence ───────────────────────────────────────────────
    evidence = report.get("evidence", report.get("retrieved_evidence", []))
    evidence_html = ""
    if evidence:
        items = "".join(
            f'<li style="margin-bottom:6px; font-size:13px; color:#6366f1; line-height:1.55;">{e}</li>'
            for e in evidence
        )
        evidence_html = f"""
<div style="margin-bottom:20px;">
    <p style="margin:0 0 9px; font-size:10px; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#a5b4fc;">Evidence Used</p>
    <ul style="margin:0; padding-left:18px; line-height:1.6;">{items}</ul>
</div>"""

    # ── Score badge background ─────────────────────────────────
    score_badge_bg = f"linear-gradient(135deg, {status_color}18 0%, {status_color}0a 100%)"

    return f"""
<div style="
    font-family:{_FONT}; padding:22px;
    border:1px solid #e0e7ff; border-radius:14px;
    background:#ffffff; color:#1e1b4b;
    box-shadow:0 2px 16px rgba(79,70,229,0.07);">

    <!-- Header row: status + score -->
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; flex-wrap:wrap; gap:10px;">
        <div>
            <span style="
                display:inline-block; font-size:11px; font-weight:700;
                letter-spacing:.06em; text-transform:uppercase;
                color:{status_color}; background:{status_bg};
                padding:4px 12px; border-radius:20px; border:1px solid {status_color}30;">
                {status_label}
            </span>
            <p style="margin:6px 0 0; font-size:12px; color:#a5b4fc;">
                {report.get('input_type','unknown')} &middot; {report.get('mode','offline')}
            </p>
        </div>
        <div style="
            text-align:center; background:{score_badge_bg};
            padding:10px 18px; border-radius:12px; border:1px solid {status_color}20; flex-shrink:0;">
            <span style="font-size:32px; font-weight:700; color:{status_color};
                         line-height:1; letter-spacing:-.03em; display:block;">
                {score:.2f}
            </span>
            <span style="font-size:9px; color:{status_color}99; text-transform:uppercase;
                         letter-spacing:.1em; font-weight:600;">bias score</span>
        </div>
    </div>

    <!-- Score bar -->
    <div style="height:5px; background:#e0e7ff; border-radius:3px; margin-bottom:22px; overflow:hidden;">
        <div style="width:{bar_pct}%; height:100%; background:linear-gradient(90deg, {status_color}cc, {status_color});
                    border-radius:3px;"></div>
    </div>

    <!-- Target group -->
    <div style="margin-bottom:20px;">
        <p style="margin:0 0 5px; font-size:10px; font-weight:700; text-transform:uppercase;
                  letter-spacing:.08em; color:#a5b4fc;">Target Group</p>
        <p style="margin:0; font-size:14px; color:#3730a3; font-weight:500;">
            {report.get('target_group') or '—'}
        </p>
    </div>

    {dims_html}
    {reasoning_html}
    {phrases_html}
    {rewrite_html}
    {evidence_html}

    <!-- Footer -->
    <div style="border-top:1px solid #eef2ff; padding-top:12px; margin-top:4px;
                display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
        <span style="font-size:11px; color:#c7d2fe; font-weight:500;">
            Confidence&nbsp;<strong style="color:#a5b4fc;">{confidence:.0%}</strong>
        </span>
        <span style="font-size:11px; color:#c7d2fe; font-weight:600; letter-spacing:0.04em;">
            BiasLens
        </span>
    </div>
</div>"""


def pretty_print_json(report: dict) -> str:
    return json.dumps(report, indent=2)
