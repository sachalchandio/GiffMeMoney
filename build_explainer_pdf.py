# -*- coding: utf-8 -*-
"""
Generates GiffMeMoney-Explainer.pdf — a presentation-grade product explainer
covering: what it is, how it works end-to-end, why these quant models (Markowitz
& co.), how it beats manual investing, an FAQ, and an honest "what it lacks" gap
analysis (features + what's needed to be a production, money-making product).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    PageBreak, NextPageTemplate, KeepTogether, ListFlowable, ListItem, HRFlowable,
)

# ---- Brand palette --------------------------------------------------------
EMERALD      = colors.HexColor("#10b981")
EMERALD_DK   = colors.HexColor("#047857")
EMERALD_DEEP = colors.HexColor("#064e3b")
INDIGO       = colors.HexColor("#6366f1")
INDIGO_DK    = colors.HexColor("#4338ca")
INK          = colors.HexColor("#0f172a")
SLATE        = colors.HexColor("#334155")
MUTE         = colors.HexColor("#64748b")
LINE         = colors.HexColor("#e2e8f0")
BG_SOFT      = colors.HexColor("#f1f5f9")
BG_MINT      = colors.HexColor("#ecfdf5")
BG_INDIGO    = colors.HexColor("#eef2ff")
AMBER_BG     = colors.HexColor("#fffbeb")
AMBER_LN     = colors.HexColor("#f59e0b")
RED          = colors.HexColor("#dc2626")
WHITE        = colors.white

OUT = "GiffMeMoney-Explainer.pdf"
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

# ---- Styles ---------------------------------------------------------------
ss = getSampleStyleSheet()

def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

H1 = style("H1", fontName="Helvetica-Bold", fontSize=19, leading=23,
           textColor=EMERALD_DEEP, spaceBefore=4, spaceAfter=8)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
           textColor=INDIGO_DK, spaceBefore=12, spaceAfter=5)
H3 = style("H3", fontName="Helvetica-Bold", fontSize=11, leading=14,
           textColor=INK, spaceBefore=8, spaceAfter=3)
BODY = style("BODY", fontName="Helvetica", fontSize=10, leading=15,
             textColor=SLATE, alignment=TA_JUSTIFY, spaceAfter=6)
BODYL = style("BODYL", parent=BODY, alignment=TA_LEFT)
SMALL = style("SMALL", fontName="Helvetica", fontSize=8.5, leading=12, textColor=MUTE)
KICKER = style("KICKER", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
               textColor=EMERALD_DK, spaceAfter=2)
LEAD = style("LEAD", fontName="Helvetica", fontSize=11, leading=16,
             textColor=INK, alignment=TA_LEFT, spaceAfter=8)
BULLET = style("BULLET", parent=BODYL, fontSize=9.7, leading=14, spaceAfter=2)
CELL = style("CELL", fontName="Helvetica", fontSize=8.8, leading=12, textColor=SLATE)
CELLB = style("CELLB", parent=CELL, fontName="Helvetica-Bold", textColor=INK)
CELLW = style("CELLW", parent=CELL, textColor=WHITE, fontName="Helvetica-Bold")
QSTYLE = style("QSTYLE", fontName="Helvetica-Bold", fontSize=10.2, leading=14, textColor=INDIGO_DK, spaceAfter=2)
ASTYLE = style("ASTYLE", parent=BODYL, fontSize=9.7, leading=14, spaceAfter=10)

# Cover styles
COVER_TITLE = style("COVER_TITLE", fontName="Helvetica-Bold", fontSize=40, leading=44, textColor=WHITE)
COVER_SUB   = style("COVER_SUB", fontName="Helvetica", fontSize=13.5, leading=19, textColor=colors.HexColor("#d1fae5"))
COVER_TAG   = style("COVER_TAG", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#a7f3d0"))
COVER_FOOT  = style("COVER_FOOT", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#6ee7b7"))


# ---- Page furniture -------------------------------------------------------
def cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(EMERALD_DEEP)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # diagonal indigo band
    canvas.setFillColor(INDIGO_DK)
    p = canvas.beginPath()
    p.moveTo(0, PAGE_H * 0.36)
    p.lineTo(PAGE_W, PAGE_H * 0.50)
    p.lineTo(PAGE_W, PAGE_H * 0.40)
    p.lineTo(0, PAGE_H * 0.26)
    p.close()
    canvas.setFillAlpha(0.55)
    canvas.drawPath(p, fill=1, stroke=0)
    canvas.setFillAlpha(1)
    # emerald accent bar
    canvas.setFillColor(EMERALD)
    canvas.rect(MARGIN, PAGE_H - 150*mm, 46*mm, 2.6*mm, fill=1, stroke=0)
    # faux rising chart bars bottom-right
    canvas.setFillColor(EMERALD)
    canvas.setFillAlpha(0.85)
    bx = PAGE_W - MARGIN - 62*mm
    heights = [10, 16, 13, 22, 19, 30, 26, 38]
    for i, h in enumerate(heights):
        canvas.roundRect(bx + i*7.6*mm, 24*mm, 5.6*mm, h*mm, 1.4*mm, fill=1, stroke=0)
    canvas.setFillAlpha(1)
    canvas.restoreState()


def content_bg(canvas, doc):
    canvas.saveState()
    # header rule
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 13*mm, PAGE_W - MARGIN, PAGE_H - 13*mm)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.setFillColor(EMERALD_DK)
    canvas.drawString(MARGIN, PAGE_H - 11.4*mm, "GiffMeMoney")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(MUTE)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 11.4*mm, "Product Explainer  ·  Educational simulation")
    # footer
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 13*mm, PAGE_W - MARGIN, 13*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTE)
    canvas.drawString(MARGIN, 9.6*mm, "Not financial advice · No real money moves · Synthetic market data")
    canvas.drawRightString(PAGE_W - MARGIN, 9.6*mm, "Page %d" % doc.page)
    canvas.restoreState()


# ---- Reusable component builders -----------------------------------------
def chip_row(items):
    """A row of pill 'chips'."""
    cells = [Paragraph(t, CELLW) for t in items]
    t = Table([cells], colWidths=[(PAGE_W - 2*MARGIN) / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), INDIGO),
        ("INNERGRID", (0,0), (-1,-1), 4, WHITE),
        ("BOX", (0,0), (-1,-1), 4, WHITE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    return t


def callout(title, body, bg=BG_MINT, bar=EMERALD, title_color=EMERALD_DK):
    inner = [
        Paragraph(title, style("co_t", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=title_color, spaceAfter=3)),
        Paragraph(body, style("co_b", parent=BODYL, fontSize=9.4, leading=13.5)),
    ]
    t = Table([[inner]], colWidths=[PAGE_W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("LINEBEFORE", (0,0), (0,-1), 3, bar),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
    ]))
    return t


def kpi_band(stats):
    """stats: list of (number, label)."""
    cells = []
    for num, lab in stats:
        block = [
            Paragraph(num, style("kpi_n", fontName="Helvetica-Bold", fontSize=20, leading=22, textColor=EMERALD, alignment=TA_CENTER)),
            Paragraph(lab, style("kpi_l", fontName="Helvetica", fontSize=8.2, leading=10.5, textColor=colors.HexColor("#cbd5e1"), alignment=TA_CENTER)),
        ]
        cells.append(block)
    t = Table([cells], colWidths=[(PAGE_W - 2*MARGIN)/len(stats)]*len(stats))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), INK),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#1e293b")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 11),
        ("BOTTOMPADDING", (0,0), (-1,-1), 11),
    ]))
    return t


def styled_table(data, col_widths, header_bg=EMERALD_DK, zebra=True, body_align=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("LINEBELOW", (0,0), (-1,-1), 0.5, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                cmds.append(("BACKGROUND", (0,r), (-1,r), BG_SOFT))
    t.setStyle(TableStyle(cmds))
    return t


def section_title(num, text):
    tbl = Table(
        [[Paragraph(str(num), style("snum", fontName="Helvetica-Bold", fontSize=15, textColor=WHITE, alignment=TA_CENTER)),
          Paragraph(text, style("stext", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=EMERALD_DEEP))]],
        colWidths=[11*mm, PAGE_W - 2*MARGIN - 11*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), INDIGO),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (0,0), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (1,0), (1,0), 10),
        ("ROUNDEDCORNERS", [3,3,3,3]),
    ]))
    return tbl


def bullets(items, color=EMERALD):
    flow = []
    for it in items:
        flow.append(ListItem(Paragraph(it, BULLET), value="•", leftIndent=6))
    return ListFlowable(flow, bulletType="bullet", start="•", bulletColor=color,
                        leftIndent=14, bulletFontSize=9, spaceBefore=2, spaceAfter=6)


# ---- Build the story ------------------------------------------------------
story = []

# ===== COVER ===============================================================
story.append(Spacer(1, 58*mm))
story.append(Paragraph("QUANT-BACKED INVESTMENT ADVISOR  ·  PRODUCT EXPLAINER", COVER_TAG))
story.append(Spacer(1, 6))
story.append(Paragraph("GiffMeMoney", COVER_TITLE))
story.append(Spacer(1, 10))
story.append(Paragraph(
    "How it works, why it uses the math it does (Markowitz &amp; 72 other models), "
    "how it beats investing by hand — and an honest look at what it still needs.", COVER_SUB))
story.append(Spacer(1, 90*mm))
story.append(Paragraph("Prepared for end-users &amp; stakeholders", COVER_FOOT))
story.append(Paragraph("Educational simulation — not financial advice. No real money ever moves.", COVER_FOOT))
story.append(NextPageTemplate("content"))
story.append(PageBreak())

# ===== 1. WHAT IT IS =======================================================
story.append(section_title(1, "What GiffMeMoney is, in one minute"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "GiffMeMoney is an intelligent, quant-backed investment advisor that tells you "
    "<b>where</b> to invest and — just as importantly — <b>why</b>. For every asset it tracks, it runs "
    "<b>73 named quantitative finance models</b>, blends them into a single, plain-English recommendation, "
    "and projects expected returns across five time horizons with honest confidence bands and downside. "
    "You can then fund a sandbox wallet, split money across a data-driven basket, and watch live profit "
    "&amp; loss — all on a built-in market simulator that runs end-to-end with no sign-ups and no API keys.", LEAD))
story.append(Spacer(1, 4))
story.append(kpi_band([
    ("73", "Quant models per asset"),
    ("8", "Model families"),
    ("5", "Projection horizons"),
    ("24", "Assets simulated"),
    ("326", "Automated tests"),
]))
story.append(Spacer(1, 10))
story.append(Paragraph("Who it is for", H3))
story.append(bullets([
    "<b>Newer investors</b> who want to understand the reasoning behind a call, not just a buy/sell arrow.",
    "<b>Students &amp; hobbyist quants</b> who want real, from-scratch implementations of textbook finance models.",
    "<b>Builders &amp; stakeholders</b> evaluating finance tooling who need to see the engine, the formulas, and the honest downside.",
]))
story.append(callout(
    "The core promise",
    "Most apps show you a number. GiffMeMoney shows you the <b>73 models that produced the number</b>, their "
    "formulas, their realized backtest performance, and the explicit bear case — so the recommendation is "
    "transparent and auditable instead of a black box.",
    bg=BG_INDIGO, bar=INDIGO, title_color=INDIGO_DK))
story.append(PageBreak())

# ===== 2. THE PROBLEM ======================================================
story.append(section_title(2, "The problem it solves"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Investing by hand asks an ordinary person to do a professional's job. Doing it well means juggling "
    "dozens of trade-offs at once, every single day, across every asset you might hold:", BODY))
story.append(bullets([
    "<b>Too much information, no synthesis.</b> Price, momentum, valuation, fundamentals, risk — each lives in a different place and contradicts the others.",
    "<b>Emotion beats discipline.</b> People buy euphoria and sell panic. A model does not get scared or greedy.",
    "<b>No sense of the downside.</b> Most people can guess an upside; almost nobody quantifies how much they can lose, or how likely a loss is.",
    "<b>Poor diversification.</b> Picking five stocks you like is not the same as building a basket whose risks offset each other.",
    "<b>No feedback loop.</b> Without backtesting, you never learn whether a strategy actually worked or you just got lucky.",
]))
story.append(callout(
    "What GiffMeMoney does instead",
    "It runs the whole professional toolkit for you — technical, fundamental, valuation, statistical, factor, "
    "portfolio, risk and derivatives models — combines them into one reliability-weighted answer, quantifies "
    "the downside on every call, and lets you act on a diversified, math-optimized basket in a risk-free sandbox.",
    bg=BG_MINT, bar=EMERALD, title_color=EMERALD_DK))
story.append(Spacer(1, 6))

# ===== 3. END TO END =======================================================
story.append(section_title(3, "How it works — the end-to-end process"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Every recommendation flows through the same seven-stage pipeline. Nothing is hidden; each stage is "
    "inspectable in the app.", BODY))

pipeline = [
    ("1 · Data", "A deterministic market simulator generates multi-year price history (OHLCV) plus live ticks every second for 24 assets, and plausible fundamentals (earnings, free cash flow, dividends, book value). It needs no API keys, so the full experience runs in minutes. Real data providers (Finnhub, Polygon, CoinGecko, Binance) plug into the same interface later."),
    ("2 · Strategies", "For each asset, all 73 models run. Each emits a score from -100 to +100, a confidence level, the exact formula it used, the raw metrics, and a plain-English rationale."),
    ("3 · Composite", "The 73 signals are combined into one reliability-weighted composite, calibrated to a clear stance: STRONG&nbsp;BUY → BUY → HOLD → SELL → STRONG&nbsp;SELL. Models with better track records carry more weight."),
    ("4 · Projection", "A single projection engine forecasts expected return across 5 horizons (1 Day, 1 Week, 1 Month, 1 Year, 5 Years), each with a fat-tailed confidence band, probability of profit, bull / base / bear scenarios, and tail-risk (CVaR — the average loss in the worst outcomes)."),
    ("5 · Backtest", "Timing strategies are replayed over history and scored against simply buying and holding — 14 metrics including CAGR, Sharpe, Sortino, max drawdown and win-rate — so you see what actually worked, not what sounds good."),
    ("6 · Allocate", "The Allocation Advisor ranks the universe by composite score and runs a Markowitz optimization to build a diversified basket sized to your risk tolerance (conservative / balanced / aggressive)."),
    ("7 · Invest &amp; track", "You fund a simulated wallet (card validated, never charged), buy the basket, and watch real-time per-position and total profit &amp; loss stream live. No real order is ever placed."),
]
rows = [[Paragraph("Stage", CELLW), Paragraph("What happens", CELLW)]]
for name, desc in pipeline:
    rows.append([Paragraph("<b>" + name + "</b>", CELLB), Paragraph(desc, CELL)])
story.append(Spacer(1, 4))
story.append(styled_table(rows, [30*mm, PAGE_W - 2*MARGIN - 30*mm], header_bg=INDIGO_DK))
story.append(Spacer(1, 8))
story.append(callout(
    "One engine, consistent numbers",
    "The same projection engine drives both the headline expected-return table and the Monte-Carlo simulation, "
    "so the figures agree across the app to within about one percentage point. There is no second set of books.",
    bg=BG_SOFT, bar=MUTE, title_color=SLATE))
story.append(Spacer(1, 6))

# ===== 4. WHY THE MATH (MARKOWITZ) ========================================
story.append(section_title(4, "Why this math — Markowitz &amp; the 8 families"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "None of the models are decorative. Each family answers a different, well-known question that a serious "
    "investor has to answer — and each is a real, peer-reviewed technique implemented from scratch on "
    "numpy / scipy / pandas (no black-box libraries), so the formulas are genuine and auditable.", BODY))

story.append(Paragraph("Why Markowitz mean-variance optimization?", H2))
story.append(Paragraph(
    "Markowitz's <b>Modern Portfolio Theory</b> (Nobel Prize, 1990) is the single most important idea in the app, "
    "and it answers the question hand-investing almost always gets wrong: <b>given the assets I like, how much of "
    "each should I actually hold?</b>", BODYL))
story.append(bullets([
    "<b>It optimizes the basket, not the pick.</b> Choosing good assets is only half the job; Markowitz decides the <i>weights</i> that give the most expected return for a given level of risk.",
    "<b>It prices diversification mathematically.</b> Because assets do not move perfectly together, a blend can carry less risk than its individual parts. The optimizer exploits exactly those correlations — something almost impossible to do by eye.",
    "<b>It produces the efficient frontier.</b> The set of 'best possible' portfolios, so you can pick the trade-off that matches your risk tolerance instead of guessing.",
    "<b>It is the right tool for the job.</b> The Allocation Advisor and Portfolio Optimizer both use it to turn a list of high-scoring assets into a concrete, risk-sized set of weights — the exact step a human does poorly.",
]))
story.append(callout(
    "In plain English",
    "Markowitz is what turns 'these ten assets look good' into 'put 14% here, 9% there, 22% there' — the answer "
    "that actually controls how much you make and how much you can lose. That is why it sits at the center of the "
    "Invest experience.",
    bg=BG_MINT, bar=EMERALD, title_color=EMERALD_DK))

story.append(Paragraph("The eight families and what each is <i>for</i>", H2))
fam = [
    ("Technical", "25", "Reads price &amp; momentum: is the trend up or down, overbought or oversold?", "MACD, RSI, Bollinger %B, Supertrend, Ichimoku, ADX"),
    ("Fundamental", "12", "Is the underlying business financially healthy and safe?", "Piotroski F-Score, Altman Z-Score, Graham, dividend safety"),
    ("Valuation", "8", "Is it cheap or expensive versus what it is actually worth?", "DCF, Gordon DDM, Magic Formula, owner-earnings yield"),
    ("Statistical", "8", "What do the numbers themselves imply about future moves &amp; regime?", "Monte Carlo (GBM), GARCH volatility, mean-reversion, pairs"),
    ("Factor", "7", "Which proven return drivers (size, value, quality) does it load on?", "CAPM, Fama–French 3 &amp; 5-factor, quality-minus-junk"),
    ("Portfolio", "7", "How should the whole basket be weighted to balance risk?", "Markowitz, risk parity, min-variance, all-weather, vol-target"),
    ("Risk-Adjusted", "5", "Is the return worth the risk, and how much should I bet?", "Sharpe, Sortino, VaR/CVaR, Kelly criterion, 12-1 momentum"),
    ("Derivatives", "1", "What is fair option pricing and sensitivity?", "Black–Scholes price + greeks + implied volatility"),
]
rows = [[Paragraph("Family", CELLW), Paragraph("#", CELLW), Paragraph("Question it answers", CELLW), Paragraph("Examples", CELLW)]]
for n, c, q, ex in fam:
    rows.append([Paragraph("<b>"+n+"</b>", CELLB), Paragraph(c, CELL), Paragraph(q, CELL), Paragraph(ex, CELL)])
story.append(Spacer(1, 2))
story.append(styled_table(rows, [26*mm, 8*mm, 64*mm, PAGE_W - 2*MARGIN - 98*mm], header_bg=EMERALD_DK))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "No single family is reliable alone — momentum misses value, value misses momentum, and neither sees risk. "
    "Blending all eight is precisely what a disciplined institutional desk does, and what an individual realistically cannot.", SMALL))
story.append(PageBreak())

# ===== 5. PROJECTIONS & HONESTY ===========================================
story.append(section_title(5, "Honest forecasts, not hype"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "A forecast that only shows upside is marketing, not analysis. GiffMeMoney's projection engine was "
    "deliberately hardened against over-optimism with a set of rigor rules, so every number is believable and "
    "every call shows its downside.", BODY))
rigor = [
    ("Realistic magnitudes", "Forecasts are shrunk toward a sensible market prior and capped — no fantasy +800% projections."),
    ("Credible bands", "Confidence ranges are floored and capped to believable bounds; multi-year totals are shown next to their annualized (CAGR) equivalent."),
    ("Consistent engine", "The analysis page and the Monte-Carlo simulation come from one engine and agree to ~1 percentage point."),
    ("Calibrated confidence", "Confidence varies across ~0.2–0.9 based on signal agreement, data quality, model reliability and how clear the market regime is."),
    ("A real stance mix", "The reliability-weighted blend yields a realistic spread of calls — not everything is a lukewarm HOLD."),
    ("Downside on every call", "Each analysis states the probability of loss, 1-year CVaR (expected shortfall), max drawdown, and an explicit bear case."),
    ("Unambiguous units", "Every figure is labelled — daily vs annual vs horizon — so nothing is silently mis-scaled."),
    ("Clean &amp; disclaimed", "No invalid numbers ever reach the screen, and a standard 'not financial advice' disclaimer rides along with every result."),
]
rows = [[Paragraph("Rigor guarantee", CELLW), Paragraph("What it means for you", CELLW)]]
for t, d in rigor:
    rows.append([Paragraph("<b>"+t+"</b>", CELLB), Paragraph(d, CELL)])
story.append(styled_table(rows, [46*mm, PAGE_W - 2*MARGIN - 46*mm], header_bg=INDIGO_DK))
story.append(Spacer(1, 8))
story.append(callout(
    "Why this matters",
    "The honesty <i>is</i> the product. Anyone can print a big green number. A tool you can trust shows the same "
    "call's probability of loss and worst-case shortfall in the same breath — which is exactly what lets you size "
    "a position sensibly.",
    bg=AMBER_BG, bar=AMBER_LN, title_color=colors.HexColor("#b45309")))
story.append(PageBreak())

# ===== 6. BETTER THAN MANUAL ==============================================
story.append(section_title(6, "Why it beats investing by hand"))
story.append(Spacer(1, 8))
comp = [
    ("Breadth of analysis", "A few indicators, checked occasionally, by memory.", "73 models on every asset, recomputed continuously."),
    ("Synthesis", "Conflicting signals you have to reconcile yourself.", "One reliability-weighted composite with a plain-English why."),
    ("Emotion", "Fear &amp; greed drive the timing.", "Rules and math — no panic, no euphoria."),
    ("Downside awareness", "Usually ignored or guessed.", "Probability of loss, CVaR, drawdown and a bear case on every call."),
    ("Diversification", "Hand-picked, by gut feel.", "Markowitz-optimized weights sized to your risk tolerance."),
    ("Evidence", "'It felt like it worked.'", "Backtests against buy-and-hold with 14 realized metrics."),
    ("Transparency", "Opaque tips from forums / influencers.", "Every formula, source and metric visible and auditable."),
    ("Time &amp; cost", "Hours of manual research.", "Seconds, free, with no sign-up or data vendor."),
]
rows = [[Paragraph("Dimension", CELLW), Paragraph("Manual investing", CELLW), Paragraph("With GiffMeMoney", CELLW)]]
for dim, man, gmm in comp:
    rows.append([Paragraph("<b>"+dim+"</b>", CELLB), Paragraph(man, CELL), Paragraph(gmm, style("g", parent=CELL, textColor=EMERALD_DK))])
t = Table(rows, colWidths=[36*mm, (PAGE_W-2*MARGIN-36*mm)/2, (PAGE_W-2*MARGIN-36*mm)/2], repeatRows=1)
cmds = [
    ("BACKGROUND", (0,0), (0,0), SLATE),
    ("BACKGROUND", (1,0), (1,0), MUTE),
    ("BACKGROUND", (2,0), (2,0), EMERALD_DK),
    ("TEXTCOLOR", (0,0), (-1,0), WHITE),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ("LINEBELOW", (0,0), (-1,-1), 0.5, LINE),
    ("LINEAFTER", (1,0), (1,-1), 0.6, LINE),
]
for r in range(1, len(rows)):
    if r % 2 == 0:
        cmds.append(("BACKGROUND", (0,r), (1,r), BG_SOFT))
        cmds.append(("BACKGROUND", (2,r), (2,r), BG_MINT))
    else:
        cmds.append(("BACKGROUND", (2,r), (2,r), colors.HexColor("#f6fefb")))
t.setStyle(TableStyle(cmds))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The point is not that the model is always right — no model is. It is that the model is <b>consistent, "
    "diversified, downside-aware and transparent</b>, which is exactly where unaided human investing tends to break down.", BODY))
story.append(PageBreak())

# ===== 7. FAQ =============================================================
story.append(section_title(7, "Basic questions, answered"))
story.append(Spacer(1, 8))
faqs = [
    ("Is this real money?",
     "No. Every wallet, deposit, withdrawal and trade is a simulated ledger entry. No card is ever charged and no order is ever placed with a real exchange. It is an educational sandbox by design."),
    ("Where does the market data come from?",
     "From a built-in deterministic simulator. Price history is seeded from a hash of each ticker, so every run is reproducible and testable. Real data providers can be plugged in later behind the same interface."),
    ("Do I need API keys or a signup to try it?",
     "No. The whole app runs end-to-end with zero configuration. There is also a one-click demo account, and anonymous sandbox access works too."),
    ("How can it run 73 models — are they real?",
     "Yes. They are all named, literature-backed techniques (CAPM, GARCH, Black–Scholes, Markowitz, Piotroski, and so on) implemented from scratch on numpy/scipy. You can read each formula and its sources in the Strategy Lab."),
    ("Can I trust the projections?",
     "Treat them as honest model estimates, not guarantees. They are deliberately de-hyped (capped magnitudes, calibrated confidence) and every one is shown with its probability of loss and worst-case shortfall."),
    ("What does 'composite score' mean?",
     "It is the reliability-weighted blend of all 73 model signals for an asset, mapped to a clear stance from STRONG&nbsp;BUY to STRONG&nbsp;SELL. Better-performing models count for more."),
    ("Is my data / login secure?",
     "Auth uses salted PBKDF2-SHA256 password hashing and signed JWTs, with per-user wallets. Raw card numbers are never stored — only a masked form. (It is still a demo-grade implementation; see the gaps section.)"),
    ("Is this financial advice?",
     "No — explicitly not. It is a learning and simulation tool. Do not make real investment decisions based on it. A disclaimer accompanies every projection in the app."),
]
for q, a in faqs:
    block = [Paragraph("Q.  " + q, QSTYLE), Paragraph("A.  " + a, ASTYLE)]
    story.append(KeepTogether(block))
story.append(PageBreak())

# ===== 8. WHAT IT LACKS (features) ========================================
story.append(section_title(8, "What it lacks — honest gap analysis"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Stakeholders deserve candour. Below is what the product does <b>not</b> yet do. These are not bugs — they are "
    "the deliberate edges of an educational simulation, and they map directly to a roadmap.", BODY))

story.append(Paragraph("A. Missing or thin product features", H2))
gaps_feat = [
    ("Real market data", "Runs on a synthetic simulator. The adapters for Finnhub / Polygon / CoinGecko / Binance exist as interfaces but real, validated live feeds are not the default."),
    ("Limited universe", "Only ~24 assets across equities, crypto and ETFs. No bonds, options chains, FX, futures, or international coverage yet."),
    ("No news / sentiment", "No earnings-call, news-flow, or social-sentiment signals — all models are price/fundamental/statistical."),
    ("No alerts or notifications", "No price/score alerts, no email/push when a stance flips or a target is hit."),
    ("No tax / fees modelling", "No commissions, slippage, spreads, or tax-lot accounting — so simulated returns are idealized."),
    ("Thin personalization", "Risk tolerance is a 3-way switch; no goals, time-horizon planning, or full risk-profiling questionnaire."),
    ("No mobile app", "Responsive web only — no native iOS / Android app."),
    ("No portfolio rebalancing", "It can build a basket but does not yet schedule or suggest ongoing rebalances/drift correction."),
]
rows = [[Paragraph("Area", CELLW), Paragraph("Gap", CELLW)]]
for a, g in gaps_feat:
    rows.append([Paragraph("<b>"+a+"</b>", CELLB), Paragraph(g, CELL)])
story.append(styled_table(rows, [42*mm, PAGE_W - 2*MARGIN - 42*mm], header_bg=EMERALD_DK))
story.append(PageBreak())

# ===== 9. WHAT IT LACKS TO BE PRODUCTION / MONEY-MAKING ====================
story.append(section_title(9, "What it needs to be a real, money-making product"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The codebase already includes go-live <i>scaffolding</i> — pluggable live-data adapters, a broker execution "
    "layer (paper-first), opt-in database persistence, and Docker deployment — but live trading is intentionally "
    "hard-gated OFF. To become a genuine commercial product, these are the pieces that must be finished, "
    "hardened, and (critically) made legal:", BODY))

story.append(Paragraph("A. To become functionally 'real'", H2))
story.append(bullets([
    "<b>Wire up real market data for production</b> — finish and validate the live adapters, with caching, rate-limit handling, and graceful fallback, as the default path (not the simulator).",
    "<b>Real brokerage integration</b> — the Alpaca broker layer exists and defaults to paper; finishing live order routing safely (it is currently hard-gated behind multiple acknowledgements) is the single biggest step from 'toy' to 'real'.",
    "<b>Durable persistence at scale</b> — replace the default in-memory store (and demo SQLite) with a managed, backed-up production database.",
    "<b>Hardened authentication</b> — add email verification, rate limiting, password reset, MFA, and a real (rotated) secret instead of the dev default.",
    "<b>Payments that actually move money</b> — integrate a real PSP (Stripe / Plaid) behind the existing PaymentProvider interface, with KYC/AML onboarding.",
    "<b>Observability &amp; reliability</b> — logging, monitoring, alerting, error tracking, backups, and a tested disaster-recovery path.",
]))

story.append(Paragraph("B. To become legally able to make money", H2))
story.append(callout(
    "This is the gating item, not the code",
    "Recommending securities or routing real orders is a <b>regulated activity</b>. In most jurisdictions this requires "
    "registration (e.g. as an investment adviser or broker-dealer), compliance, suitability rules, audited disclosures, "
    "and data-protection obligations. No amount of engineering substitutes for this; it must be solved before a single "
    "real dollar is touched.",
    bg=colors.HexColor("#fef2f2"), bar=RED, title_color=RED))
story.append(Spacer(1, 6))

story.append(Paragraph("C. A plausible business model", H2))
biz = [
    ("Freemium SaaS", "Free educational tier; paid tier unlocks real-data analysis, more assets, alerts, and advanced optimization."),
    ("Subscription advisory", "Monthly fee for the recommendation engine + portfolio monitoring (requires adviser registration)."),
    ("Brokerage / PFOF", "Earn on execution or payment-for-order-flow once a real broker integration is live and licensed."),
    ("AUM fee (robo-advisor)", "A small % of assets under management for automated, rebalanced, Markowitz-optimized portfolios."),
    ("B2B / white-label", "License the 73-model engine and projection API to other fintechs and banks."),
    ("Data &amp; API", "Sell the composite scores, backtests and projections as an API to quants and builders."),
]
rows = [[Paragraph("Path to revenue", CELLW), Paragraph("How it would work", CELLW)]]
for p, h in biz:
    rows.append([Paragraph("<b>"+p+"</b>", CELLB), Paragraph(h, CELL)])
story.append(styled_table(rows, [46*mm, PAGE_W - 2*MARGIN - 46*mm], header_bg=INDIGO_DK))
story.append(Spacer(1, 8))
story.append(callout(
    "Bottom line",
    "The <b>engine</b> — 73 audited models, honest projections, Markowitz allocation, backtesting — is the hard, "
    "differentiated part, and it is built and tested. The remaining work is <b>productionizing the plumbing</b> "
    "(live data, broker, payments, scale) and <b>clearing the regulatory bar</b>. Engineering is the smaller half; "
    "licensing and compliance are the larger.",
    bg=BG_MINT, bar=EMERALD, title_color=EMERALD_DK))
story.append(PageBreak())

# ===== 10. SUMMARY / DISCLAIMER ===========================================
story.append(section_title(10, "In summary"))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "GiffMeMoney takes the full toolkit a professional investor would use — 73 quantitative models across eight "
    "families, anchored by Markowitz portfolio optimization — and makes it transparent, honest, and one click away. "
    "It tells you where to invest and shows its work; it quantifies the downside instead of hiding it; and it lets "
    "you act on a diversified, risk-sized basket in a completely safe sandbox. That combination — breadth, honesty, "
    "and auditability — is what makes it a better starting point than investing by gut feel.", LEAD))
story.append(Spacer(1, 4))
story.append(chip_row(["Transparent", "Diversified", "Downside-aware", "Auditable", "Free to try"]))
story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
story.append(Spacer(1, 8))
story.append(Paragraph("Important disclaimer", style("disc_h", fontName="Helvetica-Bold", fontSize=10.5, textColor=RED, spaceAfter=4)))
story.append(Paragraph(
    "GiffMeMoney is an <b>educational simulation on synthetic market data. It is NOT financial advice.</b> "
    "Projections are model estimates, not guarantees. No real money moves — the wallet, deposits, withdrawals and "
    "trades are simulated ledger entries; no card is ever charged and no order is ever placed with any real venue. "
    "Do not make real investment decisions based on this software.", style("disc_b", parent=BODYL, fontSize=9, textColor=MUTE)))

# ---- Document assembly ----------------------------------------------------
frame_cover = Frame(MARGIN, MARGIN, PAGE_W - 2*MARGIN, PAGE_H - 2*MARGIN, id="cover")
frame_content = Frame(MARGIN, 16*mm, PAGE_W - 2*MARGIN, PAGE_H - 32*mm, id="content")

doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=MARGIN, bottomMargin=MARGIN,
                      title="GiffMeMoney — Product Explainer",
                      author="GiffMeMoney")
doc.addPageTemplates([
    PageTemplate(id="cover", frames=[frame_cover], onPage=cover_bg),
    PageTemplate(id="content", frames=[frame_content], onPage=content_bg),
])
doc.build(story)
print("WROTE", OUT)
