"""Render WRITEUP.md as a readable A4 PDF; optional reportlab/pypdf dependencies.

Run with the bundled artifact Python, or install reportlab and pypdf separately.
"""
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader


def main():
    repo = Path(__file__).resolve().parent.parent
    output = repo / "metrics" / "writeup.pdf"
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 7
    styles["Heading1"].fontSize = 16
    styles["Heading2"].fontSize = 11
    styles["Heading2"].spaceBefore = 9
    styles["Heading2"].spaceAfter = 5
    def fmt(text):
        return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html.escape(text))
    story = []
    for block in (repo / "WRITEUP.md").read_text(encoding="utf-8").split("\n\n"):
        if block.startswith("#"):
            heading, _, body = block.partition("\n")
            story.append(Paragraph(fmt(re.sub(r"^#+ ", "", heading)),
                                   styles["Heading1" if heading.startswith("# ") else "Heading2"]))
            block = body
        text = " ".join(block.splitlines()).strip()
        if not text:
            continue
        if block.startswith("|"):
            rows = [line for line in block.splitlines() if not re.match(r"\|[-: |]+\|$", line)]
            data = [[Paragraph(fmt(cell.strip()), styles["BodyText"]) for cell in row.strip("|").split("|")] for row in rows]
            table = Table(data, colWidths=[164, 106, 105, 104])
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf1f5")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend([table, Spacer(1, 7)])
        else:
            style = "Heading1" if text.startswith("# ") else "Heading2" if text.startswith("## ") else "BodyText"
            story.append(Paragraph(fmt(re.sub(r"^#+ ", "", text)), styles[style]))
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 48, 25, str(document.page))
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48,
                      topMargin=36, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
    count = len(PdfReader(output).pages)
    print(f"Rendered {count} pages: {output}")
    if count > 2:
        raise SystemExit("Write-up exceeds two pages")


if __name__ == "__main__":
    main()
