"""Format notebook math for LaTeX export without modifying the notebook."""
import re

from nbconvert.preprocessors import Preprocessor


class ReportMathPreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "markdown" or "mathjax-macros" in cell.metadata.get("tags", []):
            return cell, resources

        # Pandoc's Markdown reader reliably recognises dollar-delimited inline math.
        cell.source = re.sub(r"\\\((.*?)\\\)", r"$\1$", cell.source, flags=re.DOTALL)

        def align_iae(match):
            body = match.group(1)
            if r"\iae\nolimits" not in body or r"\begin{aligned}" in body:
                return match.group(0)
            # The source already separates the integral and MC approximation with
            # a line break; an aligned environment makes that break effective.
            body = body.strip()
            body = body.replace("=", "&=", 1)
            body = body.replace(r"\approx", r"&\approx", 1)
            return "$$\n" + r"\begin{aligned}" + "\n" + body + "\n" + r"\end{aligned}" + "\n$$"

        cell.source = re.sub(r"\$\$(.*?)\$\$", align_iae, cell.source, flags=re.DOTALL)
        return cell, resources


c = get_config()
c.Exporter.preprocessors = [ReportMathPreprocessor]
