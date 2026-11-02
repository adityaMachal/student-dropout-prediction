#!/usr/bin/env bash
# INT315 Cluster Computing - LaTeX Report Compilation Script
set -e

cd "$(dirname "$0")"

echo "=========================================================="
echo "📄 Compiling INT315 Project Report (report.tex -> report.pdf)"
echo "=========================================================="

# Pass 1: Initial compilation and aux generation
echo "▶ Running pdflatex (Pass 1)..."
pdflatex -interaction=nonstopmode report.tex > /dev/null

# Pass 2: Resolve cross-references and table of contents
echo "▶ Running pdflatex (Pass 2)..."
pdflatex -interaction=nonstopmode report.tex

echo -e "\n✅ Compilation Successful! Generated PDF: report.pdf"
ls -lh report.pdf
