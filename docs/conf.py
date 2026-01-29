# conf.py

project = "QwaveMPS Documentation"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

# Use index.md as the root
master_doc = "index"

# Only Markdown (keep it simple)
source_suffix = {".md": "markdown"}

# MyST features similar to MkDocs markdown_extensions
myst_enable_extensions = [
    "colon_fence",   # ::: blocks
    "amsmath",       # better math blocks/environments
    "dollarmath",    # $...$ and $$...$$
    "deflist",
    "fieldlist",
]

exclude_patterns = ["_build", ".DS_Store"]

# Theme similar vibe to MkDocs readthedocs
html_theme = "sphinx_rtd_theme"   # pip install sphinx-rtd-theme

# MathJax config (optional; default usually fine)
# mathjax3_config = {"tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]}}
