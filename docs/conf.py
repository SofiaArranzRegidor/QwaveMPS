# conf.py

project = "QwaveMPS Documentation"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
]

# Use index.md as the root
master_doc = "index"

# Only Markdown (keep it simple)
source_suffix = {".md": "markdown",
    ".rst": "restructuredtext",}

# MyST features similar to MkDocs markdown_extensions
myst_enable_extensions = [
    "colon_fence",   # ::: blocks
    "amsmath",       # better math blocks/environments
    "dollarmath",    # $...$ and $$...$$
    "deflist",
    "fieldlist",
]

exclude_patterns = ["_build", ".DS_Store"]
exclude_patterns += ["examplesV1.md"]


# Theme similar vibe to MkDocs readthedocs
html_theme = "sphinx_rtd_theme"   # pip install sphinx-rtd-theme

# MathJax config (optional; default usually fine)
# mathjax3_config = {"tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]}}


extensions += ["sphinx_gallery.gen_gallery"]

sphinx_gallery_conf = {
    # where your example scripts live (relative to conf.py)
    "examples_dirs": "../Examples",     # or "examples" if it’s inside docs/
    # where the built gallery pages should go (relative to conf.py)
    "gallery_dirs": "auto_examples",
   "ignore_pattern": r"^_",
    # Only run python files in Examples/ (and typically you name them Example_*.py)
    "filename_pattern": r"Example_.*\.py",
    "remove_config_comments": True,
}
