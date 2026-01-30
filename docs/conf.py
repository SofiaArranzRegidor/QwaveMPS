# conf.py

project = "QwaveMPS Documentation"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Config for autoapi
autoapi_member_order = "alphabetical"
autoapi_dirs = ["../QwaveMPS/"]
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autodoc_typehints = "description"
autodoc_typehints = "signature"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["__init__.py", ""]


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


#'''
# -- Custom configuration ----------------------------------------------------
# Skip modules in the autoapi extension to avoid duplication errors
def skip_modules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip

def setup(app):
    app.connect("autoapi-skip-member", skip_modules)
    app.add_css_file("hide_links.css")  # Custom CSS to hide jupyter links
#'''