extensions = [
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

myst_enable_extensions = [
    "colon_fence",   # ::: blocks (MkDocs-style)
    "deflist",       # definition lists
    "fieldlist",     # :param:, :returns: style fields
    "amsmath",       # better math support
    "dollarmath",    # $...$ and $$...$$
]

extensions += [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # Google / NumPy docstrings
]
