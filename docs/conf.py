# conf.py

project = "QwaveMPS"

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
autoapi_dirs = ["../src/QwaveMPS/"]
autoapi_type = "python"
autoapi_keep_files = False

autoapi_member_order = "alphabetical"
#autoapi_member_order = "bysource"

autoapi_add_toctree_entry = False
autoapi_ignore_module_all = True
#autoapi_ignore = ["__init__.py"]

#autodoc_typehints = "description" # No typehint in the function signature in the API, shorter
autodoc_typehints = "signature" # Put the typehint in the function signature in the API

autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
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

exclude_patterns = ["_build", "features_to_add.txt", ".DS_Store"]

# Theme 
html_theme = "pydata_sphinx_theme" 

html_theme_options = {
    # Top-right icon links (GitHub, etc.)
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SofiaArranzRegidor/QwaveMPS",
            "icon": "fa-brands fa-github",
        },
    ],

}

html_context = {
    #"github_user": "<your-org-or-user>",
    "github_repo": "QwaveMPS",
    "github_version": "main",
    "doc_path": "docs",  # change if your conf.py is elsewhere
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]


# MathJax config (optional; default usually fine)
# mathjax3_config = {"tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]}}

from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # where your example scripts live (relative to conf.py)
    "examples_dirs": "../Examples",     # or "examples" if it’s inside docs/
    # where the built gallery pages should go (relative to conf.py)
    "gallery_dirs": "auto_examples",
   "ignore_pattern": r"^_",
    # Only run python files in Examples/ (and typically you name them Example_*.py)
    "filename_pattern": r"Example_.*\.py",
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
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