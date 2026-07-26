from __future__ import annotations

project = "volas"
author = "Kael Zhang"
copyright = "Kael Zhang"

extensions = ["myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
exclude_patterns = ["README.md", "_build", "animated_gif"]

html_theme = "sphinx_rtd_theme"
html_title = "volas documentation"
html_show_sphinx = False
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_context = {
    "display_github": True,
    "github_user": "kaelzhang",
    "github_repo": "volas",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

myst_heading_anchors = 3
