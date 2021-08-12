# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import stanford_theme

# sys.path.insert(0, os.path.abspath('../../jitcsim'))
sys.path.insert(0, os.path.abspath('../../jitcsim'))

# -- Project information -----------------------------------------------------

project = 'jitcsim'
copyright = '2021, Abolfazl Ziaeemehr'
author = 'Abolfazl Ziaeemehr'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
              # 'sphinxcontrib.plantuml',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.imgmath',
              'sphinx.ext.mathjax'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'nature' #'stanford_theme'
# html_theme_path = [stanford_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '12pt',

#     'fontpkg': r"""
# \PassOptionsToPackage{bookmarksnumbered}{hyperref}
# """,

    # Additional stuff for the LaTeX preamble.
    'preamble': r"""
\usepackage{setspace}
""",

#     'footer': r"""
# """,

#     'maketitle': r'''
# \pagenumbering{arabic}
# ''',
}
