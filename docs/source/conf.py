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
# from setuptools_scm import get_version

# sys.path.insert(0, os.path.abspath('../../jitcsim/examples/scripts'))
# sys.path.insert(0, os.path.abspath('../../jitcsim/models'))
sys.path.insert(0, os.path.abspath('../../jitcsim'))

# -- Project information -----------------------------------------------------

project = 'jitcsim'
copyright = '2021, Abolfazl Ziaeemehr'
author = 'Abolfazl Ziaeemehr'

# The full version, including alpha/beta/rc tags
release = '0.3'
# release = version = get_version(root='..', relative_to=__file__)


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
              'sphinx.ext.mathjax',
              #   'sphinx.ext.autosectionlabel'
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
html_theme = 'nature'


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


# Enable numref
numfig = True
