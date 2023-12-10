import os
import sys
from setuptools_scm import get_version

sys.path.insert(0, os.path.abspath('../jitcsim'))

project = 'jitcsim'
copyright = '2021, Abolfazl Ziaeemehr'
author = 'Abolfazl Ziaeemehr'

needs_sphinx = '1.3'
release = version = get_version(root='..', relative_to=__file__)

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
              'sphinx.ext.graphviz',
              #   'sphinx.ext.autosectionlabel'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build']
master_doc = 'index'
pygments_style = 'colorful'
html_theme = 'nature'
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
numfig = True

def on_missing_reference(app, env, node, contnode):
	if node['reftype'] == 'any':
		return contnode
	else:
		return None

def setup(app):
	app.connect('missing-reference', on_missing_reference)
