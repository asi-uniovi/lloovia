import jinja2
import os
import sys

if len(sys.argv)<2:
    print("Usage: %s filename.pgf" % sys.argv[0])
    sys.exit(0)

filename = sys.argv[1]


latex_jinja_env = jinja2.Environment(
        block_start_string = '((*',
        block_end_string = '*))',
        variable_start_string = '(((',
        variable_end_string = ')))',
        comment_start_string = '((=',
        comment_end_string = '=))',
        autoescape = False,
        loader = jinja2.FileSystemLoader(os.path.abspath('.'))
        )
template = latex_jinja_env.get_template("fig-template-latex.jinja")

if "cost" in filename:
    fix_dollars = r"\everymath{\$}"
else:
    fix_dollars = ""
print (template.render(filename = filename, fix_dollars = fix_dollars))
