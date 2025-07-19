import argparse
import os
import markdown2
from weasyprint import HTML, CSS
from pygments.formatters import HtmlFormatter

def export_markdown_to_pdf(md_file_path):
    """Converts a markdown file to a styled PDF file with comprehensive markdown support."""
    
    # --- 1. Define output directory and create it if it doesn't exist ---
    pdf_output_dir = 'pdfs'
    os.makedirs(pdf_output_dir, exist_ok=True)

    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {md_file_path}")
        return

    # --- 2. Generate CSS for syntax highlighting from Pygments ---
    # Using the 'default' theme which is clean and professional
    pygments_css = HtmlFormatter(style='default').get_style_defs('.codehilite')

    # --- 3. Comprehensive CSS for perfect markdown rendering ---
    base_css = """
    @page {
        size: A4;
        margin: 2cm;
    }
    
    /* Base typography */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #24292e;
        background-color: #ffffff;
        font-size: 14px;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        line-height: 1.25;
        margin-top: 24px;
        margin-bottom: 16px;
        page-break-after: avoid;
    }
    
    h1 {
        font-size: 2em;
        border-bottom: 1px solid #eaecef;
        padding-bottom: .3em;
    }
    
    h2 {
        font-size: 1.5em;
        border-bottom: 1px solid #eaecef;
        padding-bottom: .3em;
    }
    
    h3 {
        font-size: 1.25em;
    }
    
    h4 {
        font-size: 1em;
    }
    
    h5 {
        font-size: .875em;
    }
    
    h6 {
        font-size: .85em;
        color: #6a737d;
    }
    
    /* Paragraphs */
    p {
        margin-top: 0;
        margin-bottom: 16px;
    }
    
    /* Lists - CRITICAL: Explicit list-style-type to ensure markers appear */
    ul {
        list-style-type: disc !important;
        padding-left: 2em;
        margin-top: 0;
        margin-bottom: 16px;
    }
    
    ol {
        list-style-type: decimal !important;
        padding-left: 2em;
        margin-top: 0;
        margin-bottom: 16px;
    }
    
    li {
        margin-bottom: 0.25em;
        display: list-item !important;
    }
    
    li > p {
        margin-top: 16px;
        margin-bottom: 0;
    }
    
    li + li {
        margin-top: 0.25em;
    }
    
    /* Nested lists */
    ul ul, ol ol, ul ol, ol ul {
        margin-top: 0;
        margin-bottom: 0;
    }
    
    /* Code */
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        padding: .2em .4em;
        margin: 0;
        font-size: 85%;
        background-color: rgba(27,31,35,.05);
        border-radius: 3px;
    }
    
    pre {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        padding: 16px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        background-color: #f6f8fa;
        border-radius: 6px;
        margin-top: 0;
        margin-bottom: 16px;
        page-break-inside: avoid;
    }
    
    pre code {
        display: inline;
        max-width: auto;
        padding: 0;
        margin: 0;
        overflow: visible;
        line-height: inherit;
        word-wrap: normal;
        background-color: transparent;
        border: 0;
    }
    
    .codehilite {
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 16px;
        overflow: auto;
        margin-top: 0;
        margin-bottom: 16px;
        page-break-inside: avoid;
    }
    
    /* Links */
    a {
        color: #0366d6;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Tables */
    table {
        border-spacing: 0;
        border-collapse: collapse;
        width: 100%;
        margin-top: 0;
        margin-bottom: 16px;
        page-break-inside: avoid;
    }
    
    table th {
        font-weight: 600;
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        padding: 6px 13px;
    }
    
    table td {
        border: 1px solid #d0d7de;
        padding: 6px 13px;
    }
    
    table tr {
        background-color: #ffffff;
        border-top: 1px solid #c6cbd1;
    }
    
    table tr:nth-child(2n) {
        background-color: #f6f8fa;
    }
    
    /* Images */
    img {
        max-width: 100%;
        height: auto;
        box-sizing: content-box;
    }
    
    /* Blockquotes */
    blockquote {
        padding: 0 1em;
        color: #6a737d;
        border-left: .25em solid #dfe2e5;
        margin: 0 0 16px 0;
    }
    
    blockquote > :first-child {
        margin-top: 0;
    }
    
    blockquote > :last-child {
        margin-bottom: 0;
    }
    
    /* Horizontal rules */
    hr {
        height: .25em;
        padding: 0;
        margin: 24px 0;
        background-color: #e1e4e8;
        border: 0;
    }
    
    /* Task lists */
    .task-list-item {
        list-style-type: none;
    }
    
    .task-list-item-checkbox {
        margin: 0 .2em .25em -1.6em;
        vertical-align: middle;
    }
    
    /* Footnotes */
    .footnote {
        font-size: 0.8em;
        color: #6a737d;
    }
    
    /* Badges/Shields (common in README files) */
    img[src*="shields.io"], img[src*="badge"] {
        display: inline;
        margin: 0 2px;
        vertical-align: middle;
    }
    """
    
    combined_css = base_css + pygments_css

    # --- 4. Convert markdown to HTML with ALL possible extras ---
    # Enable every useful markdown2 extra for comprehensive support
    extras = [
        'fenced-code-blocks',  # ```code``` blocks
        'tables',              # | table | support |
        'footnotes',           # [^1] footnote support
        'header-ids',          # # Header {#id} support
        'smarty-pants',        # Smart quotes and dashes
        'code-friendly',       # Better code handling
        'cuddled-lists',       # Lists without blank lines
        'metadata',            # YAML front matter
        'nofollow',            # Add rel="nofollow" to links
        'pyshell',             # Python shell highlighting
        'spoiler',             # >! spoiler text support
        'strike',              # ~~strikethrough~~ support
        'tag-friendly',        # Better HTML tag handling
        'task_list',           # - [ ] checkbox support
        'wiki-tables',         # More table formats
        'xml'                  # Better XML/HTML handling
    ]
    
    # Convert with comprehensive extras
    html_body = markdown2.markdown(md_content, extras=extras)

    # --- 5. Wrap in proper HTML structure ---
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{os.path.splitext(os.path.basename(md_file_path))[0]}</title>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # --- 6. Define output path and generate PDF ---
    base_name = os.path.splitext(os.path.basename(md_file_path))[0]
    pdf_path = os.path.join(pdf_output_dir, f"{base_name}.pdf")

    # Generate PDF with proper HTML structure
    html = HTML(string=full_html, base_url=os.path.dirname(os.path.abspath(md_file_path)))
    css = CSS(string=combined_css)
    html.write_pdf(pdf_path, stylesheets=[css])

    print(f"Successfully exported {md_file_path} to {pdf_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a Markdown file to a styled PDF with comprehensive markdown support.')
    parser.add_argument('markdown_file', type=str, help='The path to the markdown file to convert.')
    args = parser.parse_args()

    export_markdown_to_pdf(args.markdown_file)
