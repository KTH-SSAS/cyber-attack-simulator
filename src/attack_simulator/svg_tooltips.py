import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

ET.register_namespace("", "http://www.w3.org/2000/svg")

SCRIPT = """
<script type="text/ecmascript">
<![CDATA[

function init(event) {
    if (window.svgDocument == null) {
        svgDocument = event.target.ownerDocument;
    }
}

function tooltip(object, visibility) {
    var id = object.id.split('_').slice(-1);
    var tip = svgDocument.getElementById('tooltip_' + id);
    tip.setAttribute('visibility', visibility)
}

]]>
</script>
"""

FALLBACK = "If you do NOT see an image here,<br/> Open This Frame in a New Tab/Window!"

BBOX = dict(boxstyle="round,pad=.5", color="white", alpha=0.75, zorder=100)


def add_tooltips(pos, tooltips, radius=0.3, ax=None):
    """Add trigger areas and tooltip text"""
    if ax is None:
        ax = plt.gca()
    for i in pos:
        # trigger area
        ax.add_patch(
            plt.Circle(
                pos[i],
                radius,
                gid=f"trigger_{i}",
                color="white",
                alpha=0.1,
                zorder=50,
            )
        )
        # tooltip text
        ax.annotate(
            tooltips[i],
            pos[i],
            gid=f"tooltip_{i}",
            xytext=(0, 20),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=BBOX,
            zorder=100,
        )


def postprocess_frame(filename, keys):
    """Add script and callbacks to SVG frame, hide tooltips by default"""
    tree = ET.ElementTree(file=filename)
    root = tree.getroot()
    root.set("onload", "init(event)")
    for key in keys:
        e = root.find(f'.//*[@id="trigger_{key}"]')
        e.set("onmouseover", "tooltip(this, 'visible')")
        e.set("onmouseout", "tooltip(this, 'hidden')")
        e = root.find(f'.//*[@id="tooltip_{key}"]')
        e.set("visibility", "hidden")
    root.insert(0, ET.XML(SCRIPT))
    tree.write(filename)


def postprocess_html(filename):
    """Change `img` to `object` to allow interactivity

    This hack relies on the contents of `matplotlib._animation_data`
    for the HTML source included by `matplotlib.animation.HTMLWriter`.
    """
    with open(filename) as saved:
        html = saved.read()
    html = html.replace("Image", "Object")
    html = html.replace("src", "data")
    html = re.sub(
        r"<img ([^>]*)>", rf'<object type="image/svg+xml" width="100%" \1>{FALLBACK}</object>', html
    )


def make_paths_relative(filename):
    # Make frames dir path relative to the episode's base directory
    with open(filename) as saved:
        html = saved.read()
    html = re.sub(r"render/.*/ep-\d+/", "", html)
    with open(filename, "w") as replace:
        replace.write(html)
