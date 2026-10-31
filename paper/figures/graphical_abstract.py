"""Graphical abstract for the MedIA submission: Figure 1 with the leaderboard as panel C (fig1_results.pdf,
built by fig1.py with FIG1C=results) instead of the participation map, so that the image is landscape and shows
the main result. Elsevier asks for >= 1328 px wide, longest side horizontal. Run from anywhere; needs pdftoppm."""
import os, subprocess, tempfile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
subprocess.run(["pdftoppm", "-r", "400", "-png", "-singlefile", os.path.join(HERE, "fig1_results.pdf"), os.path.join(tmp, "f1")], check=True)
im = Image.open(os.path.join(tmp, "f1.png")).convert("RGB")
im.save(os.path.join(HERE, "graphical_abstract.png"))
im.save(os.path.join(HERE, "graphical_abstract.pdf"), resolution=400)
print(im.size)
