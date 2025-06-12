import os
import cv2
import numpy as np
from lxml import etree
from pathlib import Path
import matplotlib.pyplot as plt

xml_path = "annotations/annotations.xml"
images_dir = "images_viscon_processed/dl"
output_mask_dir = "annotations/masks"
overlay_dir = "annotations/overlays"
line_thickness = 10

Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
Path(overlay_dir).mkdir(parents=True, exist_ok=True)

tree = etree.parse(xml_path)
root = tree.getroot()

for image_tag in root.findall("image"):
    filename = image_tag.attrib["name"]
    width = int(image_tag.attrib["width"])
    height = int(image_tag.attrib["height"])
    base_name = Path(filename).stem

    img_path = os.path.join(images_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping missing image: {filename}")
        continue

    mask = np.zeros((height, width), dtype=np.uint8)

    for poly_tag in image_tag.findall("polyline"):
        points_str = poly_tag.attrib["points"]
        point_pairs = [
            tuple(map(float, p.split(",")))
            for p in points_str.strip().split(";")
        ]
        if len(point_pairs) >= 2:
            cv2.polylines(
                mask,
                [np.array(point_pairs, dtype=np.int32)],
                isClosed=False,
                color=255,
                thickness=line_thickness,
            )

    mask_path = os.path.join(output_mask_dir, base_name + ".png")
    cv2.imwrite(mask_path, mask)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]

    overlay_path = os.path.join(overlay_dir, base_name + ".png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("Masks and overlays generated.")