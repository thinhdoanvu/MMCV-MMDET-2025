import cv2
import json
import matplotlib.pyplot as plt

# ===============================
# Paths
# ===============================
image_path = r"C:\Users\VU\Documents\OBD\AICUP25\images\train\patient0001_0236.png"
annotation_json = r"C:\Users\VU\Documents\OBD\AICUP25\labels\train\_annotations.coco.json"
with open(annotation_json, 'r') as f:
    annotations = json.load(f)
image_filename = "patient0001_0236.png"

# ===============================
# Load image
# ===============================
image = cv2.imread(image_path)
h, w, _ = image.shape

# ===============================
# Load COCO annotations
# ===============================
with open(annotation_json, "r", encoding="utf-8") as f:
    coco = json.load(f)

# ===============================
# Get image_id
# ===============================
image_id = None
for img in coco["images"]:
    if img["file_name"] == image_filename:
        image_id = img["id"]
        break

if image_id is None:
    raise ValueError("Image not found in COCO annotations")

# ===============================
# Category mapping
# ===============================
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

# ===============================
# Colors (BGR)
# ===============================
colors = {
    1: (0, 0, 0),  # class 1
}

# ===============================
# Draw bounding boxes
# ===============================
for ann in coco["annotations"]:
    if ann["image_id"] != image_id:
        continue

    x, y, bw, bh = ann["bbox"]
    xmin, ymin = int(x), int(y)
    xmax, ymax = int(x + bw), int(y + bh)

    class_id = ann["category_id"]
    class_name = cat_id_to_name.get(class_id, str(class_id))

    color = colors.get(class_id, (0, 0, 0))

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

    # ---- Draw label ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
    text_y = ymin - 5 if ymin - 5 > th else ymin + th + 5

    cv2.putText(
        image,
        class_name,
        (xmin, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

# ===============================
# Show & save
# ===============================
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
# plt.savefig("IP020000341_gt_coco.jpg", bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()
