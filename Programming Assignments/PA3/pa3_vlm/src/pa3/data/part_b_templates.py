SYN_CLASSES = ["spiral", "triangle", "circle", "cross", "checkerboard", "gradient"]
GEOMETRIC = {"triangle", "circle", "cross"}
SYMM_AXES = {"spiral": "0", "triangle": "3", "circle": "infinite", "cross": "4", "checkerboard": "4", "gradient": "1"}
IMG_PROMPTS = ["Generate a {cls} image.", "Draw a small {cls}.", "Create a 16 by 16 {cls}."]


def make_b_vqa(labels):
    rows = []
    for i, y in enumerate(labels.tolist()):
        cls = SYN_CLASSES[int(y)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "shape", "question": "What shape is in this image?", "answer": cls})
        asked = cls if (i + int(y)) % 2 == 0 else SYN_CLASSES[(int(y) + 2) % len(SYN_CLASSES)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "is_class", "question": f"Is there a {asked}?", "answer": "yes" if asked == cls else "no"})
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "geo", "question": "Geometric or non-geometric?", "answer": "geometric" if cls in GEOMETRIC else "non-geometric"})
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "symmetry", "question": "How many axes of symmetry?", "answer": SYMM_AXES[cls]})
    return rows


def make_img_prompts(labels):
    rows = []
    for i, y in enumerate(labels.tolist()):
        cls = SYN_CLASSES[int(y)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "prompt": IMG_PROMPTS[i % len(IMG_PROMPTS)].format(cls=cls)})
    return rows

