CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
VEHICLE = {"airplane", "automobile", "ship", "truck"}
LIVING = {"bird", "cat", "deer", "dog", "frog", "horse"}
CAN_FLY = {"airplane", "bird"}
ANIMAL = LIVING

CAPTION_TEMPLATES = [
    "a photo of a {cls}.",
    "this image shows a {cls}.",
    "a small {cls} is visible.",
    "there is a {cls} in the picture.",
    "a CIFAR-10 image of a {cls}.",
    "the object is a {cls}.",
]


def make_captions(labels):
    rows = []
    for i, y in enumerate(labels.tolist()):
        cls = CIFAR_CLASSES[int(y)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "caption": CAPTION_TEMPLATES[i % len(CAPTION_TEMPLATES)].format(cls=cls)})
    return rows


def make_vqa(labels):
    rows = []
    for i, y in enumerate(labels.tolist()):
        cls = CIFAR_CLASSES[int(y)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "what_object", "question": "What object is shown?", "answer": cls})
        asked = cls if (i + int(y)) % 2 == 0 else CIFAR_CLASSES[(int(y) + 3) % len(CIFAR_CLASSES)]
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "is_class", "question": f"Is there a {asked}?", "answer": "yes" if asked == cls else "no"})
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "vehicle_living", "question": "Vehicle or living thing?", "answer": "vehicle" if cls in VEHICLE else "living"})
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "can_fly", "question": "Can it fly?", "answer": "yes" if cls in CAN_FLY else "no"})
        rows.append({"image_idx": i, "class_id": int(y), "class": cls, "template": "is_animal", "question": "Is this an animal?", "answer": "yes" if cls in ANIMAL else "no"})
    return rows

