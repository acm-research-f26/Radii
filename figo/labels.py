"""UMD mask label convention."""

BACKGROUND = 0
WALL = 1  # uterine wall (myometrium)
CAVITY = 2  # uterine cavity / endometrium
MYOMA = 3  # fibroid (all fibroids share this label)
CYST = 4  # nabothian cyst

NUM_CLASSES = 5
LABEL_NAMES = {WALL: "wall", CAVITY: "cavity", MYOMA: "myoma", CYST: "cyst"}
VALID_LABELS = frozenset(range(NUM_CLASSES))
