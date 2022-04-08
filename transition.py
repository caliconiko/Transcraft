import cv2
from pathlib import Path
import numpy as np

version = "22w14a"

BLUE = "5bcffa"
PINK = "f5abb9"
WHITE = "ffffff"

def transify(image_path:Path):
    WINNAME = "Transition"

    image_file = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    image_size = np.array([image_file.shape[i] for i in range(2)])

    quint_image = cv2.resize(image_file, image_size*5, interpolation=cv2.INTER_NEAREST)

    image_channels = cv2.split(quint_image)

    image_alpha = cv2.inRange(image_channels[3], 1, 255)

    cv2.imshow(WINNAME, image_alpha)
    cv2.waitKey(0)


textures = [
    "entity/bee/bee.png", 
    # "misc/spyglass_scope.png"
]

here_dir = Path(__file__).resolve().parent

TEXTURE_DIR = Path("assets/minecraft/textures")
INPUT_VERSION_DIR = Path("input-versions")

for texture in textures:
    print(texture)
    texture_path = here_dir.joinpath(INPUT_VERSION_DIR, version, TEXTURE_DIR, texture)
    transify(texture_path)