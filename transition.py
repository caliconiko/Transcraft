import cv2
from pathlib import Path
import numpy as np

version = "22w14a"

BLUE = "5bcffa"
PINK = "f5abb9"
WHITE = "ffffff"

def unpad(image:np.ndarray, amount = 1):
    return image[amount:-amount,amount:-amount]

def morph(img:np.ndarray, func=cv2.dilate, kernel_size=1, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return func(img, kernel, iterations)

def transify(image_path:Path):
    WINNAME = "Transition"

    # Get the image
    image_file = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    image_size = np.array([image_file.shape[i] for i in range(2)])

    # Get image channels
    image_channels = cv2.split(image_file)

    # Get alpha part of image
    image_alpha = cv2.inRange(image_channels[3], 1, 255)

    # Using FloodFill to remove holes in image
    padded_image_alpha = np.pad(image_alpha, ((1,1),(1,1)), "constant", constant_values=0)
    floodfill_image_alpha = np.zeros_like(padded_image_alpha)
    floodfill_mask = np.pad(padded_image_alpha, ((1,1),(1,1)), "constant", constant_values=0)
    cv2.floodFill(floodfill_image_alpha, floodfill_mask, (0,0), 255)

    # Trimming and inverting result of flood-filling
    floodfill_unpad = unpad(floodfill_image_alpha)
    image_alpha_no_holes = cv2.bitwise_not(floodfill_unpad)

    SCALE = 10

    big_image = cv2.resize(image_alpha_no_holes, image_size*SCALE, interpolation=cv2.INTER_NEAREST)
    padded_big_im = np.pad(big_image, ((1,1),(1,1)), "constant", constant_values=0)

    corner_result = np.zeros_like(padded_big_im)

    gray = np.float32(padded_big_im)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    corner_result[res[:,1],res[:,0]]=255
    corner_result = morph(corner_result, cv2.dilate, kernel_size=2, iterations=1)

    good_corners = cv2.bitwise_and(corner_result, padded_big_im)
    good_corners = unpad(good_corners)

    quint_image = cv2.resize(image_alpha_no_holes, image_size*5, interpolation=cv2.INTER_NEAREST)

    color_q_image = cv2.cvtColor(quint_image, cv2.COLOR_GRAY2BGR)

    corner_coords = np.array(np.where(good_corners == 255))
    corner_coords = corner_coords//2

    print(corner_coords)

    color_q_image[corner_coords[0], corner_coords[1]] = (255,0,0)

    dst = morph(dst, cv2.dilate, 1, 1)
    cv2.imshow(WINNAME, color_q_image)
    cv2.waitKey(0)


textures = [
    "entity/bee/bee.png", 
    "entity/allay/allay.png", 
    # "misc/spyglass_scope.png",
]

here_dir = Path(__file__).resolve().parent

TEXTURE_DIR = Path("assets/minecraft/textures")
INPUT_VERSION_DIR = Path("input-versions")

for texture in textures:
    print(texture)
    texture_path = here_dir.joinpath(INPUT_VERSION_DIR, version, TEXTURE_DIR, texture)
    transify(texture_path)