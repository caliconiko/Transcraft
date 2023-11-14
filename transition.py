import cv2
from pathlib import Path
import numpy as np
from numpy.random import random 
from distutils.dir_util import copy_tree
from time import time

version = "1.19"

BLUE = "5bcffa"
PINK = "f5abb9"
WHITE = "ffffff"

here_dir = Path(__file__).resolve().parent

hor = cv2.imread(str(here_dir / "hor.png"))
ver = cv2.imread(str(here_dir / "ver.png"))

log = []
total = 0

def unpad(image:np.ndarray, amount = 1):
    return image[amount:-amount,amount:-amount]

def morph(img:np.ndarray, func=cv2.dilate, kernel_size=1, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return func(img, kernel, iterations)

def transify(image_path:Path):
    WINNAME = "Transition"

    # Get the image
    image_file = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    image_size = np.array([image_file.shape[1], image_file.shape[0]])

    # Get image channels
    image_channels = cv2.split(image_file)
    
    no_alpha = False
    if len(image_channels) <4:
        log.append(str(image_path) + " is sus")
        no_alpha=True
        # return image_file
    # Get alpha part of image
    if not no_alpha:
        image_alpha = cv2.inRange(image_channels[3], 1, 255)
    else:
        image_alpha = cv2.inRange(cv2.cvtColor(image_file, cv2.COLOR_RGB2GRAY), 1, 255)

    # Using FloodFill to remove holes in image
    padded_image_alpha = np.pad(image_alpha, ((1,1),(1,1)), "constant", constant_values=0)
    floodfill_image_alpha = np.zeros_like(padded_image_alpha)
    floodfill_mask = np.pad(padded_image_alpha, ((1,1),(1,1)), "constant", constant_values=0)
    cv2.floodFill(floodfill_image_alpha, floodfill_mask, (0,0), 255)

    # Trimming and inverting result of flood-filling
    floodfill_unpad = unpad(floodfill_image_alpha)
    image_alpha_no_holes = cv2.bitwise_not(floodfill_unpad)

    CORNER_PROCESS_SCALE = 10
    RECT_PROCESS_SCALE = 5
    # Enbiggen image for corner processing
    big_image = cv2.resize(image_alpha_no_holes, image_size*CORNER_PROCESS_SCALE, interpolation=cv2.INTER_NEAREST)
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
    corner_result = morph(corner_result, cv2.dilate, kernel_size=3, iterations=1)

    # Corners that intersect with the image
    good_corners = cv2.bitwise_and(corner_result, padded_big_im)
    good_corners = unpad(good_corners)

    # Ensmallen image
    smalled_image = cv2.resize(image_alpha_no_holes, image_size*RECT_PROCESS_SCALE, interpolation=cv2.INTER_NEAREST)

    # Put corners on small image
    corner_coords = np.array(np.where(good_corners == 255))
    corner_coords = corner_coords//(CORNER_PROCESS_SCALE//RECT_PROCESS_SCALE)

    # Get actual corners
    corner_co_im = np.zeros_like(smalled_image)
    corner_co_im[corner_coords[0],corner_coords[1]] = 255
    good_corner_co = np.transpose(np.where(corner_co_im == 255))

    # Make the corners EMIT
    corner_crosses = np.zeros_like(smalled_image)
    for co in good_corner_co:
        cross = np.zeros_like(smalled_image)
        cross[co[0]] = 255
        cross[::,co[1]] = 255
        
        mask = cv2.bitwise_not(cv2.bitwise_and(cross, smalled_image))
        mask = np.pad(mask, ((1,1),(1,1)), "constant", constant_values=255)
        target_cross = np.zeros_like(smalled_image)
        cv2.floodFill(target_cross, mask, (co[1],co[0]), 255)
        
        corner_crosses = cv2.bitwise_or(corner_crosses, target_cross)

    # Get rectangles in image form
    invert_quint_image = cv2.bitwise_not(smalled_image)
    rectangles_im = cv2.bitwise_not(cv2.bitwise_or(invert_quint_image, corner_crosses))


    # Find contours
    contours, _ = cv2.findContours(rectangles_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles
    rectangles = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        rectangles.append((x,y,w,h))

    # Draw rectangles
    rgb_smol_image = cv2.cvtColor(smalled_image, cv2.COLOR_GRAY2RGB)

    drawn_rectangles = np.zeros_like(rgb_smol_image)
    for rect in rectangles:
        x,y,w,h = rect
        # think about this print(rect) ^^
        unscaled_im = np.copy(hor) if w>=h else np.copy(ver)

        scaled_im = cv2.resize(unscaled_im, (w+2,h+2), interpolation=cv2.INTER_NEAREST)

        try:
            drawn_rectangles[y-1:y-1+scaled_im.shape[0], x-1:x-1+scaled_im.shape[1]] = scaled_im
        except Exception as e:
            # print(e)
            # cv2.imshow("drawn_rectangles", drawn_rectangles)
            # cv2.imshow("scaled_im", scaled_im)
            # cv2.waitKey(0)
            pass # ;)
    
    # cv2.imshow("output", cv2.resize(drawn_rectangles, image_size*RECT_PROCESS_SCALE*4, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("rect", cv2.resize(rectangles_im, image_size*RECT_PROCESS_SCALE*4, interpolation=cv2.INTER_NEAREST))

    # cv2.waitKey(0)

    # Mix with alpha
    rgba_smol_image = cv2.resize(image_file, image_size*RECT_PROCESS_SCALE, interpolation=cv2.INTER_NEAREST)
    drawn_rectangles = cv2.cvtColor(drawn_rectangles, cv2.COLOR_RGB2RGBA)
    rgba_smol_image = cv2.cvtColor(rgba_smol_image, cv2.COLOR_RGB2RGBA)

    drawn_rectangles[:,:,-1] = rgba_smol_image[:,:,-1]

    return drawn_rectangles


textures = [
    "block",
    "entity",
    "gui",
    "item",
    "map",
    "models",
    "mob_effect",
    "painting",
    "particle",
    "environment/moon_phases.png",
    "environment/rain.png",
    "environment/snow.png",
    "environment/sun.png",
    "environment/end_sky.png",
]

bad_textures = [
    "background",
]

TEXTURE_DIR = Path("assets/minecraft/textures")

version_dir = Path(version)

INPUT_VERSION_DIR = Path("input-versions")
OUTPUT_VERSION_DIR = Path("output-versions")

out_version_dir = here_dir/OUTPUT_VERSION_DIR/version_dir
in_version_dir = here_dir/INPUT_VERSION_DIR/version_dir
print(out_version_dir)
if out_version_dir.is_dir():
    print("it is a dir")
else:
    copy_tree(str(in_version_dir), str(out_version_dir))

def transize(path):
    global total
    transified = transify(path)
    path_parts = path.parts
    index = path_parts.index(str(INPUT_VERSION_DIR))

    out_path = Path(*(path_parts[:index]+(OUTPUT_VERSION_DIR,)+path_parts[index+1:])) 
    cv2.imwrite(str(out_path), transified)
    total+=1
    print(out_path)

    # cv2.imshow("image", transified)
    # key = cv2.waitKey(0)
    # if key>0:
    #     if chr(key) == 'q':
    #         exit()

FULL_TEXTURE_DIR = here_dir/INPUT_VERSION_DIR/version_dir/TEXTURE_DIR

start_time = time()

for texture in textures:
    full_texture_path = FULL_TEXTURE_DIR/texture

    if full_texture_path.is_dir():
        png_files = full_texture_path.rglob('*.png')
        for i, path in enumerate(png_files):
            found_bad_texture = False

            for bad_texture in bad_textures:
                if path.parent.name == bad_texture:
                    found_bad_texture = True
                    break
            
            if found_bad_texture:
                continue
            
            print(f"{path} [{i}]")
            transize(path)
    else:
        transize(full_texture_path)

print(f"{time()-start_time}")
print(log)
print(f"{total=}")