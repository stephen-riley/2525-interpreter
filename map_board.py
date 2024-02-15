from predict2525 import predict_2525
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import cv2
import logging as log
import numpy as np
import torch

PROJECT='679982020723'
ENDPOINT='1037686088946155520'
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75


log.basicConfig(
    format='[%(asctime)s %(levelname)-8s] %(message)s',
    level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

image = cv2.imread('simpler map.png')
size = image.size

log.debug("loading model...")
# DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')
DEVICE = torch.device('cpu')
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(DEVICE)      # small
# sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth").to(DEVICE)    # medium
# sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth").to(DEVICE)      # large

log.debug("Constructing generator...")
mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")
log.debug("Generating...")
masks = mask_generator.generate(image)


def color_dist(c):
    origin = np.array([0, 0, 0])
    color = np.array([c[0], c[1], c[2]])
    dist = np.linalg.norm(origin - color)
    return dist


def sc(c):
    return [int(c[0]), int(c[1]), int(c[2])]

data = {}

log.debug("Processing masks...")
ctr = 0
for m in masks:
    x, y, w, h = m["bbox"]
    ratio = w/h if h/w < 1.0 else h/w

    # if x > 850 and y < 250:
    #     print("here")

    if w > 90 and ratio < 1.2:
        contour_mask = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.uint8)
        contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        cv2.drawContours(
            contour_mask, [contour], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mean_color = cv2.mean(image, mask=contour_mask)
        d = color_dist(mean_color)

        mask_image = (m['segmentation'] * 255).astype(np.uint8)  # Convert to uint8 format
        cv2.imwrite('mask.png', mask_image)

        print()
        print(f"len = {len(m['segmentation'].tolist())}")
        # print(json.dumps(m["segmentation"].tolist(), indent=2))
        print()

        print(f"{m['bbox']}, mean {sc(mean_color)}, d {int(d)}")
        if d < 385:
            ctr += 1
            cropped = image[y:y+h, x:x+w]
            cv2.imwrite(f"post-it_{ctr}.png", cropped)
            prediction = predict_2525(PROJECT,ENDPOINT,f"post-it_{ctr}.png")

            cv2.bitwise_and(image, image, image, mask=mask_image)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)
            textsize = cv2.getTextSize(prediction, FONT, FONT_SCALE, 2)[0]
            tx = int(x+(w/2) - textsize[0]/2)
            ty = int(y - textsize[1] - 10)
            cv2.putText(image, prediction, (tx, ty), FONT, FONT_SCALE, (0,0,0), 2)
            cv2.imwrite('processed.png', image)

cv2.imwrite('processed.png', image)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()