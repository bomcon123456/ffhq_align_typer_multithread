import os
from math import ceil
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool as Pool
from pickle5 import pickle
import os.path as osp
from natsort import natsorted
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2
import PIL.Image
import PIL.ImageFile
from PIL import Image
import scipy.ndimage
import dlib
from imutils import face_utils


# from lib.landmarks_pytorch import LandmarksEstimation
import face_alignment
import typer
from pathlib import Path
from batch_face import RetinaFace


IMAGE_EXT = (".jpg", ".jpeg", ".png")

p = "shape_predictor_68_face_landmarks.dat"
detector = RetinaFace(network="resnet50", gpu_id=0)
predictor = dlib.shape_predictor(p)


def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)


def adjust_coordinate(coord, new_shape, old_shape):
    ratio_x = float(old_shape[0] / new_shape[0])
    ratio_y = float(old_shape[1] / new_shape[1])

    coord = coord / np.array([ratio_x, ratio_y])
    return coord


def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def image_align_68(
    image,
    face_landmarks,
    output_size=1024,
    transform_size=4096,
    enable_padding=True,
):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    np.random.seed(12345)
    # bbox = cropByInputLM(
    #     image, face_landmarks, rescale=[1.4255, 2.0591, 1.6423, 1.3087]
    # )
    # cv2_shape = image.shape
    # print(cv2_shape)
    # print(bbox)
    # if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > cv2_shape[1] or bbox[3] > cv2_shape[0]:
    #     return None

    lm = face_landmarks
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    is_big = True
    if abs(eye_to_eye[0]) < 64 or abs(eye_to_mouth[1]) < 64:
        is_big = False

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.fromarray(image)
    original_size = img.size

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        lm = adjust_coordinate(lm, img.size, original_size)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        lm = lm - np.array([crop[0], crop[1]])
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        lm += np.array([pad[0], pad[1]])
        img = PIL.Image.fromarray(img, "RGB")
        quad += pad[:2]

    # Transform.
    cv2_image = np.array(img).copy()
    cv2_image = cv2_image[:, :, ::-1].copy()

    target = np.array(
        [
            (0, 0),
            (0, transform_size),
            (transform_size, transform_size),
            (transform_size, 0),
        ],
        np.float32,
    )
    M = cv2.getPerspectiveTransform(np.float32(quad + 0.5), target)
    transformed_image = cv2.warpPerspective(
        cv2_image, M, (transform_size, transform_size), cv2.INTER_LINEAR
    )

    lm = cv2.perspectiveTransform(np.expand_dims(lm, axis=1), M)  # Adjust landmarks
    lm = np.squeeze(lm, 1)

    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        original_size = transformed_image.shape
        transformed_image = cv2.resize(transformed_image, (output_size, output_size))
        lm = adjust_coordinate(lm, transformed_image.shape, original_size)

    img_np = np.array(img)

    return img_np, lm, is_big


def read_image_opencv(image_path):
    # Read image in BGR order
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("uint8")


def func(filepaths, input_dir: Path, output_dir: Path, size: int, idx=0):
    for img_file_ in tqdm(filepaths, desc=f"Process {idx}"):
        img_file = img_file_.as_posix()
        outpath_ = output_dir / img_file_.relative_to(input_dir)
        if outpath_.exists():
            continue
        # Open input image
        img = read_image_opencv(img_file)

        detected_faces = detector(img, cv=False)
        # For each detected face, find the landmark.
        for (i, (box, _, score)) in enumerate(detected_faces):
            if i > 0:
                break
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            is_big = True
            if w < 256 or h < 256:
                is_big = False
            if score < 0.98:
                continue
            outpath = (
                outpath_.parent
                / f"{'big' if is_big else 'small'}"
                / f"{outpath_.stem}_{i}.png"
            )
            outpath.parent.mkdir(exist_ok=True, parents=True)
            rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape).astype(np.float32)

            try:
                (img, lm, _) = image_align_68(
                    img,
                    np.asarray(shape),
                    output_size=size,
                )
            except Exception as e:
                continue
            # Save output image
            cv2.imwrite(
                outpath.as_posix(),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        if len(detected_faces) == 0:
            print("#. Warning: No landmarks found in {}".format(img_file))
            with open(
                f"issues_{idx}.txt", "a" if osp.exists("issues_{idx}.txt") else "w"
            ) as f:
                f.write("{}\n".format(img_file))


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="Input directory"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    size: int = typer.Option(1024, help="image size"),
    pickle_path: Path = typer.Option(None, help="Pickle path"),
    index: int = typer.Option(-1, help="index, -1 to run all"),
    nsplits: int = typer.Option(5, help="index, -1 to run all"),
):
    if pickle_path is None:
        cachedir = input_dir / "filelist.pickle"
    else:
        cachedir = pickle_path
    if cachedir.exists():
        with open(cachedir, "rb") as handle:
            filepaths = pickle.load(handle)
        filepaths = list(map(lambda x: input_dir / x, filepaths))
    else:
        filepaths = list(input_dir.rglob("*.[jp][pn]g"))
        filepaths = natsorted(filepaths)
        with open(cachedir, "wb") as handle:
            pickle.dump(filepaths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if index >= nsplits:
        raise Exception(f"Index should be less than nsplits ({nsplits})")
    if index != -1:
        each_split_size = len(filepaths) // nsplits
        print(f"Processing index: {index}")
        filepaths = filepaths[
            (index * each_split_size) : min(
                (index + 1) * each_split_size, len(filepaths)
            )
        ]
        print(f"Split size: {len(filepaths)}")

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = cpu_count()

    chunked_lists = chunk_into_n(filepaths, 5)
    new_process = []
    for i, list_ in enumerate(chunked_lists):
        new_arg = (list_, input_dir, output_dir, size, i + index)
        new_process.append(new_arg)
    # func(*new_process[0])
    with Pool(5) as pool:
        pool.starmap(func, new_process)

    # func(filepaths, input_dir, output_dir, size, 0)


if __name__ == "__main__":
    app()
