import os
import json
from math import ceil
import multiprocessing
from multiprocessing import Pool, cpu_count
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
from aligner import norm_crop

# from lib.landmarks_pytorch import LandmarksEstimation
import face_alignment
import typer
from pathlib import Path

IMAGE_EXT = (".jpg", ".jpeg", ".png")


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def adjust_coordinate(coord, new_shape, old_shape):
    ratio_x = float(old_shape[0] / new_shape[0])
    ratio_y = float(old_shape[1] / new_shape[1])

    coord = coord / np.array([ratio_x, ratio_y])
    return coord


def chunk_into_n(lst, n):
    size = ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def read_image_opencv(image_path):
    # Read image in BGR order
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("uint8")


def func(filepaths, input_dir: Path, output_dir: Path, size: int, idx=0):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False
    )
    for img_file_ in tqdm(filepaths, desc=f"Process {idx}"):
        img_file = img_file_.as_posix()
        img_name = img_file_.stem
        if img_name.split("_")[-1] in ["bottom", "top"]:
            continue
        outpath = output_dir / img_file_.relative_to(input_dir)
        outpath: Path
        outpath = outpath.with_suffix(".json")
        if outpath.exists():
            continue
        # Open input image
        try:
            img = read_image_opencv(img_file)
        except Exception:
            continue

        # Landmark estimation
        preds = fa.get_landmarks_from_image(img)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        with open(outpath, "w") as f:
            json.dump(preds, f, indent=2, cls=NumpyArrayEncoder)


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
    with Pool(1) as pool:
        pool.starmap(func, new_process)

    # func(filepaths, input_dir, output_dir, size, 0)


def move_func(paths: Path):
    for path in paths:
        if path.exists():
            os.rename(path, path.with_suffix(".json"))


def lmks68_to_5(lmk68):
    if isinstance(lmk68, list):
        lmk68 = np.array(lmk68)
    lmk68 = lmk68.reshape(68, 2)
    left_eye = (lmk68[36] + lmk68[39]) / 2
    right_eye = (lmk68[42] + lmk68[45]) / 2
    nose = lmk68[33]
    left_mouth = lmk68[48]
    right_mouth = lmk68[54]
    lmk5 = [left_eye, right_eye, nose, left_mouth, right_mouth]
    return np.array(lmk5).reshape(5, 2)


def align_func(imgpaths, img_dir, json_dir, out_dir):
    for imgpath in tqdm(imgpaths):
        relative_path = imgpath.relative_to(img_dir)
        out_path = out_dir / relative_path.with_suffix(".png")
        if out_path.exists():
            continue

        json_path = json_dir / relative_path.with_suffix(".json")
        if not json_path.exists():
            print(f"{json_path} is not exists")
            continue
        try:
            with open(json_path, "r") as f:
                preds = json.load(f)
        except Exception as e:
            print(f"{json_path} is None")
            continue
        img = cv2.imread(imgpath.as_posix())
        # for pred in preds:
        #     lmk5 = lmks68_to_5(pred)
        max_idx = 0
        if preds is None:
            print(f"{json_path} is None")
            continue
        if len(preds) > 1:
            max_area = 0
            for i, pred in enumerate(preds):
                lmk5 = lmks68_to_5(pred)
                height = lmk5[-1, 1] - lmk5[0, 1]
                width = lmk5[1, 0] - lmk5[0, 0]
                area = height * width
                if area > max_area:
                    max_idx = i
                max_area = max(area, max_area)

        pred = preds[max_idx]
        lmk5 = lmks68_to_5(pred)
        aligned = norm_crop(img, lmk5)
        if sum(np.array(aligned.shape) > 0) != 3:
            continue
        out_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(out_path.as_posix(), aligned)


@app.command()
def align_insightface(
    img_dir: Path = typer.Argument(..., help="img dir"),
    json_dir: Path = typer.Argument(..., help="json dir"),
    out_dir: Path = typer.Argument(..., help="json dir"),
    workers: int = typer.Option(1, help="workers"),
):

    filepaths = list(img_dir.rglob("*.[jp][pn]g"))
    chunked_lists = chunk_into_n(filepaths, 16)
    new_process = []
    for i, list_ in enumerate(chunked_lists):
        new_arg = (list_, img_dir, json_dir, out_dir)
        new_process.append(new_arg)
    with Pool(workers) as pool:
        pool.starmap(align_func, new_process)


@app.command()
def png_to_json(
    input_path: Path = typer.Argument(
        ..., help="path where file need to change into json"
    ),
    workers: int = typer.Option(1, help="workers"),
):

    filepaths = list(input_path.rglob("*.[jp][pn]g"))
    chunked_lists = chunk_into_n(filepaths, 5)
    new_process = []
    for i, list_ in enumerate(chunked_lists):
        new_arg = (list_,)
        new_process.append(new_arg)
    with Pool(workers) as pool:
        pool.starmap(move_func, new_process)


if __name__ == "__main__":
    app()
