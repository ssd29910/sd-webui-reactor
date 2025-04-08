import os, glob, random
from collections import Counter
from PIL import Image
from math import isqrt, ceil
from typing import List
import logging
import hashlib
import torch
from safetensors.torch import save_file, safe_open
from insightface.app.common import Face
from tqdm import tqdm
import urllib.request

from modules.images import FilenameGenerator, get_next_sequence_number
from modules import shared, script_callbacks
from scripts.reactor_globals import DEVICE, BASE_PATH, FACE_MODELS_PATH, IS_SDNEXT

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")

MODELS_PATH = None

def set_Device(value):
    global DEVICE
    DEVICE = value
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(DEVICE)

def get_Device():
    global DEVICE
    return DEVICE

def set_SDNEXT():
    global IS_SDNEXT
    IS_SDNEXT = True

def get_SDNEXT():
    global IS_SDNEXT
    return IS_SDNEXT

def make_grid(image_list: List):
    # Compte les occurrences de chaque taille dans image_list
    size_counter = Counter(image.size for image in image_list)
    # Récupère la taille la plus fréquente
    common_size = size_counter.most_common(1)[0][0]
    # Filtre image_list pour ne garder que celles de taille commune
    image_list = [image for image in image_list if image.size == common_size]
    size = common_size
    if len(image_list) > 1:
        num_images = len(image_list)
        rows = isqrt(num_images)
        cols = ceil(num_images / rows)
        square_size = (cols * size[0], rows * size[1])
        square_image = Image.new("RGB", square_size)
        for i, image in enumerate(image_list):
            row = i // cols
            col = i % cols
            square_image.paste(image, (col * size[0], row * size[1]))
        return square_image
    return None

def get_image_path(image, path, basename, seed=None, prompt=None, extension='png', p=None, suffix=""):
    namegen = FilenameGenerator(p, seed, prompt, image)
    save_to_dirs = shared.opts.save_to_dirs
    if save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)
    os.makedirs(path, exist_ok=True)
    if seed is None:
        file_decoration = ""
    elif shared.opts.save_to_dirs:
        file_decoration = shared.opts.samples_filename_pattern or "[seed]"
    else:
        file_decoration = shared.opts.samples_filename_pattern or "[seed]-[prompt_spaces]"
    file_decoration = namegen.apply(file_decoration) + suffix
    add_number = shared.opts.save_images_add_number or file_decoration == ''
    if file_decoration != "" and add_number:
        file_decoration = f"-{file_decoration}"
    if add_number:
        basecount = get_next_sequence_number(path, basename)
        fullfn = None
        for i in range(500):
            fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
            fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
            if not os.path.exists(fullfn):
                break
    else:
        fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    pnginfo = {}
    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    # script_callbacks.before_image_saved_callback(params)
    fullfn = params.filename
    fullfn_without_extension, extension = os.path.splitext(params.filename)
    if hasattr(os, 'statvfs'):
        max_name_len = os.statvfs(path).f_namemax
        fullfn_without_extension = fullfn_without_extension[:max_name_len - max(4, len(extension))]
        params.filename = fullfn_without_extension + extension
        fullfn = params.filename
    return fullfn

def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)
    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def get_image_md5hash(image: Image.Image):
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()

def save_face_model(face: Face, filename: str) -> None:
    try:
        tensors = {
            "bbox": torch.tensor(face["bbox"]),
            "kps": torch.tensor(face["kps"]),
            "det_score": torch.tensor(face["det_score"]),
            "landmark_3d_68": torch.tensor(face["landmark_3d_68"]),
            "pose": torch.tensor(face["pose"]),
            "landmark_2d_106": torch.tensor(face["landmark_2d_106"]),
            "embedding": torch.tensor(face["embedding"]),
            "gender": torch.tensor(face["gender"]),
            "age": torch.tensor(face["age"]),
        }
        save_file(tensors, filename)
    except Exception as e:
        print(f"Error: {e}")

def get_models():
    global MODELS_PATH
    models_path_init = os.path.join(models_path, "insightface/*")
    models = glob.glob(models_path_init)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    models_names = []
    for model in models:
        model_path_split = os.path.split(model)
        if MODELS_PATH is None:
            MODELS_PATH = model_path_split[0]
        model_name = model_path_split[1]
        models_names.append(model_name)
    return models_names

def load_face_model(filename: str):
    face = {}
    model_path_full = os.path.join(FACE_MODELS_PATH, filename)
    with safe_open(model_path_full, framework="pt") as f:
        for k in f.keys():
            face[k] = f.get_tensor(k).numpy()
    return Face(face)

def get_facemodels():
    models_path_full = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path_full)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_model_names(get_models_func):
    models = get_models_func()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "None")
    return names

def get_images_from_folder(path: str):
    files_path = os.path.join(path, "*")
    files = glob.glob(files_path)
    images = []
    images_names = []
    for x in files:
        if x.lower().endswith(('jpg', 'png', 'jpeg', 'webp', 'bmp')):
            images.append(Image.open(x))
            images_names.append(os.path.basename(x))
    return images, images_names

def get_random_image_from_folder(path: str):
    images, names = get_images_from_folder(path)
    random_image_index = random.randint(0, len(images) - 1)
    return [images[random_image_index]], [names[random_image_index]]

def get_images_from_list(imgs: List):
    images = []
    images_names = []
    for x in imgs:
        images.append(Image.open(os.path.abspath(x.name)))
        images_names.append(os.path.basename(x.name))
    return images, images_names

def download(url, path, name):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc=f'[ReActor] Downloading {name} to {path}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

def check_nsfwdet_model(path: str):
    """
    Vérifie la présence du modèle NSFW. Si le modèle n'est pas présent,
    le téléchargement est bypassé.
    """
    if not os.path.exists(path):
        print("Modèle NSFW non trouvé. Téléchargement bypassé.")
    else:
        print("Modèle NSFW trouvé.")
