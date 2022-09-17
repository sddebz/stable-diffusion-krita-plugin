import contextlib
import threading
import math
import shutil
import time
import yaml
from typing import Optional
from fastapi.responses import FileResponse
from fastapi import UploadFile

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

from webui import *

from PIL import Image

app = FastAPI()


def load_config():
    with open("krita_config.yaml") as file:
        return yaml.safe_load(file)


def save_img(image, sample_path, filename):
    path = os.path.join(sample_path, filename)
    image.save(path)
    return os.path.basename(path)


def fix_aspect_ratio(base_size, max_size, orig_width, orig_height):
    def rnd(r, x):
        z = 64
        return z * round(r * x / z)

    ratio = orig_width / orig_height

    if orig_width > orig_height:
        width, height = rnd(ratio, base_size), base_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    else:
        width, height = base_size, rnd(1 / ratio, base_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    new_ratio = width / height

    print(f"img size: {orig_width}x{orig_height} -> {width}x{height}, "
          f"aspect ratio: {ratio:.2f} -> {new_ratio:.2f}, {100 * (new_ratio - ratio) / ratio :.2f}% change")
    return width, height


def collect_prompt(opts, key):
    prompts = opts[key]
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, list):
        return ", ".join(prompts)
    if isinstance(prompts, dict):
        prompt = ""
        for item, weight in prompts.items():
            if not prompt == "":
                prompt += " "
            if weight is None:
                prompt += f"{item}"
            else:
                prompt += f"{item}:{weight}"
        return prompt
    raise Exception("wtf man, fix your prompts")

class ImageRequest(BaseModel):
    file_name: str

class Txt2ImgRequest(BaseModel):
    orig_width: int
    orig_height: int

    prompt: Optional[str]
    negative_prompt: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]
    tiling: Optional[bool]

    use_gfpgan: Optional[bool]
    face_restorer: Optional[str]
    codeformer_weight: Optional[float]


class Img2ImgRequest(BaseModel):
    mode: Optional[int]

    src_path: str
    mask_path: Optional[str]

    prompt: Optional[str]
    negative_prompt: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]
    denoising_strength: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]
    tiling: Optional[bool]

    use_gfpgan: Optional[bool]
    face_restorer: Optional[str]
    codeformer_weight: Optional[float]

    upscale_overlap: Optional[int]
    upscaler_name: Optional[str]

    inpainting_fill: Optional[int]
    inpaint_full_res: Optional[bool]
    mask_blur: Optional[int]
    invert_mask: Optional[bool]


class UpscaleRequest(BaseModel):
    src_path: str
    upscaler_name: Optional[str]
    downscale_first: Optional[bool]


def get_sampler_index(sampler_name: str):
    for index, sampler in enumerate(modules.sd_samplers.samplers):
        name, constructor, aliases = sampler
        if sampler_name == name or sampler_name in aliases:
            return index
    return 0


def get_upscaler_index(upscaler_name: str):
    for index, upscaler in enumerate(shared.sd_upscalers):
        if upscaler.name == upscaler_name:
            return index
    return 0


def set_face_restorer(face_restorer: str, codeformer_weight: float):
    shared.opts.face_restoration_model = face_restorer
    shared.opts.code_former_weight = codeformer_weight

@app.get("/config")
async def read_item():
    opt = load_config()['plugin']
    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    filename = f"{int(time.time())}"
    path = os.path.join(sample_path, filename)
    src_path = os.path.abspath(path)
    return {"new_img": src_path + ".png",
            "new_img_mask": src_path + "_mask.png",
            "upscalers": [upscaler.name for upscaler in shared.sd_upscalers],
            **opt}

@app.post("/result")
async def get_result(req: ImageRequest):
    print(f'get_result: {req.file_name}')
    opt = load_config()['txt2img']
    print(f'sample_path: {opt["sample_path"]}')
    path = os.path.join(opt['sample_path'], req.file_name)
    print(f'loading {path}')
    return FileResponse(path)

@app.post("/txt2img")
async def f_txt2img(req: Txt2ImgRequest):
    print(f"txt2img: {req}")

    opt = load_config()['txt2img']
    sample_path = opt['sample_path']
    set_face_restorer(req.face_restorer or opt['face_restorer'],
                      req.codeformer_weight or opt['codeformer_weight'])

    sampler_index = get_sampler_index(req.sampler_name or opt['sampler_name'])

    seed = opt['seed']
    if req.seed is not None and not req.seed == '':
        seed = int(req.seed)

    width, height = fix_aspect_ratio(req.base_size or opt['base_size'], req.max_size or opt['max_size'],
                                     req.orig_width, req.orig_height)

    output_images, info, html = modules.txt2img.txt2img(
        req.prompt or collect_prompt(opt, "prompts"),
        req.negative_prompt or collect_prompt(opt, "negative_prompt"),
        "None",
        "None",
        req.steps or opt['steps'],
        sampler_index,
        req.use_gfpgan or opt['use_gfpgan'],
        req.tiling or opt['tiling'],
        req.batch_count or opt['n_iter'],
        req.batch_size or opt['batch_size'],
        req.cfg_scale or opt['cfg_scale'],
        seed,
        None,
        0,
        0,
        0,
        height,
        width,
        0
    )

    os.makedirs(sample_path, exist_ok=True)
    resized_images = [modules.images.resize_image(0, image, req.orig_width, req.orig_height) for image in output_images]
    outputs = [save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
               for i, image in enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}

@app.post("/saveimg")
async def f_saveimg(file: UploadFile):
    print(f'saveimg: {file.filename}')
    opt = load_config()['plugin']
    path = os.path.join(opt['sample_path'], file.filename)
    print(f'saving {path}')
    with open(path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return {"path": path}

@app.post("/img2img")
async def f_img2img(req: Img2ImgRequest):
    print(f"img2img: {req}")

    opt = load_config()['img2img']
    opt_plugin = load_config()['plugin']
    set_face_restorer(req.face_restorer or opt['face_restorer'],
                      req.codeformer_weight or opt['codeformer_weight'])

    sampler_index = get_sampler_index(req.sampler_name or opt['sampler_name'])

    seed = opt['seed']
    if req.seed is not None and not req.seed == '':
        seed = int(req.seed)

    mode = req.mode or opt['mode']

    path = os.path.join(opt_plugin['sample_path'], req.src_path)
    image = Image.open(path)
    orig_width, orig_height = image.size

    if mode == 1:
        mask_path = os.path.join(opt_plugin['sample_path'], req.mask_path)
        mask = Image.open(mask_path).convert('L')
    else:
        mask = None

    # because API in webui changed
    if mode == 3:
        mode = 2

    upscaler_index = get_upscaler_index(req.upscaler_name or opt['upscaler_name'])

    base_size = req.base_size or opt['base_size']
    if mode == 2:
        width, height = base_size, base_size
        if upscaler_index > 0:
            image = image.convert("RGB")
    else:
        width, height = fix_aspect_ratio(base_size, req.max_size or opt['max_size'],
                                         orig_width, orig_height)

    output_images, info, html = modules.img2img.img2img(
        req.prompt or collect_prompt(opt, 'prompts'),
        req.negative_prompt or collect_prompt(opt, 'negative_prompt'),
        "None",
        "None",
        image,
        {"image": image, "mask": mask},
        mask,
        1,
        req.steps or opt['steps'],
        sampler_index,
        req.mask_blur or opt['mask_blur'],
        req.inpainting_fill or opt['inpainting_fill'],
        req.use_gfpgan or opt['use_gfpgan'],
        req.tiling or opt['tiling'],
        mode,
        req.batch_count or opt['n_iter'],
        req.batch_size or opt['batch_size'],
        req.cfg_scale or opt['cfg_scale'],
        req.denoising_strength or opt['denoising_strength'],
        seed,
        None,
        0,
        0,
        0,
        height,
        width,
        opt['resize_mode'],
        upscaler_index,
        req.upscale_overlap or opt['upscale_overlap'],
        req.inpaint_full_res or opt['inpaint_full_res'],
        False,  # req.invert_mask or opt['invert_mask'],
        0
    )

    resized_images = [modules.images.resize_image(0, image, orig_width, orig_height) for image in output_images]

    if mode == 1:
        def remove_not_masked(img):
            masked_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            masked_img.paste(img, (0, 0), mask=mask)
            return masked_img

        resized_images = [remove_not_masked(x) for x in resized_images]

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    outputs = [save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
               for i, image in enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/upscale")
async def f_upscale(req: UpscaleRequest):
    print(f"upscale: {req}")

    opt = load_config()['upscale']
    opt_plugin = load_config()['plugin']
    path = os.path.join(opt_plugin['sample_path'], req.src_path)
    image = Image.open(path).convert('RGB')
    orig_width, orig_height = image.size

    upscaler_index = get_upscaler_index(req.upscaler_name or opt['upscaler_name'])
    upscaler = shared.sd_upscalers[upscaler_index]

    if upscaler.name == 'None':
        print(f"No upscaler selected, will do nothing")
        return

    if req.downscale_first or opt['downscale_first']:
        image = modules.images.resize_image(0, image, orig_width // 2, orig_height // 2)

    upscaled_image = upscaler.upscale(image, 2 * orig_width, 2 * orig_height)
    resized_image = modules.images.resize_image(0, upscaled_image, orig_width, orig_height)

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    output = save_img(resized_image, sample_path, filename=f"{int(time.time())}.png")
    print(f"finished: {output}")
    return {"output": output}


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def start():
    config = uvicorn.Config("krita_server:app", host="127.0.0.1", port=8000, log_level="info")
    server = Server(config=config)

    with server.run_in_thread():
        webui()


if __name__ == "__main__":
    start()
