## API part
from fastapi import FastAPI
import uvicorn

from webui import *

app = FastAPI()


def load_config():
    with open("krita_config.yaml") as file:
        return yaml.safe_load(file)


class ImgTools:
    def __init__(self):
        self.current_img = None
        self.current_img_path = ""

    def load_img(self, path):
        if self.current_img_path != path:
            self.current_img = Image.open(path)
        return self.current_img

    def save_img(self, image, sample_path, filename):
        path = os.path.join(sample_path, filename)
        image.save(path)
        return os.path.abspath(path)


imgtools = ImgTools()


def fix_aspect_ratio(opt, orig_width, orig_height):
    def rnd(r, x):
        z = 64
        return z * round(r * x / z)

    base_size = opt['base_size']
    max_size = opt['max_size']
    ratio = orig_width / orig_height

    if orig_width > orig_height:
        width, height = rnd(ratio, base_size), base_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    else:
        width, height = base_size, rnd(1 / ratio, base_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    print(f"img size: {orig_width}x{orig_height} -> {width}x{height}")
    return width, height


def collect_prompt(opt):
    prompts = opt['prompts']
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


@app.get("/config")
async def read_item():
    opt = load_config()['plugin']
    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    filename = f"{int(time.time())}.png"
    path = os.path.join(sample_path, filename)
    src_path = os.path.abspath(path)
    print(f"src path: {src_path}")
    return {"new_img": src_path, **opt}


@app.get("/txt2img")
async def f_txt2img(orig_width: int, orig_height: int, prompt: str):
    print(f"txt2img")

    opt = load_config()['txt2img']
    width, height = fix_aspect_ratio(opt, orig_width, orig_height)
    output_images, seed, info, stats = txt2img(prompt=prompt or collect_prompt(opt),
                                               ddim_steps=opt['ddim_steps'],
                                               sampler_name=opt['sampler_name'],
                                               toggles=[1 if opt['normalize_prompt_weights'] else None,
                                                        4 if opt['use_gfpgan'] else None,
                                                        5 if opt['use_realesrgan'] else None],
                                               realesrgan_model_name=opt['realesrgan_model_name'],
                                               ddim_eta=opt['ddim_eta'],
                                               n_iter=opt['n_iter'],
                                               batch_size=opt['batch_size'],
                                               cfg_scale=opt['cfg_scale'],
                                               seed=opt['seed'],
                                               height=height,
                                               width=width,
                                               fp=None)

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    resized_images = [resize_image(0, image, orig_width, orig_height) for image in output_images]
    outputs = [imgtools.save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png") for i, image in
               enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}\n{stats}")
    return {"outputs": outputs}


@app.get("/img2img")
async def f_img2img(src_path: str, prompt: str):
    print(f"src path: {src_path}")

    opt = load_config()['img2img']
    image = imgtools.load_img(src_path)
    orig_width, orig_height = image.size
    width, height = fix_aspect_ratio(opt, orig_width, orig_height)
    output_images, seed, info, stats = img2img(prompt=prompt or collect_prompt(opt),
                                               image_editor_mode='Crop',
                                               init_info=image,
                                               mask_mode='Keep masked area',
                                               mask_blur_strength=0,
                                               ddim_steps=opt['ddim_steps'],
                                               sampler_name=opt['sampler_name'],
                                               toggles=[1 if opt['normalize_prompt_weights'] else None,
                                                        6 if opt['use_gfpgan'] else None,
                                                        7 if opt['use_realesrgan'] else None],
                                               realesrgan_model_name=opt['realesrgan_model_name'],
                                               n_iter=opt['n_iter'],
                                               batch_size=opt['batch_size'],
                                               cfg_scale=opt['cfg_scale'],
                                               denoising_strength=opt['denoising_strength'],
                                               seed=opt['seed'],
                                               height=height,
                                               width=width,
                                               resize_mode=opt['resize_mode'],
                                               fp=None)

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    resized_images = [resize_image(0, image, orig_width, orig_height) for image in output_images]
    outputs = [imgtools.save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png") for i, image in
               enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}\n{stats}")
    return {"outputs": outputs}


@app.get("/upscale")
async def f_upscale(src_path: str):
    print(f"upscale {src_path}")

    opt = load_config()["upscale"]
    image = imgtools.load_img(src_path)
    orig_width, orig_height = image.size
    width, height = fix_aspect_ratio(opt, orig_width, orig_height)
    resized_image = resize_image(0, image, width, height)

    output = run_RealESRGAN(resized_image, opt['realesrgan_model_name'])
    torch_gc()

    resized_output = resize_image(0, output, orig_width, orig_height)
    sample_path = opt['sample_path']
    output_path = imgtools.save_img(resized_output, sample_path, filename=f"{int(time.time())}_resized.png")
    return {"output": output_path}


if __name__ == '__main__':
    uvicorn.run("krita_server:app", host="127.0.0.1", port=8000, log_level="info")
