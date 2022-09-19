import math
import os
import urllib.parse
import urllib.request
import json
from urllib import request, parse

import requests
from krita import *

default_url = "http://127.0.0.1:8000"
# get an OS-sane tmp dir
default_tmp_dir = os.path.join(os.path.expanduser("~"), "tmp")

samplers = ["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms']
samplers_img2img = ["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms']
upscalers = ["None", "Lanczos"]
realesrgan_models = ['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B']

MODE_IMG2IMG = 0
MODE_INPAINT = 1
MODE_SD_UPSCALE = 3



class Script(QObject):
    def __init__(self):
        # Persistent settings (should reload between Krita sessions)
        self.config = QSettings(QSettings.IniFormat, QSettings.UserScope, "krita", "krita_diff_plugin")
        self.restore_defaults(if_empty=True)
        self.working = False

    def cfg(self, name, type):
        return self.config.value(name, type=type)

    def set_cfg(self, name, value, if_empty=False):
        if not if_empty or not self.config.contains(name):
            self.config.setValue(name, value)

    def restore_defaults(self, if_empty=False):
        self.set_cfg('base_url', default_url, if_empty)
        self.set_cfg('just_use_yaml', False, if_empty)
        self.set_cfg('create_mask_layer', True, if_empty)
        self.set_cfg('delete_temp_files', True, if_empty)
        self.set_cfg('workaround_timeout', 100, if_empty)
        self.set_cfg('png_quality', -1, if_empty)
        self.set_cfg('fix_aspect_ratio', True, if_empty)
        self.set_cfg('only_full_img_tiling', True, if_empty)
        self.set_cfg('tmp_dir', default_tmp_dir, if_empty)

        self.set_cfg('txt2img_prompt', "", if_empty)
        self.set_cfg('txt2img_sampler', samplers.index("k_euler_a"), if_empty)
        self.set_cfg('txt2img_steps', 20, if_empty)
        self.set_cfg('txt2img_cfg_scale', 7.5, if_empty)
        self.set_cfg('txt2img_batch_count', 1, if_empty)
        self.set_cfg('txt2img_batch_size', 1, if_empty)
        self.set_cfg('txt2img_base_size', 512, if_empty)
        self.set_cfg('txt2img_max_size', 704, if_empty)
        self.set_cfg('txt2img_seed', "", if_empty)
        self.set_cfg('txt2img_use_gfpgan', False, if_empty)
        self.set_cfg('txt2img_tiling', False, if_empty)

        self.set_cfg('img2img_prompt', "", if_empty)
        self.set_cfg('img2img_sampler', samplers_img2img.index("k_euler_a"), if_empty)
        self.set_cfg('img2img_steps', 50, if_empty)
        self.set_cfg('img2img_cfg_scale', 12.0, if_empty)
        self.set_cfg('img2img_denoising_strength', 0.40, if_empty)
        self.set_cfg('img2img_batch_count', 1, if_empty)
        self.set_cfg('img2img_batch_size', 1, if_empty)
        self.set_cfg('img2img_base_size', 512, if_empty)
        self.set_cfg('img2img_max_size', 704, if_empty)
        self.set_cfg('img2img_seed', "", if_empty)
        self.set_cfg('img2img_use_gfpgan', False, if_empty)
        self.set_cfg('img2img_tiling', False, if_empty)
        self.set_cfg('img2img_upscaler_name', 0, if_empty)

        self.set_cfg('upscale_upscaler_name', 0, if_empty)
        self.set_cfg('upscale_downscale_first', False, if_empty)

    def update_config(self):
        with urllib.request.urlopen(self.cfg('base_url', str) + '/config') as req:
            res = req.read()
            self.opt = json.loads(res)

        for upscaler in self.opt['upscalers']:
            if upscaler not in upscalers:
                upscalers.append(upscaler)

        self.app = Krita.instance()
        self.doc = self.app.activeDocument()
        self.node = self.doc.activeNode()
        self.selection = self.doc.selection()

        if self.selection is None:
            self.x = 0
            self.y = 0
            self.width = self.doc.width()
            self.height = self.doc.height()
        else:
            self.x = self.selection.x()
            self.y = self.selection.y()
            self.width = self.selection.width()
            self.height = self.selection.height()

    # Server API    @staticmethod
    def post(self, url, body):
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        body = json.dumps(body)
        body_encoded = body.encode('utf-8')
        req.add_header('Content-Length', str(len(body_encoded)))
        with urllib.request.urlopen(req, body_encoded) as res:
            return json.loads(res.read())

    def txt2img(self):
        tiling = self.cfg('txt2img_tiling', bool)
        if self.cfg("only_full_img_tiling", bool) and self.selection is not None:
            tiling = False

        params = {
            "orig_width": self.width,
            "orig_height": self.height,
            "prompt": self.fix_prompt(
                self.cfg('txt2img_prompt', str) if not self.cfg('txt2img_prompt', str).isspace() else None),
            "sampler_name": samplers[self.cfg('txt2img_sampler', int)],
            "steps": self.cfg('txt2img_steps', int),
            "cfg_scale": self.cfg('txt2img_cfg_scale', float),
            "batch_count": self.cfg('txt2img_batch_count', int),
            "batch_size": self.cfg('txt2img_batch_size', int),
            "base_size": self.cfg('txt2img_base_size', int),
            "max_size": self.cfg('txt2img_max_size', int),
            "seed": self.cfg('txt2img_seed', str) if not self.cfg('txt2img_seed', str).isspace() else '',
            "tiling": tiling,
            "use_gfpgan": self.cfg("txt2img_use_gfpgan", bool)
        } if not self.cfg('just_use_yaml', bool) else {
            "orig_width": self.width,
            "orig_height": self.height
        }
        return self.post(self.cfg('base_url', str) + '/txt2img', params)

    def img2img(self, path, mask_path, mode):
        tiling = self.cfg('txt2img_tiling', bool)
        if mode == MODE_SD_UPSCALE or (self.cfg("only_full_img_tiling", bool) and self.selection is not None):
            tiling = False

        params = {
            "mode": mode,
            "src_path": path,
            "mask_path": mask_path,
            "prompt": self.fix_prompt(
                self.cfg('img2img_prompt', str) if not self.cfg('img2img_prompt', str).isspace() else None),
            "sampler_name": samplers_img2img[self.cfg('img2img_sampler', int)],
            "steps": self.cfg('img2img_steps', int),
            "cfg_scale": self.cfg('img2img_cfg_scale', float),
            "denoising_strength": self.cfg('img2img_denoising_strength', float),
            "batch_count": self.cfg('img2img_batch_count', int),
            "batch_size": self.cfg('img2img_batch_size', int),
            "base_size": self.cfg('img2img_base_size', int),
            "max_size": self.cfg('img2img_max_size', int),
            "seed": self.cfg('img2img_seed', str) if not self.cfg('img2img_seed', str).isspace() else '',
            "tiling": tiling,
            "use_gfpgan": self.cfg("img2img_use_gfpgan", bool),
            "upscaler_name": upscalers[self.cfg('img2img_upscaler_name', int)]
        } if not self.cfg('just_use_yaml', bool) else {
            "src_path": path,
            "mask_path": mask_path
        }
        return self.post(self.cfg('base_url', str) + '/img2img', params)

    def simple_upscale(self, path):
        params = {
            "src_path": path,
            "upscaler_name": upscalers[self.cfg("upscale_upscaler_name", int)],
            "downscale_first": self.cfg("upscale_downscale_first", bool)
        } if not self.cfg('just_use_yaml', bool) else {
            "src_path": path
        }
        return self.post(self.cfg('base_url', str) + '/upscale', params)

    def fix_prompt(self, prompt):
        return ', '.join(filter(bool, [x.strip() for x in prompt.splitlines()]))

    def find_final_aspect_ratio(self):
        base_size = self.cfg('img2img_base_size', int)
        max_size = self.cfg('img2img_max_size', int)

        def rnd(r, x):
            z = 64
            return z * round(r * x / z)

        ratio = self.width / self.height

        if self.width > self.height:
            width, height = rnd(ratio, base_size), base_size
            if width > max_size:
                width, height = max_size, rnd(1 / ratio, max_size)
        else:
            width, height = base_size, rnd(1 / ratio, base_size)
            if height > max_size:
                width, height = rnd(ratio, max_size), max_size

        return width / height

    def try_fix_aspect_ratio(self):
        # SD will need to resize image to match required size of 512x(512 + 64*j). That may fuck aspect ratio.
        # That's why we will try to make selection slightly bigger to unfuck aspect ratio a little

        if self.selection is not None and self.cfg("fix_aspect_ratio", bool):
            ratio = self.width / self.height
            final_ratio = self.find_final_aspect_ratio()

            delta = abs(final_ratio - ratio)
            delta_rev = abs(1 / final_ratio - 1 / ratio)
            x_limit = math.ceil(delta * self.width)
            y_limit = math.ceil(delta_rev * self.height)

            best_delta = delta
            best_x1, best_y1 = self.x, self.y
            best_x2, best_y2 = self.x + self.width, self.y + self.height
            for x in range(x_limit):
                x1 = max(0, self.x - (x // 2))
                x2 = min(self.doc.width(), x1 + self.width + x)
                for y in range(y_limit):
                    y1 = max(0, self.y - (y // 2))
                    y2 = min(self.doc.height(), y1 + self.height + y)

                    curr_ratio = (x2 - x1) / (y2 - y1)
                    curr_delta = abs(curr_ratio - final_ratio)
                    if curr_delta < best_delta:
                        best_delta = curr_delta
                        best_x1, best_y1 = x1, y1
                        best_x2, best_y2 = x2, y2

            self.x = best_x1
            self.y = best_y1
            self.width = best_x2 - best_x1
            self.height = best_y2 - best_y1

    def save_img(self, path, is_mask=False):
        if is_mask:
            pixel_bytes = self.node.pixelData(self.x, self.y, self.width, self.height)
        else:
            pixel_bytes = self.doc.pixelData(self.x, self.y, self.width, self.height)

        image_data = QImage(pixel_bytes, self.width, self.height, QImage.Format_RGBA8888).rgbSwapped()
        image_data.save(path, "PNG", self.cfg('png_quality', int))
        print(f"Saved {'mask' if is_mask else 'image'}: {path}")

    # Krita tools
    def create_layer(self, name):
        root = self.doc.rootNode()
        layer = self.doc.createNode(name, "paintLayer")
        root.addChildNode(layer, None)
        print(f"created layer: {layer}")
        return layer

    def image_to_ba(self, image):
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        return QByteArray(ptr.asstring())

    def insert_img(self, layer_name, qimage, visible=True):
        layer = self.create_layer(layer_name)
        ba = self.image_to_ba(qimage)

        if not visible:
            layer.setVisible(False)
        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"inserted image to layer: {layer}")

    def insert_img_from_path(self, layer_name, path, visible=True):
        image = QImage()
        image.load(path, "PNG")
        ba = self.image_to_ba(image)

        layer = self.create_layer(layer_name)
        if not visible:
            layer.setVisible(False)

        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"Inserted image: {path}")

    def apply_txt2img(self):
        response = self.txt2img()
        outputs = response['outputs']

        print(f'Fetching remote images from server. Images: {outputs}')
        for i, output in enumerate(outputs):
            url = self.cfg('base_url', str) + '/result'

            data = json.dumps({'file_name': output}).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)

            image = QImage()
            image.loadFromData(response.read(), "PNG")

            layer = self.create_layer(f"txt2img {i + 1}: {output}")
            layer.setPixelData(self.image_to_ba(image), self.x, self.y, self.width, self.height)

        self.doc.refreshProjection()

    def apply_img2img(self, mode):
        path        = self.cfg('tmp_dir', str) + '/img2img.png'
        mask_path   = self.cfg('tmp_dir', str) + '/img2img_mask.png'
        path_filename   = os.path.basename(path)
        mask_filename   = os.path.basename(mask_path)

        if mode == MODE_INPAINT:
            print(f'Saving mask locally: {mask_path}')
            self.save_img(mask_path, is_mask=True)
            self.node.setVisible(False)
            self.doc.refreshProjection()
        self.save_img(path)

        response = requests.request("POST", self.cfg('base_url', str) + '/saveimg', files=[('file',(path_filename,open(path,'rb'),'image/png'))])
        if response.status_code != 200:
            print(f"Error while uploading image to server: {response.status_code}")
            return

        if mode == MODE_INPAINT:
            response = requests.request("POST", self.cfg('base_url', str) + '/saveimg', files=[('file',(mask_filename,open(mask_path,'rb'),'image/png'))])
            if response.status_code != 200:
                print(f"Error while uploading image to server: {response.status_code}")
                return

        response = self.img2img(path_filename, mask_filename, mode)
        outputs = response['outputs']

        print(f"Getting images: {outputs}")
        layer_name_prefix = "inpaint" if mode == MODE_INPAINT else "sd upscale" if mode == MODE_SD_UPSCALE else "img2img"
        for i, output in enumerate(outputs):
            req = urllib.request.Request(url = self.cfg('base_url', str) + '/result',
                                         data=json.dumps({'file_name': os.path.basename(output)}).encode('utf-8'),
                                         headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)
            qimage = QImage()
            qimage.loadFromData(response.read(), "PNG")

            self.insert_img(f"{layer_name_prefix} {i + 1}: {os.path.basename(output)}", qimage, i + 1 == len(outputs))

        if mode == MODE_INPAINT:
            self.clear_temp_images([path, mask_path])
        else:
            self.clear_temp_images([path])

        self.doc.refreshProjection()

    def apply_simple_upscale(self):
        path = self.cfg('tmp_dir', str) + '/simple_upscale.png'
        path_filename = os.path.basename(path)

        self.save_img(path)
        response = requests.request("POST", self.cfg('base_url', str) + '/saveimg',
                                    files=[('file',(path_filename,open(path,'rb'),'image/png'))])
        if response.status_code != 200:
            print(f"Error while uploading image to server: {response.status_code}")
            return

        response = self.simple_upscale(path_filename)
        output = response['output']
        print(f"Getting image: {output}")
        req = urllib.request.Request(url = self.cfg('base_url', str) + '/result',
                                     data=json.dumps({'file_name': os.path.basename(output)}).encode('utf-8'),
                                     headers={'content-type': 'application/json'})
        response = urllib.request.urlopen(req)
        qimage = QImage()
        qimage.loadFromData(response.read(), "PNG")

        self.insert_img(f"upscale: {path_filename}", qimage)
        self.clear_temp_images([path])
        self.doc.refreshProjection()

    def create_mask_layer_internal(self):
        try:
            if self.selection is not None:
                self.app.action('add_new_transparency_mask').trigger()
                print(f"created mask layer")
                self.doc.setSelection(self.selection)
        finally:
            self.working = False

    def create_mask_layer_workaround(self):
        if self.cfg('create_mask_layer', bool):
            self.working = True
            QTimer.singleShot(self.cfg('workaround_timeout', int), lambda: self.create_mask_layer_internal())

    def clear_temp_images(self, files):
        if self.cfg('delete_temp_files', bool):
            for file in files:
                os.remove(file)

    # Actions
    def action_txt2img(self):
        if self.working:
            pass
        self.update_config()
        self.try_fix_aspect_ratio()
        self.apply_txt2img()
        self.create_mask_layer_workaround()

    def action_img2img(self):
        if self.working:
            pass
        self.update_config()
        self.try_fix_aspect_ratio()
        self.apply_img2img(mode=MODE_IMG2IMG)
        self.create_mask_layer_workaround()

    def action_sd_upscale(self):
        if self.working:
            pass
        self.update_config()
        self.apply_img2img(mode=MODE_SD_UPSCALE)
        self.create_mask_layer_workaround()

    def action_inpaint(self):
        if self.working:
            pass
        self.update_config()
        self.try_fix_aspect_ratio()
        self.apply_img2img(mode=MODE_INPAINT)

    def action_simple_upscale(self):
        if self.working:
            pass
        self.update_config()
        self.apply_simple_upscale()


script = Script()
