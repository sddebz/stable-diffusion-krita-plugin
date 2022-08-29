### [MAIN REPO](https://github.com/hlky/stable-diffusion)

## This repo is just for a simple Krita pligin, based on webui code

## Example

https://user-images.githubusercontent.com/112324253/187082905-1b41816e-1705-433b-8ba2-06b78f79c920.mp4


## Usage

1. Start krita.cmd
2. Open Krita
3. Select area you want to use. If you have no selection, whole image will be used
4. Press Ctrl + Alt + Q (txt2img) or Ctrl + Alt + W (img2img)
5. Type optional prompt. If left empty it will use the prompt defined in krita_config.yaml
6. Look at progress bar in your server console window
7. Plugin will insert generated images as new layers into Krita. Only last one will be set visible. Selection will be
   converted into transparency mask on the last layer.

Image in Krita has to be RGBA 8bit, you need to select some layer to proceed. Plugin uses merged image as a source for img2img.

## Installation

### Server part installation

Just use guide from parent repo. You should make sure webui.cmd works.

This repo adds fastapi and uvicorn libraries as a requirement. Please update your env to include it.
```shell
conda env update -n ldo --file environment.yaml --prune
```

If this doesn't work you may need to add it manually with pip.
```shell
conda env activate ldo
pip install fastapi uvicorn
```

There are no changes in base files from parent repo except environment.yaml. If you want to update parent repo version, it should work unless api from `webui.py` changes.

You'll also need Krita 5.1. It may work on other versions, but Krita plugin api is fucking bugged.

### Plugin installation

1. Open Krita and go into Settings - Manage Resources... - Open Resource Folder
2. Go into folder `pykrita` (create it if it doesn't exist)
3. Copy contents of folder `krita/plugin` into `pykrita` folder. You should have `krita_diff` folder
   and `krita_diff.desktop` file in pykrita folder.
4. Restart Krita
5. Go into Settings - Configure Krita... - Python Plugin Manager
6. Activate plugin "Krita Stable Diffusion Plugin"
7. Restart Krita

Now you must have 3 menu entries in Tools - Scripts:

- Apply txt2img transform - Ctrl + Alt + Q
- Apply img2img transform - Ctrl + Alt + W
- Apply upscale transform - Ctrl + Alt + E

You may alter shortcuts in Krita config as usual.

### Server part

Before using plugin you have to start webui-like server using command `krita.cmd` OR just run

```shell
conda env activate ldo
python scripts/krita_server.py
```

If you ever need your GPU for Genshin, just stop server. No need to exit Krita. If server crashes, restart it. Krita
should be fine.

## Configuration

Script uses mostly the same configuration options as webui. They have to be saved in file `krita_config.yaml`
Edit this file to change prompt or other options

### Prompt

There are 3 formats for prompt in yaml:

- Simple string

```
img2img:
    prompts: "your mom"
```

- list of separate strings - they will be concatenated

```
img2img:
    prompts: 
        - "your mom"
        - "my mom"
```

- dictionary with multiple prompts - they will be used as weighted prompts

```
img2img:
    prompts: 
        "your mom": 1 
        "my mom": 2
```


### Image size

You may want to update option max_size in config, if you have more than 8Gb of VRAM. Maybe even update base_size. I have 3070 with 8Gb and repo settings work for me.

The logic is following. Image from Krita is resized so min(height, width) = base_size. If resulting max(height, width) > max_size, image is downscaled so max(height, width) = max_size.

Size step is 64 pixels, so it's better to select areas aspect ratios: 512x512, 576x512, 640x512, 704x512,... you get the idea. If your selection is not right it may fuck up aspect ratio, however I personally think it isn't very noticeable.
