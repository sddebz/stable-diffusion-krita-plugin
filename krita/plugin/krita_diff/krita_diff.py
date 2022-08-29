import urllib.parse
import urllib.request
import json

from krita import *
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLineEdit

workaround_timeout = 100
base_url = "http://127.0.0.1:8000"


class Script:
    def __init__(self):
        self.opt = self.get_config()
        self.app = Krita.instance()
        self.doc = self.app.activeDocument()
        # self.node = self.doc.activeNode()
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

    def get_config(self):
        with urllib.request.urlopen(base_url + '/config') as req:
            res = req.read()
            print(res)
            return json.loads(res)

    def txt2img(self, prompt=""):
        params = {
            "prompt": prompt,
            "orig_width": self.width,
            "orig_height": self.height
        }
        query_string = urllib.parse.urlencode(params)
        with urllib.request.urlopen(base_url + '/txt2img?' + query_string) as req:
            res = req.read()
            print(res)
            return json.loads(res)['outputs']

    def img2img(self, path, prompt=""):
        params = {
            "prompt": prompt,
            "src_path": path
        }
        query_string = urllib.parse.urlencode(params)
        with urllib.request.urlopen(base_url + '/img2img?' + query_string) as req:
            res = req.read()
            print(res)
            return json.loads(res)['outputs']

    def upscale(self, path):
        params = {
            "src_path": path
        }
        query_string = urllib.parse.urlencode(params)
        with urllib.request.urlopen(base_url + '/upscale?' + query_string) as req:
            res = req.read()
            print(res)
            return json.loads(res)['output']

    def save_img(self, path):
        pixel_bytes = self.doc.pixelData(self.x, self.y, self.width, self.height)
        image_data = QImage(pixel_bytes, self.width, self.height, QImage.Format_RGBA8888).rgbSwapped()
        image_data.save(path, "PNG", self.opt['png_quality'])
        print(f"Saved image: {path}")

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

    def insert_img(self, path, visible=True):
        image = QImage()
        image.load(path, "PNG")
        ba = self.image_to_ba(image)

        layer = self.create_layer(path)
        if not visible:
            layer.setVisible(False)

        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"Inserted image: {path}")

    def create_mask_layer(self):
        if self.selection is None:
            return

        self.app.action('add_new_transparency_mask').trigger()
        print(f"created mask layer")
        self.doc.setSelection(self.selection)

    def apply_txt2img(self, prompt=""):
        outputs = self.txt2img(prompt)
        print(f"Getting images: {outputs}")
        for i, output in enumerate(outputs):
            self.insert_img(output, i + 1 == len(outputs))
        self.doc.refreshProjection()

    def apply_img2img(self, prompt=""):
        path = self.opt['new_img']
        self.save_img(path)
        outputs = self.img2img(path, prompt)
        print(f"Getting images: {outputs}")
        for i, output in enumerate(outputs):
            self.insert_img(output, i + 1 == len(outputs))
        self.doc.refreshProjection()

    def apply_upscale(self):
        path = self.opt['new_img']
        self.save_img(path)
        output = self.upscale(path)
        print(f"Getting images: {output}")
        self.insert_img(output)
        self.doc.refreshProjection()


class MyExtension(Extension):

    def __init__(self, parent):
        # This is initialising the parent, always important when subclassing.
        super().__init__(parent)

    def setup(self):
        pass

    def action_txt2img(self):
        script = Script()
        
        def on_enter(prompt):
            script.apply_txt2img(prompt)
            QTimer.singleShot(workaround_timeout, self.action_mask)
        prompt_dialog = PromptDialog("txt2img Prompt")
        prompt_dialog.start(on_enter)

    def action_img2img(self):
        script = Script()

        def on_enter(prompt):
            script.apply_img2img(prompt)
            QTimer.singleShot(workaround_timeout, self.action_mask)
        prompt_dialog = PromptDialog("img2img Prompt")
        prompt_dialog.start(on_enter)

    def action_upscale(self):
        script = Script()
        script.apply_upscale()
        QTimer.singleShot(workaround_timeout, self.action_mask)

    def action_mask(self):
        script = Script()
        script.create_mask_layer()

    def createActions(self, window):
        txt2img_action = window.createAction("txt2img", "Apply txt2img transform", "tools/scripts")
        txt2img_action.triggered.connect(self.action_txt2img)
        img2img_action = window.createAction("img2img", "Apply img2img transform", "tools/scripts")
        img2img_action.triggered.connect(self.action_img2img)
        upscale_x_action = window.createAction("upscale_x", "Apply upscale transform", "tools/scripts")
        upscale_x_action.triggered.connect(self.action_upscale)

class PromptDialog:
    def __init__(self, title):
        self.title = title

    def start(self, callback):
        textInput = QLineEdit()
        enterButton = QPushButton("Submit")

        layout = QVBoxLayout()
        layout.addWidget(textInput)
        layout.addWidget(enterButton)

        newDialog = QDialog()
        newDialog.setWindowTitle(self.title)
        newDialog.setLayout(layout)
        newDialog.setFixedWidth(400)
        def enter_text():
            prompt_text = textInput.text()
            newDialog.close()
            callback(prompt_text)
        textInput.returnPressed.connect(enter_text)
        enterButton.clicked.connect(enter_text)
        newDialog.exec_()

# And add the extension to Krita's list of extensions:
Krita.instance().addExtension(MyExtension(Krita.instance()))
