"""Interfaz local para arrastrar una imagen y clasificarla con un checkpoint."""
import argparse
from functools import partial
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch

from data import ROOT, image_tensor
from model import load_checkpoint, select_device

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
HTML = """<!doctype html>
<html lang="es">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gatos y perros · CNN</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#172033;font:16px system-ui,sans-serif}
main{max-width:900px;margin:60px auto;padding:24px}h1{font-size:32px;margin-bottom:8px}
.subtitle{color:#586378;line-height:1.6}.card{background:white;padding:28px;border-radius:20px;box-shadow:0 12px 40px #1720330b}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:28px}
#drop{display:flex;align-items:center;justify-content:center;flex-direction:column;min-height:270px;border:2px dashed #b8c4d8;border-radius:14px;padding:20px;text-align:center;cursor:pointer;background:#fafbff}
#drop.over{border-color:#365be8;background:#eef2ff}input{max-width:100%;margin-top:16px}#preview{display:none;max-width:100%;max-height:230px;object-fit:contain;border-radius:8px;margin-bottom:12px}
button{width:100%;background:#365be8;color:white;padding:14px;border:0;border-radius:10px;font:inherit;font-weight:600;cursor:pointer;margin-top:16px}button:disabled{opacity:.5;cursor:default}
#status{min-height:48px;color:#586378;line-height:1.5}#result{font-size:26px;margin:14px 0 24px}.row{display:flex;justify-content:space-between;margin-top:18px}progress{width:100%;height:16px;accent-color:#365be8}.note{font-size:13px;color:#667085;line-height:1.5}.warning{color:#855700;background:#fff3cf;padding:12px;border-radius:10px}
@media(max-width:640px){main{margin:12px auto;padding:16px}.layout{grid-template-columns:1fr}.card{padding:20px}}
</style>
<main><h1>¿Gato o perro?</h1><p class="subtitle">Prueba tu modelo con una imagen nueva.</p>
<div class="card"><p class="note">Modelo: __MODEL_NAME__</p>__SMOKE_WARNING__
<div class="layout"><section>
<label id="drop" for="image"><img id="preview" alt="Imagen seleccionada"><span>Arrastra una imagen aquí<br>o selecciónala desde tu equipo</span><input id="image" type="file" accept="image/*"></label>
<button id="predict" disabled>Clasificar imagen</button><p class="note">Máximo 10 MB por imagen.</p>
</section><section aria-live="polite"><p id="status" role="status">Selecciona una imagen para empezar.</p><h2 id="result">—</h2>
<div id="scores" hidden><div class="row"><span>Gato</span><span id="cat-value"></span></div><progress id="cat" max="1" value="0" aria-label="Probabilidad de gato"></progress>
<div class="row"><span>Perro</span><span id="dog-value"></span></div><progress id="dog" max="1" value="0" aria-label="Probabilidad de perro"></progress></div>
<p class="note">Las probabilidades del modelo no son una medida calibrada de certeza.</p></section></div></div></main>
<script>
const input=document.querySelector('#image'), drop=document.querySelector('#drop'), button=document.querySelector('#predict'), status=document.querySelector('#status'), preview=document.querySelector('#preview');
let selected=null, previewUrl=null;
function select(file){
  selected=null;button.disabled=true;document.querySelector('#scores').hidden=true;document.querySelector('#result').textContent='—';
  if(previewUrl){URL.revokeObjectURL(previewUrl);previewUrl=null;}preview.style.display='none';preview.removeAttribute('src');
  if(!file){status.textContent='Selecciona una imagen para empezar.';return;}
  if(file.size>10*1024*1024){status.textContent='La imagen supera el límite de 10 MB.';return;}
  selected=file;previewUrl=URL.createObjectURL(file);preview.src=previewUrl;preview.style.display='block';button.disabled=false;status.textContent=file.name;
}
input.addEventListener('change',()=>select(input.files[0]));
for(const name of ['dragenter','dragover'])drop.addEventListener(name,event=>{event.preventDefault();drop.classList.add('over');});
for(const name of ['dragleave','drop'])drop.addEventListener(name,event=>{event.preventDefault();drop.classList.remove('over');});
drop.addEventListener('drop',event=>select(event.dataTransfer.files[0]));
button.addEventListener('click',async()=>{
  if(!selected)return;button.disabled=true;input.disabled=true;drop.style.pointerEvents='none';status.textContent='Analizando imagen…';
  try{
    const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:selected});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'No se pudo clasificar la imagen.');
    document.querySelector('#result').textContent=data.prediction==='cat'?'Gato':'Perro';
    for(const label of ['cat','dog']){document.querySelector('#'+label).value=data.probabilities[label];document.querySelector('#'+label+'-value').textContent=(data.probabilities[label]*100).toFixed(2)+' %';}
    document.querySelector('#scores').hidden=false;status.textContent='Predicción completada.';
  }catch(error){document.querySelector('#result').textContent='—';document.querySelector('#scores').hidden=true;status.textContent=error.message;}
  finally{button.disabled=false;input.disabled=false;drop.style.pointerEvents='';}
});
</script></html>"""


def predict_image(payload, *, model, device, classes):
    """Usa el mismo preprocesamiento que entrenamiento y predicción por consola."""
    with Image.open(BytesIO(payload)) as image:
        tensor = image_tensor(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    return {"prediction": classes[max(range(len(classes)), key=probabilities.__getitem__)],
            "probabilities": dict(zip(classes, probabilities))}


def make_handler(predict, page):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            else:
                self.respond(404, {"error": "Ruta no encontrada."})

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "Ruta no encontrada."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.respond(400, {"error": "Tamaño de imagen inválido."})
                return
            if not 0 < length <= MAX_UPLOAD_BYTES:
                self.respond(413 if length > MAX_UPLOAD_BYTES else 400,
                             {"error": "Selecciona una imagen de hasta 10 MB."})
                return
            try:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    self.respond(400, {"error": "La imagen no se recibió completamente."})
                    return
                result = predict(payload)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                self.respond(400, {"error": "No se pudo leer la imagen. Usa un archivo JPG, PNG o WebP válido."})
                return
            except Exception as error:
                self.log_error("Error durante la predicción: %s", error)
                self.respond(500, {"error": "No se pudo ejecutar el modelo. Revisa la terminal."})
                return
            self.respond(200, result)

    return Handler


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("El puerto debe estar entre 1 y 65535.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/best_model.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=port_number, default=7860)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"No existe el modelo: {args.checkpoint}. Indica --checkpoint /ruta/best_model.pt")
    device = select_device(args.device)
    try:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(f"No se pudo cargar el checkpoint de CatDogCNN: {error}")
    classes = sorted(checkpoint["class_to_idx"], key=checkpoint["class_to_idx"].get)
    warning = ('<p class="warning">Prueba técnica: este modelo no tiene un entrenamiento completo.</p>'
               if checkpoint.get("config", {}).get("smoke_test", False) else "")
    page = HTML.replace("__MODEL_NAME__", escape(args.checkpoint.name)).replace("__SMOKE_WARNING__", warning)
    predict = partial(predict_image, model=model, device=device, classes=classes)
    try:
        server = HTTPServer(("127.0.0.1", args.port), make_handler(predict, page))
    except OSError as error:
        parser.error(f"No se pudo abrir el puerto {args.port}: {error}. Prueba otro con --port.")
    print(f"Modelo: {args.checkpoint.resolve()}; dispositivo: {device}", flush=True)
    print(f"Abre http://127.0.0.1:{args.port} en tu navegador. Ctrl+C para cerrar.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterfaz cerrada.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
