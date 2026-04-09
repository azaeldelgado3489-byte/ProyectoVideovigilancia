import imageio_ffmpeg
import cv2
import face_recognition as fr
import requests, os, datetime, time
import numpy as np
import subprocess

# RTSP de la cámara
RTSP = "rtsp://cisco123:admin123@10.3.167.246/stream1"

# API de eventos
API = "https://galleryip.duckdns.org/api/eventos"
TOKEN = "clavesecreta123"

# Modelo de detección
MODEL = "hog"
print(f"[INFO] Usando modelo de detección: {MODEL}")

# Cooldown
ultimos_capturados = {}
COOLDOWN = 5

# Comando FFMPEG para abrir el RTSP
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(),
    "-rtsp_transport", "tcp",
    "-i", RTSP,
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Tamaño del frame (ajustar si es necesario)
WIDTH = 1280
HEIGHT = 720

try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
except Exception as e:
    print("[ERROR] No se pudo iniciar FFMPEG:", e)
    exit(1)

print("[INFO] Stream abierto correctamente")

def leer_frame():
    raw = process.stdout.read(WIDTH * HEIGHT * 3)
    if len(raw) != WIDTH * HEIGHT * 3:
        return None
    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
    return frame

while True:
    try:
        frame_bgr = leer_frame()
        if frame_bgr is None:
            print("[WARN] No se pudo leer frame")
            time.sleep(0.2)
            continue

        ts = datetime.datetime.now()
        boxes = fr.face_locations(frame_bgr, model=MODEL)
        encs = fr.face_encodings(frame_bgr, boxes)

        for i, (top, right, bottom, left) in enumerate(boxes):
            nombre = "persona"
            confianza = 1.0
            rostro = frame_bgr[top:bottom, left:right]
            rostro_id = f"{nombre}_{i}"

            if rostro_id in ultimos_capturados:
                delta = (ts - ultimos_capturados[rostro_id]).total_seconds()
                if delta < COOLDOWN:
                    continue

            ultimos_capturados[rostro_id] = ts

            fname = f"rostro_{ts.strftime('%Y%m%d_%H%M%S')}_{nombre}.jpg"
            clip_name = f"clip_{ts.strftime('%Y%m%d_%H%M%S')}_{nombre}.mp4"

            cv2.imwrite(fname, rostro)

            # Grabar 5 segundos de video
            print("[INFO] Grabando clip...")
            frames = []
            for _ in range(50):
                frame_clip = leer_frame()
                if frame_clip is not None:
                    resized = cv2.resize(frame_clip, (640, 360))
                    frames.append(resized)
                else:
                    break

            if frames:
                out = cv2.VideoWriter(clip_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 360))
                for f in frames:
                    out.write(f)
                out.release()

            # Enviar a la API
            data = {
                "nombre": nombre,
                "confianza": confianza,
                "tipo": "entrada",
                "fecha": ts.isoformat(),
                "token": TOKEN
            }

            try:
                with open(fname, "rb") as f_img, open(clip_name, "rb") as f_vid:
                    files = {
                        "imagen": (fname, f_img, "image/jpeg"),
                        "video": (clip_name, f_vid, "video/mp4")
                    }
                    r = requests.post(API, files=files, data=data, timeout=20)
                    print(f"[INFO] API respondió: {r.status_code} {r.text}")
            except Exception as e:
                print("[ERROR] Fallo al enviar a la API:", e)
            finally:
                for f in (fname, clip_name):
                    if os.path.exists(f):
                        os.remove(f)

    except KeyboardInterrupt:
        print("[INFO] Interrumpido por el usuario.")
        break
    except Exception as e:
        print("[ERROR] Fallo en el procesamiento del frame:", e)
        time.sleep(0.5)

print("[INFO] Finalizando...")
process.terminate()
