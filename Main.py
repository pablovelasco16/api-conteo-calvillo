import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv # <-- Importamos dotenv para leer archivos .env

# Cargar las variables ocultas del archivo .env al sistema (solo funciona en tu equipo local)
load_dotenv()

# --- 1. Configuración Segura de Supabase ---
SUPABASE_URL = "https://mfpuvlmqvnesxwxgjpcu.supabase.co"

# Leemos la llave secreta desde el entorno del sistema (sea de Render o del .env)
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Bloque de seguridad estricto: Si la llave no existe, el programa se detiene y te avisa.
if not SUPABASE_KEY:
    raise ValueError("🚨 ERROR CRÍTICO: No se encontró la variable de entorno SUPABASE_SERVICE_KEY. Verifica tu archivo .env local o las variables en el panel de Render.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
model = YOLO('yolov8n.pt')

# --- 2. Función de Inserción ---
def enviar_conteo_a_supabase(punto_id, entradas, salidas, precision, modelo_ver):
    try:
        data = {
            "punto_id": punto_id,
            "entradas": entradas,
            "salidas": salidas,
            "total_momento": entradas - salidas,
            "metadata_ia": {
                "confianza": precision,
                "version": modelo_ver,
                "motor": "YOLOv8"
            }
        }
        response = supabase.table("registros_conteo").insert(data).execute()
        print(f"✅ Registro en Supabase exitoso: ID {response.data[0]['id']}")
    except Exception as e:
        print(f"❌ Error al conectar con Supabase: {e}")

# --- 3. Lógica del WebSocket y Visión ---
@app.websocket("/ws/conteo")
async def websocket_conteo(websocket: WebSocket):
    await websocket.accept()
    
    counted_ids = set()
    total_tourists = 0

    try:
        while True:
            data = await websocket.receive_text()
            encoded_data = data.split(',')[1] if ',' in data else data
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
            height, width, _ = frame.shape
            line_y = int(height / 2)
            cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Extraemos la confianza (precision) de cada detección
                confidences = results[0].boxes.conf.cpu().tolist() 

                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cy = int((y1 + y2) / 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # --- EL MOMENTO EXACTO DEL CONTEO ---
                    if cy > line_y and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_tourists += 1
                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 5)
                        
                        # Disparamos la función hacia Supabase
                        enviar_conteo_a_supabase(
                            punto_id="Plaza_Principal_Calvillo", 
                            entradas=1, 
                            salidas=0, 
                            precision=round(conf, 2), # Redondeamos la confianza a 2 decimales
                            modelo_ver="YOLOv8n"
                        )

            cv2.putText(frame, f"Turistas: {total_tourists}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            procesado_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{procesado_base64}")

    except WebSocketDisconnect:
        print("La aplicación del cliente se desconectó.")