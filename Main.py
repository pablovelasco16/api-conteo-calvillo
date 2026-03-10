# main.py
# Instalar: pip install fastapi uvicorn ultralytics opencv-python-headless numpy

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# Cargar el modelo al iniciar el servidor
model = YOLO('yolov8n.pt')

@app.websocket("/ws/conteo")
async def websocket_conteo(websocket: WebSocket):
    await websocket.accept()
    
    counted_ids = set()
    total_tourists = 0

    try:
        while True:
            # 1. Recibir el frame desde la App (formato Base64)
            data = await websocket.receive_text()
            
            # Decodificar la imagen de Base64 a un formato que OpenCV entienda
            encoded_data = data.split(',')[1] if ',' in data else data
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 2. Procesar con YOLO
            results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
            
            # Configurar la línea virtual dinámicamente según el tamaño del video que mande el celular
            height, width, _ = frame.shape
            line_y = int(height / 2)
            cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cy = int((y1 + y2) / 2)

                    # Dibujar en el frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Lógica de conteo
                    if cy > line_y and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_tourists += 1
                        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 5)

            # Escribir el total
            cv2.putText(frame, f"Turistas: {total_tourists}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 3. Codificar la imagen procesada de vuelta a Base64
            _, buffer = cv2.imencode('.jpg', frame)
            procesado_base64 = base64.b64encode(buffer).decode('utf-8')

            # 4. Enviar el frame dibujado de regreso a la App
            await websocket.send_text(f"data:image/jpeg;base64,{procesado_base64}")

    except WebSocketDisconnect:
        print("La aplicación del cliente se desconectó.")