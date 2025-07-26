import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import trimesh
import vtk
import numpy as np
import torch
import cv2
from PIL import Image
import io
import logging
from ultralytics import YOLO
from queue import Queue
import threading
import json
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limite à 10 Mo
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# File d'attente pour limiter les requêtes simultanées
request_queue = Queue(maxsize=5)
queue_lock = threading.Lock()

# Charger le modèle MiDaS
try:
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    logger.info("Modèle MiDaS chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement de MiDaS : {str(e)}")
    raise e

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Charger le modèle YOLOv8
try:
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    logger.info("Modèle YOLOv8 chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement de YOLOv8 : {str(e)}")
    raise e

# Stockage temporaire des annotations
latest_annotations = []

def create_vtk_point_cloud(positions, colors=None):
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    
    for i, pos in enumerate(positions):
        points.InsertNextPoint(pos)
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    
    if colors is not None:
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")
        for color in colors:
            color_array.InsertNextTuple3(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
        polydata.GetPointData().SetScalars(color_array)
    
    return polydata

def create_ply_file(positions, colors=None):
    vertex_count = len(positions)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header.append("end_header")

    lines = header[:]
    for i in range(vertex_count):
        pos = positions[i]
        if colors is not None:
            col = colors[i]
            line = f"{pos[0]} {pos[1]} {pos[2]} {int(col[0] * 255)} {int(col[1] * 255)} {int(col[2] * 255)}"
        else:
            line = f"{pos[0]} {pos[1]} {pos[2]}"
        lines.append(line)
    
    ply_content = "\n".join(lines)
    return io.BytesIO(ply_content.encode("ascii")), "point_cloud.ply"

def compute_topography(polydata):
    points = np.array(polydata.GetPoints().GetData())
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    
    slopes = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        slope = abs(dz / (dx + 1e-6))
        slopes.append(slope)
    
    avg_slope = np.mean(slopes) if slopes else 0
    return {
        "min_height": float(z_min),
        "max_height": float(z_max),
        "avg_slope": float(avg_slope),
    }

@app.route("/upload", methods=["POST"])
def upload_ply():
    with queue_lock:
        if request_queue.full():
            logger.warning("File d'attente pleine, requête ignorée")
            return jsonify({"error": "Serveur occupé, réessayez plus tard"}), 429
        request_queue.put_nowait(True)
    
    try:
        logger.debug("Requête /upload reçue")
        if "file" not in request.files:
            logger.error("Aucun fichier envoyé")
            request_queue.get_nowait()
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".ply"):
            logger.error(f"Fichier non PLY : {file.filename}")
            request_queue.get_nowait()
            return jsonify({"error": "Le fichier doit être au format .ply"}), 400

        cloud_id = str(uuid.uuid4())  # Générer un ID unique
        logger.debug(f"Traitement du fichier PLY : {file.filename}, cloud_id: {cloud_id}")
        file_content = file.read()
        try:
            mesh = trimesh.load(file_obj=io.BytesIO(file_content), file_type="ply")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du PLY : {str(e)}")
            request_queue.get_nowait()
            return jsonify({"error": f"Erreur lors du chargement du PLY : {str(e)}"}), 400

        if not isinstance(mesh, trimesh.PointCloud):
            logger.error("Le fichier n'est pas un nuage de points valide")
            request_queue.get_nowait()
            return jsonify({"error": "Le fichier n'est pas un nuage de points valide"}), 400

        positions = mesh.vertices.astype(np.float32)
        logger.debug(f"Positions extraites : {len(positions)} points")

        if len(positions) > 50000:
            logger.debug("Décimation du nuage de points")
            polydata = create_vtk_point_cloud(positions)
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputData(polydata)
            decimate.SetTargetReduction(0.7)
            decimate.Update()
            reduced_polydata = decimate.GetOutput()
            positions = np.array(reduced_polydata.GetPoints().GetData())
        
        positions = positions.tolist()
        
        colors = None
        if hasattr(mesh, "colors") and mesh.colors is not None:
            colors = mesh.colors[:, :3].astype(float) / 255.0
            colors = colors.tolist()
            logger.debug(f"Couleurs extraites : {len(colors)}")

        ply_file, filename = create_ply_file(positions, colors)
        logger.debug("Fichier PLY généré pour /upload")
        request_queue.get_nowait()
        return send_file(
            ply_file,
            mimetype="text/plain",
            as_attachment=True,
            download_name=filename
        ), 200, {"X-Cloud-ID": cloud_id}
    except Exception as e:
        logger.error(f"Erreur dans /upload : {str(e)}")
        request_queue.get_nowait()
        return jsonify({"error": str(e)}), 500

@app.route("/upload_image", methods=["POST"])
def upload_image():
    with queue_lock:
        if request_queue.full():
            logger.warning("File d'attente pleine, requête ignorée")
            return jsonify({"error": "Serveur occupé, réessayez plus tard"}), 429
        request_queue.put_nowait(True)

    try:
        cloud_id = str(uuid.uuid4())  # Générer un ID unique
        logger.debug(f"Requête /upload_image reçue, cloud_id: {cloud_id}")
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        logger.debug("Image reçue et ouverte")
        
        image = image.resize((160, 120), Image.LANCZOS)
        image_np = np.array(image)

        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        input_batch = transform(img).to(device)
        logger.debug("Image prétraitée pour MiDaS")

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(120, 160),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        logger.debug("Prédiction de profondeur terminée")

        depth = prediction

        results = yolo(image_np, conf=0.5)
        global latest_annotations
        latest_annotations = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo.names[cls]
                if label in ["person", "car", "truck", "animal", "bed", "tv"]:
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    z = float(depth[y_center, x_center] / np.max(depth))
                    x_norm = float((x_center - 160 / 2) / max(160, 120))
                    y_norm = float((y_center - 120 / 2) / max(160, 120))
                    latest_annotations.append({
                        "type": "sphere",
                        "position": [x_norm, -y_norm, z],
                        "radius": 0.02,
                        "color": "#ff0000",
                        "label": label,
                    })
        logger.debug(f"{len(latest_annotations)} anomalies détectées par YOLOv8")

        positions = []
        colors = []
        h, w = depth.shape
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                z = depth[y, x]
                x_norm = float((x - w / 2) / max(w, h))
                y_norm = float((y - h / 2) / max(w, h))
                z_norm = float(z / np.max(depth))
                positions.append([x_norm, -y_norm, z_norm])
                colors.append(image_np[y, x] / 255.0)

        positions = np.array(positions, dtype=np.float32)
        if len(positions) > 50000:
            polydata = create_vtk_point_cloud(positions, colors)
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputData(polydata)
            decimate.SetTargetReduction(0.7)
            decimate.Update()
            reduced_polydata = decimate.GetOutput()
            positions = np.array(reduced_polydata.GetPoints().GetData())
            colors = colors[:len(positions)] if colors else None
        logger.debug(f"Nuage de points créé : {len(positions)} points")

        ply_file, filename = create_ply_file(positions, colors)
        logger.debug("Fichier PLY généré")
        request_queue.get_nowait()
        return send_file(
            ply_file,
            mimetype="text/plain",
            as_attachment=True,
            download_name=filename
        ), 200, {"X-Cloud-ID": cloud_id}
    except Exception as e:
        logger.error(f"Erreur dans /upload_image : {str(e)}")
        request_queue.get_nowait()
        return jsonify({"error": str(e)}), 500

@app.route("/store_image", methods=["POST"])
def store_image():
    try:
        logger.debug("Requête /store_image reçue")
        if "file" not in request.files:
            logger.error("Aucun fichier envoyé")
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files["file"]
        image_id = str(uuid.uuid4())
        file.save(f"images/{image_id}.jpg")
        logger.debug(f"Image sauvegardée : {image_id}.jpg")
        return jsonify({"image_id": image_id}), 200
    except Exception as e:
        logger.error(f"Erreur dans /store_image : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_annotations", methods=["POST"])
def get_annotations():
    try:
        global latest_annotations
        serializable_annotations = []
        for anno in latest_annotations:
            serializable_anno = anno.copy()
            serializable_anno["position"] = [float(x) for x in anno["position"]]
            serializable_anno["radius"] = float(anno["radius"])
            serializable_annotations.append(serializable_anno)
        logger.debug(f"Envoi de {len(serializable_annotations)} annotations")
        return jsonify({"annotations": serializable_annotations})
    except Exception as e:
        logger.error(f"Erreur dans /get_annotations : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/run_script", methods=["POST"])
def run_script():
    try:
        logger.debug("Requête /run_script reçue")
        data = request.get_json()
        commands = data.get("commands", [])
        positions = np.array(data.get("positions", []), dtype=np.float32)

        if len(positions) == 0:
            return jsonify({"error": "Aucun nuage de points chargé"}), 400

        positions = positions.reshape(-1, 3)
        polydata = create_vtk_point_cloud(positions)
        
        annotations = []
        valid_points = []

        for cmd in commands:
            if cmd["type"] == "measure":
                bounds = polydata.GetBounds()
                dimensions = {
                    "width": float(bounds[1] - bounds[0]),
                    "height": float(bounds[3] - bounds[2]),
                    "depth": float(bounds[5] - bounds[4]),
                }
                return jsonify({"measurements": dimensions, "annotations": annotations})

            elif cmd["type"] == "highlight":
                annotations.append({
                    "type": "sphere",
                    "position": [float(x) for x in cmd.get("position", [0, 0, 0])],
                    "radius": float(cmd.get("radius", 0.015)),
                    "color": cmd.get("color", "#00ff00"),
                })

            elif cmd["type"] == "analyze_anomalies":
                x_values = positions[:, 0]
                y_values = positions[:, 1]
                x_mean = np.mean(x_values)
                y_mean = np.mean(y_values)
                num = np.sum((x_values - x_mean) * (y_values - y_mean))
                denom = np.sum((x_values - x_mean) ** 2)
                m = num / denom if denom != 0 else 0
                b = y_mean - m * x_mean

                threshold = cmd.get("threshold", 0.05)
                for i, point in enumerate(positions):
                    expected_y = m * point[0] + b
                    distance = abs(point[1] - expected_y)
                    if distance > threshold:
                        annotations.append({
                            "type": "sphere",
                            "position": [float(x) for x in point.tolist()],
                            "radius": 0.015,
                            "color": cmd.get("color", "#ff00ff"),
                        })
                    else:
                        valid_points.append([float(x) for x in point.tolist()])

            elif cmd["type"] == "draw_lines" and len(valid_points) > 1:
                annotations.append({
                    "type": "line",
                    "points": valid_points,
                    "color": cmd.get("color", "#00ff00"),
                    "thickness": float(cmd.get("thickness", 0.01)),
                })

            elif cmd["type"] == "topography":
                topo_data = compute_topography(polydata)
                annotations.append({
                    "type": "sphere",
                    "position": [0, 0, float(topo_data["max_height"])],
                    "radius": 0.03,
                    "color": cmd.get("color", "#ffff00"),
                    "label": f"Max Height: {topo_data['max_height']:.2f}",
                })
                return jsonify({"measurements": topo_data, "annotations": annotations})

        return jsonify({"annotations": annotations})
    except Exception as e:
        logger.error(f"Erreur dans /run_script : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/run_inference", methods=["POST"])
def run_inference():
    try:
        logger.debug("Requête /run_inference reçue")
        data = request.get_json()
        script = data.get("script", {})
        positions = np.array(data.get("positions", []), dtype=np.float32)

        if len(positions) == 0:
            return jsonify({"error": "Aucun nuage de points chargé"}), 400

        positions = positions.reshape(-1, 3)
        polydata = create_vtk_point_cloud(positions)
        
        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")
        z_values = positions[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        
        for z in z_values:
            t = float((z - z_min) / (z_max - z_min + 1e-6))
            r = int(255 * t)
            g = 0
            b = int(255 * (1 - t))
            color_array.InsertNextTuple3(r, g, b)
        
        polydata.GetPointData().SetScalars(color_array)
        
        annotations = []
        topo_data = compute_topography(polydata)
        annotations.append({
            "type": "sphere",
            "position": [0, 0, float(topo_data["max_height"])],
            "radius": 0.03,
            "color": script.get("actions", [{}])[0].get("color", "#ffff00"),
            "label": f"Max Height: {topo_data['max_height']:.2f}",
        })

        return jsonify({"annotations": annotations, "measurements": topo_data})
    except Exception as e:
        logger.error(f"Erreur dans /run_inference : {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
