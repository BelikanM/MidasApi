from flask import Flask, request, jsonify
from flask_cors import CORS
import trimesh
import vtk
import numpy as np
import torch
import cv2
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Charger le modèle MiDaS via torch.hub
model_type = "MiDaS_small"  # Plus léger pour le temps réel
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Transformations pour MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

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

@app.route("/upload", methods=["POST"])
def upload_ply():
    try:
        file = request.files["file"]
        if not file.filename.endswith(".ply"):
            return jsonify({"error": "Le fichier doit être au format .ply"}), 400

        mesh = trimesh.load(file, file_type="ply")
        if not isinstance(mesh, trimesh.PointCloud):
            return jsonify({"error": "Le fichier n'est pas un nuage de points valide"}), 400

        positions = mesh.vertices.astype(np.float32)
        
        if len(positions) > 50000:
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

        return jsonify({"positions": positions, "colors": colors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload_image", methods=["POST"])
def upload_image():
    try:
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        
        # Réduire la résolution pour le temps réel
        image = image.resize((320, 240), Image.LANCZOS)
        image_np = np.array(image)

        # Prétraitement pour MiDaS
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(240, 320),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        depth = prediction
        positions = []
        colors = []
        h, w = depth.shape
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                z = depth[y, x]
                x_norm = (x - w / 2) / max(w, h)
                y_norm = (y - h / 2) / max(w, h)
                z_norm = z / np.max(depth)
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

        return jsonify({"positions": positions.tolist(), "colors": colors if colors else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run_script", methods=["POST"])
def run_script():
    try:
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
                    "position": cmd.get("position", [0, 0, 0]),
                    "radius": cmd.get("radius", 0.015),
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
                            "position": point.tolist(),
                            "radius": 0.015,
                            "color": cmd.get("color", "#ff00ff"),
                        })
                    else:
                        valid_points.append(point.tolist())

            elif cmd["type"] == "draw_lines" and len(valid_points) > 1:
                annotations.append({
                    "type": "line",
                    "points": valid_points,
                    "color": cmd.get("color", "#00ff00"),
                    "thickness": cmd.get("thickness", 0.01),
                })

        return jsonify({"annotations": annotations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run_inference", methods=["POST"])
def run_inference():
    try:
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
            t = (z - z_min) / (z_max - z_min + 1e-6)
            r = int(255 * t)
            g = 0
            b = int(255 * (1 - t))
            color_array.InsertNextTuple3(r, g, b)
        
        polydata.GetPointData().SetScalars(color_array)
        
        annotations = []
        for i in range(polydata.GetNumberOfPoints()):
            if np.random.random() > 0.5:
                point = polydata.GetPoint(i)
                annotations.append({
                    "type": "sphere",
                    "position": list(point),
                    "radius": 0.015,
                    "color": script.get("actions", [{}])[0].get("color", "#ff00ff"),
                })

        return jsonify({"annotations": annotations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
