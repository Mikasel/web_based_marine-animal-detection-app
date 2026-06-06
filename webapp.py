import argparse
import io
from PIL import Image
import datetime
import base64

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, send_from_directory, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob


from ultralytics import YOLO


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, 'my_model.pt')
DETECT_DIR = os.path.join(APP_ROOT, 'runs', 'detect')
_yolo_model = None

# Model class names / Turkish aliases -> CLASS_INFO keys
CLASS_ALIAS_TO_INFO_KEY = {
    'levrek': 'european_seabass',
    'mirmir': 'sand_steenbras',
    'barbunya': 'red_Mullet',
    'karagoz': 'two_banded_seabream',
    'kefal': 'flathead_grey_mullet',
    'kupez': 'bogue',
    'palamut': 'atlantic_bonito',
    'sardalya': 'Sardalya',
    'uskumru': 'mackerel',
    'cipura': 'gilt_head_bream',
    'iskorpit': 'scorpion_fish',
    'istavrit': 'atlantic_horse_mackerel',
    'izmarit': 'blotched_picarel',
    'denizati': 'sea_horse',
    'kirlangic': 'tub_gurnard',
    'lufer': 'bluefish',
}


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                "Place my_model.pt in the project root."
            )
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def normalize_class_name(class_name):
    normalized = class_name.lower().strip().replace(' ', '_')
    for tr_char, en_char in (
        ('ı', 'i'), ('ğ', 'g'), ('ü', 'u'), ('ş', 's'),
        ('ö', 'o'), ('ç', 'c'), ('İ', 'i'),
    ):
        normalized = normalized.replace(tr_char, en_char)
    return normalized


def resolve_class_info(class_name):
    normalized = normalize_class_name(class_name)
    info_key = CLASS_ALIAS_TO_INFO_KEY.get(normalized, normalized)
    if info_key in CLASS_INFO:
        return CLASS_INFO[info_key]
    if class_name in CLASS_INFO:
        return CLASS_INFO[class_name]
    return {}


def build_detections(model, results):
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            detections.append({
                'class': class_name,
                'confidence': conf,
                'info': resolve_class_info(class_name),
            })
    return detections


def run_detection(model, source):
    os.makedirs(DETECT_DIR, exist_ok=True)
    return model.predict(source=source, save=True, project=DETECT_DIR, exist_ok=True)


def get_latest_detection_artifacts():
    if not os.path.isdir(DETECT_DIR):
        return None, None
    subfolders = [
        f for f in os.listdir(DETECT_DIR)
        if os.path.isdir(os.path.join(DETECT_DIR, f))
    ]
    if not subfolders:
        return None, None
    latest_subfolder = max(
        subfolders,
        key=lambda x: os.path.getctime(os.path.join(DETECT_DIR, x)),
    )
    latest_dir_path = os.path.join(DETECT_DIR, latest_subfolder)
    image_files = [
        f for f in os.listdir(latest_dir_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    if not image_files:
        return None, None
    latest_image = max(
        image_files,
        key=lambda x: os.path.getctime(os.path.join(latest_dir_path, x)),
    )
    return latest_subfolder, latest_image

# Class information dictionary
CLASS_INFO = {
    'european_seabass': {
        'name': 'European Seabass',
        'scientific_name': 'Dicentrarchus labrax',
        'description': 'European seabass, also known as sea bass, is a fish species commonly found in the Atlantic Ocean and Mediterranean Sea. It has a silvery gray color, laterally compressed body, and is quite delicious.',
        'habitat': 'Coastal waters, estuaries, and lagoons',
        'size': 'Average 30-50 cm, maximum 1 meter',
        'diet': 'Small fish, crustaceans, and mollusks',
        'conservation_status': 'Common and stable',
        'additional_info': 'European seabass is commonly found in Turkey, especially along the Aegean and Mediterranean coasts. It is a commercially valuable fish species.'
    },
    'sand_steenbras': {
        'name': 'Sand steenbras',
        'scientific_name': 'Lithognathus mormyrus',
        'description': 'Sand steenbras is a silvery colored fish species that lives in sandy and muddy seabeds. Its body color ranges from light brown to yellow tones. The back is darker and becomes silvery on the sides. The belly is white. There are 14-15 brown bands extending parallel from the back. It is particularly common in the Aegean and Mediterranean.',
        'habitat': 'Sandy and muddy seabeds, coastal waters',
        'size': 'Average 25-40 cm, maximum 60 cm',
        'diet': 'Small fish, crustaceans, and mollusks',
        'conservation_status': 'Common',
        'additional_info': 'Sand steenbras is especially more delicious during winter months. It is consumed grilled and steamed.'
    },
    'red_Mullet': {
        'name': 'Red Mullet',
        'scientific_name': 'Mullus barbatus',
        'description': 'Red mullet is a reddish pink colored fish species with whiskers. It is commonly found in the Mediterranean and Aegean Sea. It is known for its delicious meat and characteristic appearance.',
        'habitat': 'Sandy and muddy seabeds, 10-300 meters depth',
        'size': 'Average 15-25 cm, maximum 40 cm',
        'diet': 'Small crustaceans, mollusks, and marine worms',
        'conservation_status': 'Common',
        'additional_info': 'Red mullet holds an important place in Turkish cuisine. It is consumed grilled, steamed, and pan-fried. It is especially more delicious during summer months.'
    },
    'two_banded_seabream': {
        'name': 'Two banded seabream',
        'scientific_name': 'Diplodus vulgaris',
        'description': 'The most distinctive feature of two banded seabream is the black band extending from top to bottom behind the head and on the tail stalk. The back, anus, and edges of the tail fin are black. There are 8 sharp teeth in each jaw and numerous molar teeth behind them. It is an omnivorous species.',
        'habitat': 'Sandy and muddy seabeds, 10-300 meters depth',
        'size': 'Average 15-25 cm, maximum 40 cm',
        'diet': 'Small fish, crustaceans, mollusks, and marine plants',
        'conservation_status': 'Common',
        'additional_info': 'They are caught using bottom trawls and longlines.'
    },
    'flathead_grey_mullet': {
        'name': 'Flathead Grey Mullet',
        'scientific_name': 'Mugil cephalus',
        'description': 'The body has a high swimming form. The head is slightly compressed sideways with a wide mouth. The back is black navy blue. The sides are white but the scales are yellow. Yellow lines extend from head to tail along the scale rows. The belly is white. The entire gill cover is golden yellow up to the eye. Yellow color and small black spots can be seen on all fins.',
        'habitat': 'Sandy and muddy seabeds, 0-300 meters depth',
        'size': 'Average 30-50 cm, maximum 80 cm',
        'diet': 'Algae, small crustaceans, and organic matter',
        'conservation_status': 'Common',
        'additional_info': 'While they feed on animal planktonic organisms in their early stages, they later consume plant organisms as well. Flathead grey mullet is a very intelligent, strong, and agile fish, making its fishing quite challenging. It rarely bites on hooks. The most efficient fishing is done with cast nets.'
    },
    'bogue': {
        'name': 'Bogue',
        'scientific_name': 'Boops boops',
        'description': 'The back is yellow-green in color. There are 13-15 spine rays and 12-16 soft rays on the dorsal fin. The lateral line is dark, almost black, and distinct. There are 69-80 scales on the lateral line. The entire body is covered with scales. The sides are silvery with 4 thin yellow bands extending from behind the gill cover to the tail. The tail is yellow with small black spots.',
        'habitat': 'Sandy and muddy seabeds, 0-300 meters depth',
        'size': 'Average 20-30 cm, maximum 50 cm',
        'diet': 'Small fish, crustaceans, and plankton',
        'conservation_status': 'Common',
        'additional_info': 'The first sexual maturity length is 13 cm (1 year); 78% of individuals up to 19 cm in length are females. It is caught using purse seine and trawl nets.'
    },
    'atlantic_bonito': {
        'name': 'Atlantic Bonito',
        'scientific_name': 'Sarda sarda',
        'description': 'Atlantic bonito is a fast swimming fish species with a dark blue-green back and silvery sides. It has 5-11 dark stripes on its back. The body is long and laterally compressed. The tail fin is forked.',
        'habitat': 'Open seas, coastal waters, 0-200 meters depth',
        'size': 'Average 40-60 cm, maximum 90 cm',
        'diet': 'Small fish, squid, and other marine creatures',
        'conservation_status': 'Common',
        'additional_info': 'Atlantic bonito is found in all of Turkey\'s seas. It is consumed grilled, steamed, and as lakerda. It is especially more delicious during autumn months. It is a schooling fish that moves quickly.'
    },
    'Sardalya': {
        'name': 'Sardalya',
        'scientific_name': 'Sardina pilchardus',
        'description': 'Sardine is a small fish species with a blue-green back and silvery sides. The body is long and laterally compressed. It has dark spots on its back. The tail fin is forked.',
        'habitat': 'Open seas, coastal waters, 0-100 meters depth',
        'size': 'Average 15-20 cm, maximum 25 cm',
        'diet': 'Plankton, small crustaceans, and larvae',
        'conservation_status': 'Common',
        'additional_info': 'Sardine is found in all of Turkey\'s seas. It is consumed grilled, steamed, and canned. It is especially more delicious during summer months. It is a schooling fish that moves quickly.'
    },
    'mackerel': {
        'name': 'Mackerel',
        'scientific_name': 'Scomber scombrus',
        'description': 'Mackerel is a fast-swimming fish species with a blue-green back and silvery sides. It has dark stripes on its back. The body is long and laterally compressed. The tail fin is forked.',
        'habitat': 'Open seas, coastal waters, 0-200 meters depth',
        'size': 'Average 20-30 cm, maximum 50 cm',
        'diet': 'Small fish, squid, and plankton',
        'conservation_status': 'Common',
        'additional_info': 'Mackerel is found in all of Turkey\'s seas. It is consumed grilled, steamed, and as lakerda. It is especially more delicious during autumn months. It is a schooling fish that moves quickly. It is caught using purse seine and longline fishing methods.'
    },
    'gilt_head_bream': {
        'name': 'Gilt Head Bream',
        'scientific_name': 'Sparus aurata',
        'description': 'Gilt-head bream is a round-bodied fish species with a dark gray-blue back and silvery sides. It has golden spots on its back and sides. There is a distinct golden band above its head. The tail fin is forked.',
        'habitat': 'Coastal waters, lagoons, 0-150 meters depth',
        'size': 'Average 25-35 cm, maximum 70 cm',
        'diet': 'Small fish, crustaceans, mollusks, and marine plants',
        'conservation_status': 'Common',
        'additional_info': 'Gilt-head bream is found in all of Turkey\'s seas. It is consumed grilled, steamed, and baked. It is especially more delicious during autumn and winter months. It is an important species that is both naturally occurring and farmed.'
    },
    'scorpion_fish': {
        'name': 'Scorpion Fish',
        'scientific_name': 'Scorpaena porcus',
        'description': 'Scorpion fish is a fish species with dark brown and reddish tones, covered with spots and lines. The body is thick and round, the head is large and spiny. It has poisonous spines on its dorsal fin. The eyes are large and protruding.',
        'habitat': 'Rocky and coral areas, 0-100 meters depth',
        'size': 'Average 15-25 cm, maximum 40 cm',
        'diet': 'Small fish, crustaceans, and mollusks',
        'conservation_status': 'Common',
        'additional_info': 'Scorpion fish is found in all of Turkey\'s seas. It is consumed grilled and steamed. Care must be taken when catching and cleaning it as its spines are poisonous. It is especially more delicious during winter months.'
    },
    'atlantic_horse_mackerel': {
        'name': 'Atlantic Horse Mackerel',
        'scientific_name': 'Trachurus trachurus',
        'description': 'Atlantic horse mackerel is a fast swimming fish species with a blue-green back and silvery sides. It has dark stripes on its back. The body is long and laterally compressed. The tail fin is forked. It has dark scales along its lateral line.',
        'habitat': 'Open seas, coastal waters, 0-200 meters depth',
        'size': 'Average 15-25 cm, maximum 50 cm',
        'diet': 'Small fish, crustaceans, and plankton',
        'conservation_status': 'Common',
        'additional_info': 'Atlantic horse mackerel is found in all of Turkey\'s seas. It is consumed grilled, steamed, and pan-fried. It is especially more delicious during autumn months. It is a schooling fish that moves quickly. It is caught using purse seine and trawl nets.'
    },
    'blotched_picarel': {
        'name': 'Blotched Picarel',
        'scientific_name': 'Spicara maena',
        'description': 'Blotched picarel is a small fish species with a blue-green back and silvery sides. The body is round and laterally compressed. It has dark spots on its back. The tail fin is forked. The dorsal fin is longer and more prominent in males.',
        'habitat': 'Coastal waters, rocky areas, 0-100 meters depth',
        'size': 'Average 10-15 cm, maximum 20 cm',
        'diet': 'Small crustaceans, plankton, and larvae',
        'conservation_status': 'Common',
        'additional_info': 'Blotched picarel is found in all of Turkey\'s seas. It is consumed grilled and pan-fried. It is especially more delicious during summer months. It is a schooling fish that moves quickly. It is caught using purse seine and longline fishing methods.'
    },
    'sea_horse': {
        'name': 'Sea Horse',
        'scientific_name': 'Hippocampus hippocampus',
        'description': 'Seahorse is a unique fish species known for its horse like head and curled tail. Its body is covered with bony plates. Its color can vary depending on its environment, usually in brown, yellow, or gray tones. Its eyes can move independently, and it can hold onto objects with its tail.',
        'habitat': 'Seagrass beds, coral reefs, lagoons, 0-50 meters depth',
        'size': 'Average 10-15 cm, maximum 20 cm',
        'diet': 'Small crustaceans, plankton, and larvae',
        'conservation_status': 'Endangered',
        'additional_info': 'Seahorse is found in all of Turkey\'s seas. Male seahorses carry and give birth to the young. It is a slow-moving species with high camouflage ability. Its population is decreasing due to marine pollution and habitat loss. Its hunting is prohibited for use in traditional Chinese medicine.'
    },
    'tub_gurnard': {
        'name': 'Tub gurnard',
        'scientific_name': 'Chelidonichthys lucerna',
        'description': 'Tub gurnard is a reddish-brown colored fish species with a large head and wide pectoral fins. Its pectoral fins are wing-shaped, which is why it is named "gurnard". The body is laterally compressed and covered with scales. It has spines on its head and three-fingered structures under its pectoral fins.',
        'habitat': 'Sandy and muddy seabeds, 20-300 meters depth',
        'size': 'Average 30-50 cm, maximum 75 cm',
        'diet': 'Small fish, crustaceans, mollusks, and marine worms',
        'conservation_status': 'Common',
        'additional_info': 'Tub gurnard is found in all of Turkey\'s seas. It is particularly common in the Aegean and Mediterranean. It is consumed grilled, steamed, and in soup. While the head part is used for making soup, the body part is preferred grilled. It is caught using bottom trawls and longlines. It is a fish species with delicious meat and high economic value.'
    },
    'bluefish': {
        'name': 'Bluefish',
        'scientific_name': 'Pomatomus saltatrix',
        'description': 'Bluefish is a fast swimming fish species with a blue-green back and silvery sides. The body is long and laterally compressed. It has dark stripes on its back. The tail fin is forked. The mouth is large and equipped with sharp teeth. It has different names according to its size: 10-15 cm is called "defne yaprağı", 15-18 cm is "çinekop", 18-25 cm is "sarıkanat", 25-35 cm is "lüfer", and over 35 cm is "kofana".',
        'habitat': 'Open seas, coastal waters, 0-200 meters depth',
        'size': 'Average 25-35 cm, maximum 130 cm',
        'diet': 'Small fish, squid, and other marine creatures',
        'conservation_status': 'Common',
        'additional_info': 'Bluefish is found in all of Turkey\'s seas. It is particularly common in the Black Sea and Marmara. It is consumed grilled, steamed, and pan fried. It is especially more delicious during autumn months. It is a schooling fish that moves quickly. It is caught using purse seine and longline fishing methods. It is one of the most valuable fish species in Turkey.'
    },
    'skipjack_tuna': {
        'name': 'Skipjack Tuna',
        'scientific_name': 'Euthynnus alletteratus',
        'description': 'Skipjack tuna is a medium-sized tuna species with a streamlined body and dark blue-black back. It has distinctive dark stripes on its back and silver/white sides. The body is elongated and laterally compressed with a forked tail. It is known for its fast swimming capabilities and is one of the most abundant tuna species.',
        'habitat': 'Open oceans, coastal waters, 0-200 meters depth',
        'size': 'Average 40-60 cm, maximum 100 cm',
        'diet': 'Small fish, squid, crustaceans, and plankton',
        'conservation_status': 'Common',
        'additional_info': 'Skipjack tuna is found in all of Turkey\'s seas, particularly abundant in the Mediterranean. It is consumed grilled, steamed, and canned. It is a schooling fish that moves quickly and is caught using purse seine and longline fishing methods. It has delicious meat and high economic value. It is one of the most important species for the canned tuna industry.'
    }
}

# Add favicon route
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'assets'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        try:
            if 'file' not in request.files:
                return jsonify({"success": False, "error": "No file part"}), 400

            f = request.files['file']
            if f.filename == '':
                return jsonify({"success": False, "error": "No selected file"}), 400

            filename = secure_filename(f.filename)
            filename = filename.replace(' ', '_').replace('(', '').replace(')', '')

            filepath = os.path.join(APP_ROOT, 'uploads', filename)
            print("upload folder is ", filepath)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            f.save(filepath)
            predict_img.imgpath = filename

            file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)
                if img is None:
                    return jsonify({"success": False, "error": "Error reading image file"}), 400

                model = get_yolo_model()
                results = run_detection(model, filepath)

                latest_subfolder, result_image = get_latest_detection_artifacts()
                if not latest_subfolder or not result_image:
                    return jsonify({"success": False, "error": "No detection results found"}), 400

                detections = build_detections(model, results)
                relative_image_path = f"/detection/{latest_subfolder}/{result_image}"

                return jsonify({
                    'success': True,
                    'image_path': relative_image_path,
                    'detections': detections
                })
            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return jsonify({"success": False, "error": "Error opening video file"}), 400

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    os.path.join(APP_ROOT, 'output.mp4'),
                    fourcc, 30.0, (frame_width, frame_height),
                )

                model = get_yolo_model()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = run_detection(model, frame)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)
                    if cv2.waitKey(1) == ord('q'):
                        break

                cap.release()
                out.release()
                return video_feed()
            else:
                return jsonify({"success": False, "error": "Unsupported file format"}), 400
        except FileNotFoundError as e:
            return jsonify({"success": False, "error": str(e)}), 500
        except Exception as e:
            print(f"Error in predict_img: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    return render_template('index.html', class_info=CLASS_INFO)

def get_latest_detection_path():
    try:
        latest_subfolder, latest_image = get_latest_detection_artifacts()
        if not latest_subfolder or not latest_image:
            return None
        return os.path.join(latest_subfolder, latest_image)
    except Exception as e:
        print(f"Error getting latest detection path: {str(e)}")
        return None

@app.route('/detection/<path:subfolder>/<path:filename>')
def serve_detection(subfolder, filename):
    try:
        # Construct the full path to the detection image
        detection_path = os.path.join(DETECT_DIR, subfolder, filename)
        print(f"Attempting to serve image from: {detection_path}")  # Debug log
        
        # Check if the file exists
        if not os.path.exists(detection_path):
            print(f"File not found at path: {detection_path}")  # Debug log
            return jsonify({"error": "Detection image not found"}), 404
            
        # Get the absolute path
        abs_path = os.path.abspath(detection_path)
        print(f"Absolute path: {abs_path}")  # Debug log
        
        # Return the file using send_file instead of send_from_directory
        return send_file(
            abs_path,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Error serving detection image: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500

# #The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    try:
        # Remove leading slash if present
        filename = filename.lstrip('/')
        # Convert forward slashes to system-specific separators
        filename = filename.replace('/', os.sep)
        
        # Check if the file exists
        if not os.path.exists(filename):
            return jsonify({"error": "File not found"}), 404
            
        # Return the file
        return send_file(filename)
        
    except Exception as e:
        print(f"Error in display function: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
        
        

def get_frame():
    mp4_files = os.path.join(APP_ROOT, 'output.mp4')
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        


@app.route('/get_latest_detection')
def get_latest_detection():
    try:
        latest_dir, latest_image = get_latest_detection_artifacts()
        if not latest_dir or not latest_image:
            return jsonify({'success': False, 'error': 'No detections found'})

        image_path = os.path.join(DETECT_DIR, latest_dir, latest_image)
        relative_path = f'/detection/{latest_dir}/{latest_image}'

        model = get_yolo_model()
        results = model.predict(source=image_path, save=False)
        detections = build_detections(model, results)

        return jsonify({
            'success': True,
            'image_path': relative_path,
            'detections': detections
        })

    except Exception as e:
        print(f"Error in get_latest_detection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/latest_detection')
def get_latest_detection_image():
    try:
        latest_path = get_latest_detection_path()
        if not latest_path:
            return jsonify({"error": "No detection images found"}), 404
            
        subfolder, filename = os.path.split(latest_path)
        return redirect(url_for('serve_detection', subfolder=subfolder, filename=filename))
    except Exception as e:
        print(f"Error getting latest detection image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = get_yolo_model()

    # Print class names from the model
    print("\nClass names in the model:")
    for idx, name in model.names.items():
        print(f"Class {idx}: {name}")
    print("\n")
    
    app.run(host="0.0.0.0", port=args.port) 
