import argparse
import io
from PIL import Image
import datetime

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
        'additional_info': 'Bluefish is found in all of Turkey\'s seas. It is particularly common in the Black Sea and Marmara. It is consumed grilled, steamed, and pan-fried. It is especially more delicious during autumn months. It is a schooling fish that moves quickly. It is caught using purse seine and longline fishing methods. It is one of the most valuable fish species in Turkey.'
    },
    'skipjack_tuna': {
        'name': 'Skipjack Tuna',
        'scientific_name': 'Euthynnus alletteratus',
        'description': 'Skipjack tuna is a medium-sized tuna species with a streamlined body and dark blue-black back. It has distinctive dark stripes on its back and silver-white sides. The body is elongated and laterally compressed with a forked tail. It is known for its fast swimming capabilities and is one of the most abundant tuna species.',
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

@app.route("/")
def hello_world():
    return render_template('index.html', class_info=CLASS_INFO)

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        f = request.files['file']
        if f.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Sanitize the filename
        filename = secure_filename(f.filename)
        filename = filename.replace(' ', '_').replace('(', '').replace(')', '')
        
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', filename)
        print("upload folder is ", filepath) 
        
        # Ensure uploads directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        f.save(filepath)
        global imgpath
        predict_img.imgpath = filename
        print("printing predict_img :::::: ", predict_img)
                                           
        file_extension = filename.rsplit('.', 1)[1].lower() 
        
        if file_extension == 'jpg':
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({"error": "Error reading image file"}), 400

            # Perform the detection
            model = YOLO('my_model.pt')
            results = model(img, save=True)
            
            # Get the latest detection folder
            folder_path = 'runs/detect'
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            if not subfolders:
                return jsonify({"error": "No detection results found"}), 400
                
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
            
            # Look for image0.jpg in the latest detection folder
            result_image_path = os.path.join(folder_path, latest_subfolder, 'image0.jpg')
            if not os.path.exists(result_image_path):
                return jsonify({"error": "Detection image not found"}), 404
            
            # Process detection results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Case insensitive matching for all Turkish fish names
                    class_name_lower = class_name.lower()
                    class_info = None
                    
                    # Map lowercase names to proper Turkish names
                    turkish_name_mapping = {
                        'levrek': 'Levrek',
                        'mirmir': 'Mirmir',
                        'barbunya': 'Barbunya',
                        'karagoz': 'Karagöz',
                        'kefal': 'Kefal',
                        'kupez': 'Kupes',
                        'palamut': 'Palamut',
                        'sardalya': 'Sardalya',
                        'uskumru': 'Uskumru',
                        'cipura': 'Çipura',
                        'iskorpit': 'İskorpit',
                        'istavrit': 'İstavrit',
                        'izmarit': 'İzmarit',
                        'denizati': 'Denizatı',
                        'kırlangıç': 'Kırlangıç',
                        'kirlangic': 'Kırlangıç',
                        'lüfer': 'Lüfer',
                        'lufer': 'Lüfer'
                    }
                    
                    if class_name_lower in turkish_name_mapping:
                        proper_name = turkish_name_mapping[class_name_lower]
                        class_info = CLASS_INFO[proper_name]
                    else:
                        class_info = CLASS_INFO.get(class_name, {})
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'info': class_info
                    })
            
            # Return the relative path for the frontend
            relative_image_path = f"/detection/{latest_subfolder}/image0.jpg"
            
            return jsonify({
                'success': True,
                'image_path': relative_image_path,
                'detections': detections
            })
        elif file_extension == 'mp4': 
            video_path = filepath
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "Error opening video file", 400

            # get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
            
            # initialize the YOLOv11 model here
            model = YOLO('my_model.pt')
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break                                                      

                results = model(frame, save=True)
                print(results)
                cv2.waitKey(1)

                res_plotted = results[0].plot()
                cv2.imshow("result", res_plotted)
                
                out.write(res_plotted)

                if cv2.waitKey(1) == ord('q'):
                    break

            return video_feed()
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    return render_template('index.html', class_info=CLASS_INFO)

def get_latest_detection_path():
    try:
        folder_path = 'runs/detect'
        if not os.path.exists(folder_path):
            return None
            
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        if not subfolders:
            return None
            
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        latest_dir_path = os.path.join(folder_path, latest_subfolder)
        
        # Get the most recent image file
        image_files = [f for f in os.listdir(latest_dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            return None
            
        latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(latest_dir_path, x)))
        return os.path.join(latest_subfolder, latest_image)
    except Exception as e:
        print(f"Error getting latest detection path: {str(e)}")
        return None

@app.route('/detection/<path:subfolder>/<path:filename>')
def serve_detection(subfolder, filename):
    try:
        # Construct the full path to the detection image
        detection_path = os.path.join('runs', 'detect', subfolder, filename)
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
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
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
        # Get the most recent predict directory
        predict_dirs = [d for d in os.listdir('runs/detect') if d.startswith('predict')]
        if not predict_dirs:
            return jsonify({'success': False, 'error': 'No detections found'})
        
        latest_dir = max(predict_dirs, key=lambda x: int(x.replace('predict', '')))
        latest_dir_path = os.path.join('runs/detect', latest_dir)
        
        # Look for image0.jpg in the latest directory
        image_path = os.path.join(latest_dir_path, 'image0.jpg')
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'No image found in latest detection'})
        
        relative_path = f'/detection/{latest_dir}/image0.jpg'
        
        # Get detection information from the latest detection
        detections = []
        model = YOLO('my_model.pt')
        results = model(image_path)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Case insensitive matching for all Turkish fish names
                class_name_lower = class_name.lower()
                class_info = None
                
                # Map lowercase names to proper Turkish names
                turkish_name_mapping = {
                    'levrek': 'Levrek',
                    'mirmir': 'Mirmir',
                    'barbunya': 'Barbunya',
                    'karagoz': 'Karagöz',
                    'kefal': 'Kefal',
                    'kupez': 'Kupes',
                    'palamut': 'Palamut',
                    'sardalya': 'Sardalya',
                    'uskumru': 'Uskumru',
                    'cipura': 'Çipura',
                    'iskorpit': 'İskorpit',
                    'istavrit': 'İstavrit',
                    'izmarit': 'İzmarit',
                    'denizati': 'Denizatı',
                    'kırlangıç': 'Kırlangıç',
                    'kirlangic': 'Kırlangıç',
                    'lüfer': 'Lüfer',
                    'lufer': 'Lüfer'
                }
                
                if class_name_lower in turkish_name_mapping:
                    proper_name = turkish_name_mapping[class_name_lower]
                    class_info = CLASS_INFO[proper_name]
                else:
                    class_info = CLASS_INFO.get(class_name, {})
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'info': class_info
                })
        
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
    model = YOLO('my_model.pt')
    
    # Print class names from the model
    print("\nClass names in the model:")
    for idx, name in model.names.items():
        print(f"Class {idx}: {name}")
    print("\n")
    
    app.run(host="0.0.0.0", port=args.port) 
