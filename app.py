from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from datetime import datetime

from app.face_recog import load_known_faces, recognize_face
from app.attendance import mark_attendance
from app.nlp_tagging import extract_tags

app = Flask(__name__)

known_encodings, known_names = load_known_faces()

@app.route('/mark-attendance', methods=['POST'])
def mark():
    data = request.json
    image_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    name = recognize_face(frame, known_encodings, known_names)
    if name:
        mark_attendance(name)
        tags = extract_tags(f"{name} marked present at {datetime.now()}")
        return jsonify({'status': 'success', 'person': name, 'tags': tags}), 200
    else:
        return jsonify({'status': 'failed', 'message': 'Face not recognized'}), 404

if __name__ == "__main__":
    app.run(debug=True)
