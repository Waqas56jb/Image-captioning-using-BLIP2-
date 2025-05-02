import os
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
from detection import process_csv_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_csv_file, args=(filepath,))
        thread.start()
        
        return jsonify({'message': 'File uploaded and processing started', 'filename': filename}), 202
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results')
def get_results():
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis_results.csv')
    
    if not os.path.exists(results_file):
        return jsonify({'error': 'Results not ready yet'}), 404
    
    results = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['cowboy_hat'] = row['cowboy_hat'].lower() == 'true'
            row['bikini'] = row['bikini'].lower() == 'true'
            row['horse'] = row['horse'].lower() == 'true'
            row['tractor'] = row['tractor'].lower() == 'true'
            row['one_man'] = row['one_man'].lower() == 'true'
            row['one_woman'] = row['one_woman'].lower() == 'true'
            row['face_count'] = int(row['face_count'])
            results.append(row)
    
    return jsonify(results)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)