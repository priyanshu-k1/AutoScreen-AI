from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session,make_response
import os
from werkzeug.utils import secure_filename
import sys
import logging
from pathlib import Path
import json
from flask_session import Session
from utils import modelProcessor
from flask.json.provider import DefaultJSONProvider
import numpy as np
import pandas as pd
from fpdf import FPDF
import io
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the resumeParser module 
try:
    from utils import resumeParser
    logger.info(f"Successfully imported resumeParser module")
except ImportError as e:
    logger.error(f"Error importing resumeParser: {str(e)}")
    # Create a fallback module with the necessary function
    class FallbackParser:
        def parse_resume(self, file_path):
            logger.warning("Using fallback parsing function - no resumeParser module")
            return {
                "file_name": os.path.basename(file_path),
                "email": "example@example.com",
                "phone": "123-456-7890",
                "skills": ["No skills detected - fallback mode"],
                "text_sample": "Resume parser not available. This is fallback content.",
                "status": "File uploaded successfully but not parsed"
            }
    # Create a fallback instance
    resumeParser = FallbackParser()
class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)
app = Flask(__name__, 
           static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FrontEnd'),
           static_url_path='',
           template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'FrontEnd'),
           )
app.secret_key = "1903@kp"
# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BackEnd', 'data')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 
app.config['SESSION_TYPE'] = 'filesystem'
app.json = CustomJSONProvider(app)
Session(app)
logo_path = os.path.join(app.static_folder, 'asserts', 'vector', 'appIcon.png')
class CustomPDF(FPDF):
    def footer(self):
        self.set_y(-15)  # Position 15mm from bottom
        self.set_font('Arial', 'I', 8)  # Italic font, size 8
        self.set_text_color(128)  # Gray color
        timestamp = datetime.now().strftime('Generated on: %d %B %Y')
        self.cell(0, 10, timestamp, 0, 0, 'C')  # Center aligned
# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
#function to convert NumPy types to native Python types
def convert_numpy_types(obj):
    # Handle numpy types
    if hasattr(obj, 'item'):  # Catches all numpy types
        return obj.item()
    # Handle containers
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    # Return unchanged for other types
    return obj
#function to simplify the job description from frontend
def simplify_data(input_data):
    if isinstance(input_data, str):
        data = json.loads(input_data.replace("'", '"')) 
    else:
        data = input_data
    job_roles = [role['value'] for role in data['jobRoles']]
    job_roles.extend(data['skills'])
    job_roles = list(set(job_roles))  # Remove duplicates
    simplified = {
        'skills': job_roles,
        'isFresher': data['isFresher'],
        'yearsExperience': data['yearsExperience']
    }
    return simplified

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads with proper validation and error handling"""
    # Configure upload limits
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB total
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
    
    # Check if request contains files
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'error': 'No file part'}), 400

    # Get files as a list
    uploaded_files = request.files.getlist('file')
    
    # Validate at least one file was selected
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        logger.warning("No files selected")
        return jsonify({'error': 'No files selected'}), 400

    # Check total content length
    total_size = sum(len(file.read()) for file in uploaded_files)
    for file in uploaded_files:  # Reset file pointers after reading
        file.seek(0)
    
    if total_size > MAX_CONTENT_LENGTH:
        logger.warning(f"Total upload size {total_size} exceeds limit")
        return jsonify({
            'error': f'Total size exceeds {MAX_CONTENT_LENGTH/1024/1024}MB limit',
            'max_size': MAX_CONTENT_LENGTH
        }), 413

    results = []
    upload_folder = app.config['UPLOAD_FOLDER']
    
    # Create upload directory if it doesn't exist
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for file in uploaded_files:
        file_info = {
            'filename': file.filename,
            'status': 'pending',
            'size': len(file.read())
        }
        file.seek(0)  # Reset file pointer after reading size
        
        # Validate individual file
        if not allowed_file(file.filename):
            file_info.update({
                'status': 'rejected',
                'error': 'Invalid file type',
                'allowed_types': list(ALLOWED_EXTENSIONS)
            })
            logger.warning(f"Rejected invalid file type: {file.filename}")
            results.append(file_info)
            continue

        if file_info['size'] > MAX_FILE_SIZE:
            file_info.update({
                'status': 'rejected',
                'error': f'File exceeds {MAX_FILE_SIZE/1024/1024}MB limit'
            })
            logger.warning(f"Rejected large file: {file.filename} ({file_info['size']} bytes)")
            results.append(file_info)
            continue

        # Process valid file
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            
            # Save file
            file.save(filepath)
            logger.info(f"Saved file: {filepath}")
            file_info.update({
                'saved_path': filepath,
                'status': 'saved'
            })

            # Parse file
            try:
                parsed_data = resumeParser.parse_resume(filepath)
                file_info.update({
                    'status': 'processed',
                    'data': parsed_data
                })
            
                
            except Exception as parse_error:
                file_info.update({
                    'status': 'parse_error',
                    'error': str(parse_error)
                })
                logger.error(f"Parse error for {filename}: {str(parse_error)}")

        except Exception as save_error:
            file_info.update({
                'status': 'save_error',
                'error': str(save_error)
            })
            logger.error(f"Save error for {filename}: {str(save_error)}")

        results.append(file_info)

    # Store results in session
    session['parsed_results'] = results
    
    # Prepare response
    successful = [r for r in results if r['status'] == 'processed']
    errors = [r for r in results if r['status'] not in ('processed', 'pending')]
    
    response = {
        'success': bool(successful),
        'processed': len(successful),
        'errors': len(errors),
        'details': results
    }
    # Only include redirect if successful
    if successful:
        response['redirect'] = url_for('show_results')
    return jsonify(response)
@app.route('/results')
def show_results():
    results = session.get('parsed_results', [])

    # Process each result to ensure consistent structure
    processed_results = []
    for result in results:
        # Initialize data dictionary if not present
        if 'data' not in result:
            result['data'] = {}
        
        # Ensure filename is available at top level
        if 'filename' not in result:
            result['filename'] = result['data'].get('file_name', 'Unknown file')
        
        # Ensure certifications exist and are properly formatted
        if 'certifications' not in result['data']:
            result['data']['certifications'] = []
        elif not isinstance(result['data']['certifications'], list):
            # Convert to list if it's not already
            result['data']['certifications'] = [result['data']['certifications']]
        
        # Clean certification data
        cleaned_certs = []
        for cert in result['data']['certifications']:
            if isinstance(cert, str):
                # If certification is just a string, convert to dict format
                cleaned_certs.append({'name': cert})
            elif isinstance(cert, dict):
                # Ensure required fields exist
                cert.setdefault('name', 'Unnamed Certification')
                cert.setdefault('provider', None)
                cert.setdefault('cert_number', None)
                cleaned_certs.append(cert)
        
        result['data']['certifications'] = cleaned_certs
        
        # Add a has_certifications flag for easy template checking
        result['data']['has_certifications'] = bool(cleaned_certs)
        
        processed_results.append(result)
    
    # Calculate summary statistics
    total_files = len(processed_results)
    total_with_certs = sum(1 for r in processed_results if r['data']['has_certifications'])
    
    # Add summary to session for potential use elsewhere
    session['results_summary'] = {
        'total_files': total_files,
        'total_with_certs': total_with_certs,
        'cert_percentage': round((total_with_certs / total_files * 100) if total_files else 0, 1)
    }
    
    return render_template('result.html', 
                         results=processed_results,
                         summary=session['results_summary'])



@app.route('/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_size = os.path.getsize(file_path)
                files.append({
                    'name': filename,
                    'size': file_size,
                    'uploaded_at': os.path.getctime(file_path)
                })
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500
    
    
@app.route('/save-preferences', methods=['POST'])
def save_preferences():
    """Save job screening preferences"""
    try:
        # Get the JSON data sent from the frontend
        preferences = request.json
        # Store the preferences in the session for easy access
        session['job_preferences'] = preferences
        
        # You could also save to a file or database
        preferences_file = os.path.join(app.config['UPLOAD_FOLDER'], 'job_preferences.json')
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f, indent=4)
        return jsonify({'success': True, 'message': 'Preferences saved successfully'})
    except Exception as e:
        logger.error(f"Error saving preferences: {str(e)}")
        return jsonify({'error': f'Error saving preferences: {str(e)}'}), 500
#This function will send the selected models and files to model for processing
model_processor = modelProcessor.ModelProcessor()


@app.route('/process-resumes', methods=['POST'])
def process_resumes():
    """Process resumes with selected model and return top matches"""
    try:
        # Get data from request
        data = request.json
        modelType=f"{data.get('model','skills')}+{data.get('intensity', 'casual')}"
        logger.info(f"\nProcessing resumes with model: {modelType}")
        # Get parsed results from session``
        parsed_results = session.get('parsed_results', [])
        job_prefs =simplify_data(session.get('job_preferences', {})) 
        
        if not parsed_results:
            return jsonify({'error': 'No parsed resumes found'}), 400
        
        # Extract the resume data we need
        resumes_data = []
        for result in parsed_results:
            if result.get('status') == 'processed' and 'data' in result:
                resumes_data.append(result['data'])
        
        # Process the resumes
        top_matches = model_processor.process_resumes(
            resumes_data=resumes_data,
            job_prefs=job_prefs,
            model_type=modelType
        )
        
        # Prepare response
        response = {
            'success': True,
            'count': len(top_matches),
            'model_used': modelType,
            'results': top_matches
        }
        session['analysis_results'] = response['results']
        try:
            return jsonify(response)
        except Exception as e:
            try:
                converted_response = convert_numpy_types(response)
                return jsonify(converted_response)
            except Exception as e:
                logger.error(f"Error converting response to JSON: {str(e)}")
                return jsonify({'error': 'Error converting response to JSON'}), 500
    
    except Exception as e:
        logger.error(f"Error processing resumes: {str(e)}")
        return jsonify({'error': f'Error processing resumes: {str(e)}'}), 500

@app.route('/export/excel', methods=['POST'])
def export_excel():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Results', startrow=5)  # Start after 5 rows (to make space for logo)

            workbook = writer.book
            worksheet = writer.sheets['Results']



            logo_path = os.path.join(app.static_folder, 'FrontEnd/asserts/vector/appIcon.png')  
            if os.path.exists(logo_path):
                worksheet.insert_image('A1', logo_path, {'x_scale': 0.5, 'y_scale': 0.5})

            # Title after logo
            title_format = workbook.add_format({
                'bold': True, 'font_color': 'white',
                'bg_color': '#4CAF50', 'align': 'center', 'valign': 'vcenter', 'font_size': 14
            })
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#D9EAD3', 'border': 1
            })
            cell_format = workbook.add_format({'border': 1})

            worksheet.merge_range('A5:{}5'.format(chr(65 + len(df.columns) - 1)), 'AutoScreen AI - Resume Screening Report', title_format)

            # Headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(5, col_num, value, header_format)

            # Data
            for row_num in range(len(df)):
                for col_num in range(len(df.columns)):
                    worksheet.write(row_num + 6, col_num, df.iloc[row_num, col_num], cell_format)

        output.seek(0)
        response = make_response(output.read())
        response.headers['Content-Disposition'] = 'attachment; filename=AutoScreenAI_Results.xlsx'
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()

        pdf = FPDF()
        pdf.add_page()

        # Add Logo
        logo_path = os.path.join(app.static_folder, 'asserts', 'vector', 'appIcon.png')
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=30)

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AutoScreen AI - Resume Screening Report", 0, 1, "C")
        pdf.ln(20)

        pdf.set_font("Arial", "B", 12)

        # Custom columns we want to show
        columns = ['file_name', 'email', 'certification_count', 'score', 'match']

        # Header Row
        for col in columns:
            pdf.cell(38, 10, col.replace('_', ' ').title(), border=1, align='C')  # Nice title format
        pdf.ln()

        pdf.set_font("Arial", "", 11)

        # Fill data
        for item in data:
            file_name = item.get('file_name', 'N/A')
            email = item.get('email', 'N/A')
            certification_count = item.get('certification_count', 0)
            score = f"{(item.get('education_score', 0) * 100):.1f}%"  # Use education_score as "Score" (convert to %)
            match = 'Match' if item.get('experience_score', 0) > 0 else 'Not a Match'

            row = [file_name, email, certification_count, score, match]

            for cell in row:
                text = str(cell)
                if len(text) > 20:  # truncate very long text
                    text = text[:17] + '...'
                pdf.cell(38, 10, text, border=1, align='C')
            pdf.ln()

        # Save PDF to memory
        pdf_output = pdf.output(dest='S').encode('latin1')

        response = make_response(pdf_output)
        response.headers['Content-Disposition'] = 'attachment; filename=AutoScreenAI_Results.pdf'
        response.headers['Content-Type'] = 'application/pdf'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
    
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a specific file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a specific file"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'File {filename} deleted successfully'})
        else:
            logger.warning(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting AUTOSCREEN-AI backend server...")
    app.run(debug=True, host='0.0.0.0', port=5000)