import os
import json
import re
import mimetypes
from flask import Flask, request, jsonify,render_template,redirect
from flask_cors import CORS
# Use google.generativeai instead of google.genai based on newer library versions
from google import genai
from google.genai import types
import logging
import traceback
import tempfile
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
# --- Firebase Initialization ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

logging.basicConfig(level=logging.INFO)

firebase_app = None
db = None

# Option 1 (Recommended for Deployment): Load credentials from an environment variable
firebase_credentials_json_string = os.environ.get('FIREBASE_CREDENTIALS_JSON')

if firebase_credentials_json_string:
    try:
        # Parse the JSON string content from the environment variable
        cred = credentials.Certificate(json.loads(firebase_credentials_json_string))
        firebase_app = firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully from environment variable.")

    except json.JSONDecodeError as e:
        print(f"Firebase initialization error: Failed to parse FIREBASE_CREDENTIALS_JSON environment variable as JSON: {e}")
        print("Ensure the FIREBASE_CREDENTIALS_JSON environment variable contains the exact JSON content of your service account key file.")
        # The app will continue but without Firebase functionality
    except Exception as e:
        print(f"Unexpected error initializing Firebase from environment variable: {e}")
         # Handle other potential errors during initialization
        # The app will continue but without Firebase functionality

else:
    # Option 2 (Fallback for Local Development): Load credentials from a local file
    # Make sure this file path is correct for your local environment
    # IMPORTANT: Ensure this file is NOT committed to your public repository!
    local_credential_path = 'nova-maths-feedback-firebase-adminsdk-fbsvc-9c15ee16fc.json'
    if os.path.exists(local_credential_path):
        try:
            cred = credentials.Certificate(local_credential_path) # Pass the file path directly
            firebase_app = firebase_admin.initialize_app(cred)
            print(f"Firebase initialized successfully from local file: {local_credential_path}")

        except Exception as e:
            print(f"Error initializing Firebase from local file '{local_credential_path}': {e}")
            print("Ensure the local credential file exists and is valid.")
            # The app will continue but without Firebase functionality
    else:
        print(f"Firebase initialization skipped: Neither FIREBASE_CREDENTIALS_JSON environment variable nor local file '{local_credential_path}' found.")
        print("Feedback submission will not work until Firebase is configured.")
        # The app will start but Firebase is not initialized

# Get Firestore client reference ONLY if the app was successfully initialized
if firebase_app:
    try:
        db = firestore.client()
        print("Firestore client obtained.")
    except Exception as e:
         print(f"Error obtaining Firestore client: {e}")
         db = None # Ensure db is None if client creation failed


app = Flask(__name__)
CORS(app, resources={r"/solve-math": {"origins": "*"}, r"/clarify-step": {"origins": "*"},r"/practice": {"origins": "*"},r"/ama": {"origins": "*"},r"/check": {"origins": "*"},r"/refresher": {"origins": "*"},r"/submit-feedback": {"origins": "*"}})

# --- IMPORTANT: Use environment variables for API keys in production ---
# --- Hardcoding keys like this is insecure and should only be for quick local testing ---



API_KEY = os.environ['GEMINI_API_KEY']
client = None # Initialize as None

if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
else:
    try:
        client = genai.Client(api_key=API_KEY)
        # logging.info("Gemini API client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API client: {e}")
        client = None # Ensure client is None if initialization fails



# --- Model Configuration ---
# Choose a model appropriate for the task (multimodal if handling files)
MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Changed to 1.5 flash - good balance
CHAT_MODEL = "gemini-2.0-flash-lite"
# Send a creative prompt to the LLM

def call_gemini(prompt):
    prompt_parts = [types.Part.from_text(text=prompt)]
    response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[types.Content(role="user", parts=prompt_parts)],
            # stream=False # Default is False, explicitly set if needed
        )
    return response

def call_ama_gemini(prompt):
    prompt_parts = [types.Part.from_text(text=prompt)]
    response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[types.Content(role="user", parts=prompt_parts)],
            # stream=False # Default is False, explicitly set if needed
        )
    return response

@app.route('/ama', methods=['POST'])
def ama():
        data = request.get_json()
        question = data.get('question', '')
        history = data.get('history', '')
        hist = f"""Here's the previous chat history: {history}"""
        # Modified instructional_text to request Hindi explanations and English math
        instructional_text ="You are Ask Me Anything on Math Theories Assistant. Respond based on user question and for low context questions respond by analyzing previous chat history. Provide your answer and explanations in Hindi. Mathematical formulas and expressions should be in English using LaTeX ($...$ or $$...$$). Also in your response for giving focus to any specific word return those words with different font colors. In your responses dont mention extra things like Okay, 'I can answer your question' and dont mention 'Based on our previous conversation, you're asking about' respond by analyzing the context and behave as you are human assistant responding what you are asked for."
        instructional_text+=hist + "\n question : "+ question
        response = call_ama_gemini(instructional_text)


        # Return the solution
        return jsonify({"answer": response.text.replace("*","")})

# --- Helper Function to Build Prompt Parts (REVISED) ---
# --- Helper Function to Build Prompt Parts (REVISED) ---
def build_math_prompt_parts(problem_text=None, uploaded_gemini_file=None):
    """
    Builds the list of 'Part' objects for the Gemini API math solver.
    Returns a list containing one text Part and optionally one file Part.
    """
    # Modified instructional_text to request Hindi explanations and English math
    instructional_text = """You are MathMind AI, an expert AI Math Tutor.
Your goal is to provide clear, accurate, and step-by-step solutions to math problems.
Provide the explanation and steps in Hindi. Mathematical formulas and expressions should be in English using LaTeX ($...$ or $$...$$).
Format the output using Markdown.
Use '### Step X:' for each step heading (e.g., '### Step 1:', '### Step 2:'). Start steps from 1.
Use '### Final Answer:' for the final result section.
Use LaTeX notation enclosed in single dollar signs ($...$) for inline math and double dollar signs ($$...$$) for display math equations. This ensures compatibility with MathJax rendering on the web.
Explain the reasoning behind each step clearly and concisely.

NEW DIAGRAM INSTRUCTIONS:
If a step requires a diagram for clear explanation (e.g., geometry, graphs, vectors):
1.  First, describe the diagram in text as part of the step's explanation.
2.  Then, IMMEDIATELY AFTER the textual description of the diagram within the same step, embed the diagram data using a special tag:

    <DIAGRAM_SVG_DATA>
    {
      "viewBox": "0 0 200 150", // Example: min-x, min-y, width, height for coordinate system
      "elements": [
        // For a line:
        { "type": "line", "x1": 10, "y1": 10, "x2": 100, "y2": 100, "stroke": "black", "strokeWidth": 2 },
        // For a circle:
        { "type": "circle", "cx": 50, "cy": 50, "r": 20, "stroke": "blue", "fill": "lightblue" },
        // For text/label:
        { "type": "text", "x": 10, "y": 25, "text": "Point A", "fill": "red", "fontSize": "10px" },
        // For a rectangle:
        { "type": "rect", "x": 10, "y": 10, "width": 50, "height": 30, "stroke": "green", "fill": "lightgreen" },
        // For a path (more complex shapes):
        { "type": "path", "d": "M10 80 Q 52.5 10, 95 80 T 180 80", "stroke": "purple", "fill": "none" }
        // Add other elements as needed (e.g., polygons, ellipses)
      ]
    }
    </DIAGRAM_SVG_DATA>
3.  Ensure the JSON inside <DIAGRAM_SVG_DATA> is valid. Use simple coordinates. The viewBox defines the drawing area; keep coordinates within this box.
4.  Do NOT generate actual <svg>...</svg> tags. Only provide the data within <DIAGRAM_SVG_DATA>.
5.  Continue with the rest of the step's explanation after the </DIAGRAM_SVG_DATA> tag if necessary.

If the input (text or file) is ambiguous, unclear, or not a math problem, state that politely and ask for clarification instead of guessing or providing an irrelevant solution.
If a file is provided, analyze the content of the file (image or PDF) to identify the math problem.

---
Problem Details:
"""
    # ... (rest of the function remains the same)
    if problem_text and not uploaded_gemini_file:
        instructional_text += f"\nText Input: {problem_text}"
    # ... and so on for file input and the closing part of the prompt.
    else:
        instructional_text += "\nText Input: None provided."

    # Add file context if a file is present
    if uploaded_gemini_file:
        instructional_text += "\nFile Input: (Analyze the content of the attached file)"
    else:
        instructional_text += "\nFile Input: None provided."

    instructional_text += "\n---\nProvide the step-by-step solution based on the information above:"

    # Create the list of Part objects
    parts = [types.Part.from_text(text=instructional_text)]
    # Always include the combined text as the first part
    # parts.append(types.Part.from_text(instructional_text))


    # Add file part if it exists
    if uploaded_gemini_file:
        logging.info(f"Adding file part with URI: {uploaded_gemini_file.uri}, MIME: {uploaded_gemini_file.mime_type}")
        try:
            # Create the file part using its URI and MIME type
            parts.append(types.Part.from_uri(
                file_uri=uploaded_gemini_file.uri,
                mime_type=uploaded_gemini_file.mime_type
            ))
        except Exception as e:
            logging.error(f"Error creating Part from URI {uploaded_gemini_file.uri}: {e}")
            # Optionally add a text part indicating the file error to the AI
            parts.append(types.Part.from_text("(Error: Could not properly attach file content for analysis due to an internal issue.)"))

    return parts
# --- Helper Function to Build Prompt Parts (REVISED) ---
def build_check_prompt_parts(problem_text=None, uploaded_gemini_file=None):
    """
    Builds the list of 'Part' objects for the Gemini API math solver.
    Returns a list containing one text Part and optionally one file Part.
    """
    # Combine all instructional text and user problem text into one string
    # Modified instructional_text to request Hindi explanations and English math
    instructional_text = f"""You are MathMind AI, an expert AI Math Tutor.
Your goal is to Carefully examine the uploaded math practice problem and its solution provided by the user.
Provide your feedback and explanations in Hindi. Mathematical formulas and expressions should be in English using LaTeX ($...$ or $$...$$).

Provide your feedback using Markdown formatting for clarity. This includes:
- Using `### Main Point` or `#### Sub-point` for distinct sections or observations.
- Using blank lines between paragraphs for better readability.
- Using bullet points (`* item` or `- item`) or numbered lists (`1. item`) for lists of observations or suggestions.
- Using **bold** text to highlight key terms or observations (like "**Observation 1:**", "**Mistake:**", "**Correct Approach:**").
- Using *italic* text for emphasis.
- Using LaTeX ($...$ or $$...$$) for any mathematical expressions or formulas in your feedback.

1. First check whether the values mentioned in the solution is matching with the values mentioned in practice problem statement.
   If not point this out.
2. Next, find out whether the underlying concept is well understood by the user, or applying wrong logic/ concept to the practice problem.

3. Verify each step thoroughly for correctness, logic, and mathematical accuracy.

2. If there are any mistakes
   - Clearly point them out (e.g., "**Mistake in Step 2:** Calculation error...").
    - Explain *why* they are incorrect.
    - Suggest the correct approach or provide the correction where necessary.

3. Be polite and constructive — encourage the user to improve and keep practicing.

4. If the entire solution is accurate and well-reasoned, sincerely appreciate the user’s effort and motivate them to continue practicing at this high level.
---
Problem Details:

Practice Problem (Actual Practice Question): {problem_text}
"""

    # Add user's text input to the combined string
    if problem_text and not uploaded_gemini_file:
        instructional_text += f"\nPractice problem Statement: {problem_text}"
    else:
        instructional_text += "\nText Input: None provided."

    # Add file context if a file is present
    if uploaded_gemini_file:
        instructional_text += "\nUser Solution: "
    else:
        instructional_text += "\nFile Input: None provided."


    # Create the list of Part objects
    parts = [types.Part.from_text(text=instructional_text)]
    # Always include the combined text as the first part
    # parts.append(types.Part.from_text(instructional_text))


    # Add file part if it exists
    if uploaded_gemini_file:
        logging.info(f"Adding file part with URI: {uploaded_gemini_file.uri}, MIME: {uploaded_gemini_file.mime_type}")
        try:
            # Create the file part using its URI and MIME type
            parts.append(types.Part.from_uri(
                file_uri=uploaded_gemini_file.uri,
                mime_type=uploaded_gemini_file.mime_type
            ))
        except Exception as e:
            logging.error(f"Error creating Part from URI {uploaded_gemini_file.uri}: {e}")
            # Optionally add a text part indicating the file error to the AI
            parts.append(types.Part.from_text("(Error: Could not properly attach file content for analysis due to an internal issue.)"))

    print("full prompt :\n",parts)

    return parts



def build_pratice_prompt_parts(problem_text, no_questions):
    """
    Builds the list of 'Part' objects for the Gemini API math solver.
    Returns a list containing one text Part and optionally one file Part.
    """
    # Combine all instructional text and user problem text into one string
    # Modified instructional_text to request Hindi problems and English math
    instructional_text = f"""You are MathMind AI, an expert AI Math Tutor.
Your goal is to generate practice problems which are having same concept as in below given problem's solution.
Generate the problem text in Hindi. Mathematical formulas and expressions within the problems should be in English using LaTeX ($...$ or $$...$$).
---
Given Problem Details:{problem_text}

Note:
Generate exactly {no_questions} practice problems based on the core concepts demonstrated in the provided problem and solution.

**Formatting Requirements (VERY IMPORTANT - Follow Strictly):**
1.  The response MUST start with "1. " followed by the first problem.
2.  Each subsequent problem MUST start on a new line with the next number, a period, and a space (e.g., "2. ", "3. ").
3.  Do NOT use any Markdown formatting like asterisks (`*`) or hash symbols (`#`) for the problem numbers or titles (e.g., DO NOT use `**Problem 1:**` or `### 1.`).
4.  Do NOT include ANY introductory text before the first problem (like "Here are the problems...").
5.  Do NOT include ANY concluding text or explanations after the last problem.
6.  The entire output should consist ONLY of the numbered list of problems.


"""


    # Add file context if a file is present

    print("practice prompt ",instructional_text)
    # Create the list of Part objects
    parts = [types.Part.from_text(text=instructional_text)]
    # Always include the combined text as the first part
    # parts.append(types.Part.from_text(instructional_text))


    return parts


@app.route('/solve-math', methods=['POST'])
def solve_math():

    # Initialize to None (expecting a single object or None, not a list)
    uploaded_gemini_file_obj = None
    temp_file_path = None

    try:
        problem_text = request.form.get('problemText', '').strip()
        problem_file_storage = request.files.get('problemFile')

        # --- File Processing ---
        if problem_file_storage and problem_file_storage.filename:
            logging.info(f"Received file: {problem_file_storage.filename}, MIME: {problem_file_storage.mimetype}")
            mime_type = problem_file_storage.mimetype

            if not mime_type or (not mime_type.startswith('image/') and mime_type != 'application/pdf'):
                logging.warning(f"Unsupported file type uploaded: {mime_type}. Skipping file.")
                # Consider returning an error if file type is invalid
                # return jsonify({"error": f"Unsupported file type: {mime_type}. Please upload an image or PDF."}), 400
            else:
                try:
                    suffix = os.path.splitext(problem_file_storage.filename)[1] or mimetypes.guess_extension(mime_type) or ''
                    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
                    os.close(fd)
                    logging.info(f"Saving uploaded file to temporary path: {temp_file_path}")

                    problem_file_storage.seek(0)
                    with open(temp_file_path, 'wb') as temp_file:
                        shutil.copyfileobj(problem_file_storage, temp_file)

                    logging.info(f"Uploading temporary file {temp_file_path} to Gemini File API...")
                    # Use genai.upload_file which returns the File object
                    uploaded_gemini_file_obj = client.files.upload(
                        file=temp_file_path
                    )
                    # Assign the returned object, not append to a list
                    logging.info(f"Successfully uploaded file via File API. URI: {uploaded_gemini_file_obj.uri}")

                except Exception as e:
                    logging.error(f"Error during file processing or upload for {problem_file_storage.filename}: {e}", exc_info=True)
                    uploaded_gemini_file_obj = None # Ensure it's None if upload failed
                    # return jsonify({"error": "Failed to process the uploaded file."}), 500 # Optional: fail request

        # --- Validation ---
        if not problem_text and not uploaded_gemini_file_obj:
            logging.error("No text problem provided and no file successfully processed.")
            return jsonify({"error": "Please enter a math problem or upload a valid image/PDF file."}), 400

        # --- Construct Prompt Parts (using the revised helper) ---
        # Pass the single uploaded file object (or None)
        prompt_parts = build_math_prompt_parts(problem_text=problem_text, uploaded_gemini_file=uploaded_gemini_file_obj)

        if not prompt_parts:
            logging.error("Failed to build prompt parts.")
            return jsonify({"error": "Internal server error: Could not prepare request for AI."}), 500

        # --- Call the Gemini API ---
        logging.info(f"Sending request to Gemini model ({MODEL_NAME}) with {len(prompt_parts)} parts...")

        # The 'contents' argument expects an iterable (like a list) of Content objects.
        # For a single user turn, provide a list containing one Content object.
        # The 'parts' argument within Content takes the list of Part objects we built.
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=[types.Content(role="user", parts=prompt_parts)],
            # stream=False # Default is False, explicitly set if needed
        )


        # --- Process Response ---
        try:

            solution_text = response.text
            # Modified concept_prompt to request Hindi description
            concept_prompt = "Based on below solution of a math problem uploaded by user. Identify the concept involve in the problem to solve it. Generate the output with only provide Concept identified in the below problem. Write one liner description of the identified concept in Hindi. Do not include any introductory text only give the concept involved in the problem. "
            concept_prompt += "Solution :\n" + solution_text
            response_concept =call_gemini(concept_prompt)
            logging.info(f"Received response_concept: {response_concept.text}")

        except ValueError:
            # This might occur if the response structure is unexpected or blocked in a way
            # not caught by prompt_feedback (less common now but good to keep)
            logging.warning(f"Accessing response.text failed. Response structure might be unexpected or blocked. Response: {response}", exc_info=True)
            return jsonify({"error": "The AI could not process the request due to content restrictions or an unexpected response format."}), 500
        except AttributeError:
             logging.error(f"AttributeError accessing response data. Response: {response}", exc_info=True)
             return jsonify({"error": "Internal server error: Failed to parse AI response structure."}), 500
        except Exception as e:
            # Catch any other unexpected errors during response processing
            logging.error(f"Unexpected error extracting text from Gemini response: {e}", exc_info=True)
            return jsonify({"error": "Internal server error: Could not parse AI response."}), 500


        if not solution_text or not solution_text.strip():
            logging.warning("Gemini returned an empty solution.")
            # Check finish reason if available
            try:
                finish_reason = response.candidates[0].finish_reason.name
                if finish_reason != "STOP":
                    logging.warning(f"Generation finished unexpectedly: {finish_reason}")
                    return jsonify({"error": f"AI generation stopped unexpectedly ({finish_reason}). The solution might be incomplete or unavailable."}), 500
            except (IndexError, AttributeError, TypeError):
                 logging.warning("Could not determine finish reason from response candidates.")
            # Generic message if empty
            return jsonify({"error": "The AI returned an empty response. Please try rephrasing or check your input."}), 500

        # Return the solution
        return jsonify({"solution": solution_text,"identifiedConcepts":response_concept.text})


    except Exception as e:
        # General exception handler for the route
        logging.error(f"An unexpected error occurred in /solve-math route: {e}", exc_info=True)
        # Basic error message classification
        error_message = f"An internal server error occurred: {type(e).__name__}" # Provide type for debugging
        err_str = str(e).lower()
        if "api key" in err_str or "permission denied" in err_str or "authentication" in err_str:
            error_message = "AI service authentication failed. Check server configuration."
        elif "quota" in err_str:
            error_message = "AI service quota exceeded. Please try again later."
        elif "rate limit" in err_str:
            error_message = "Rate limit exceeded with the AI service. Please try again later."
        elif "invalid argument" in err_str or "bad request" in err_str:
             error_message = f"Invalid request sent to AI service. Please check input. ({type(e).__name__})"


        return jsonify({"error": error_message}), 500

    finally:
        # --- Cleanup ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                logging.info(f"Cleaning up temporary file: {temp_file_path}")
                os.remove(temp_file_path)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temporary file {temp_file_path}: {cleanup_error}")

        # if uploaded_gemini_file_obj: # Check the object used for the API call
        #     try:
        #         logging.info(f"Attempting to delete uploaded file from File API: {uploaded_gemini_file_obj.name}")
        #         genai.delete_file(name=uploaded_gemini_file_obj.name)
        #         logging.info(f"Successfully deleted file {uploaded_gemini_file_obj.name} from File API.")
        #     except Exception as delete_error:
        #         logging.error(f"Error deleting file {uploaded_gemini_file_obj.name} from File API: {delete_error}")




@app.route('/check', methods=['POST'])
def check():

    # Initialize to None (expecting a single object or None, not a list)
    uploaded_gemini_file_obj = None
    temp_file_path = None

    try:
        problem_text = request.form.get('practiceProblem', '').strip()
        solution_file_storage = request.files.get('solutionFile')
        print("practice problem statement : ",problem_text)
        # --- File Processing ---
        if solution_file_storage and solution_file_storage.filename:
            logging.info(f"Received file: {solution_file_storage.filename}, MIME: {solution_file_storage.mimetype}")
            mime_type = solution_file_storage.mimetype

            if not mime_type or (not mime_type.startswith('image/') and mime_type != 'application/pdf'):
                logging.warning(f"Unsupported file type uploaded: {mime_type}. Skipping file.")
                # Consider returning an error if file type is invalid
                # return jsonify({"error": f"Unsupported file type: {mime_type}. Please upload an image or PDF."}), 400
            else:
                try:
                    suffix = os.path.splitext(solution_file_storage.filename)[1] or mimetypes.guess_extension(mime_type) or ''
                    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
                    os.close(fd)
                    logging.info(f"Saving uploaded file to temporary path: {temp_file_path}")

                    solution_file_storage.seek(0)
                    with open(temp_file_path, 'wb') as temp_file:
                        shutil.copyfileobj(solution_file_storage, temp_file)

                    logging.info(f"Uploading temporary file {temp_file_path} to Gemini File API...")
                    # Use genai.upload_file which returns the File object
                    uploaded_gemini_file_obj = client.files.upload(
                        file=temp_file_path
                    )
                    # Assign the returned object, not append to a list
                    logging.info(f"Successfully uploaded file via File API. URI: {uploaded_gemini_file_obj.uri}")

                except Exception as e:
                    logging.error(f"Error during file processing or upload for {solution_file_storage.filename}: {e}", exc_info=True)
                    uploaded_gemini_file_obj = None # Ensure it's None if upload failed
                    # return jsonify({"error": "Failed to process the uploaded file."}), 500 # Optional: fail request

        # --- Validation ---
        if not problem_text and not uploaded_gemini_file_obj:
            logging.error("No text problem provided and no file successfully processed.")
            return jsonify({"error": "Please enter a math problem or upload a valid image/PDF file."}), 400

        # --- Construct Prompt Parts (using the revised helper) ---
        # Pass the single uploaded file object (or None)
        prompt_parts = build_check_prompt_parts(problem_text=problem_text, uploaded_gemini_file=uploaded_gemini_file_obj)

        if not prompt_parts:
            logging.error("Failed to build prompt parts.")
            return jsonify({"error": "Internal server error: Could not prepare request for AI."}), 500

        # --- Call the Gemini API ---
        logging.info(f"Sending request to Gemini model ({MODEL_NAME}) with {len(prompt_parts)} parts...")

        # The 'contents' argument expects an iterable (like a list) of Content objects.
        # For a single user turn, provide a list containing one Content object.
        # The 'parts' argument within Content takes the list of Part objects we built.
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=[types.Content(role="user", parts=prompt_parts)],
            # stream=False # Default is False, explicitly set if needed
        )


        # --- Process Response ---
        try:

            check_result = response.text
            logging.info(f"solution text length: {check_result}")


        except ValueError:
            # This might occur if the response structure is unexpected or blocked in a way
            # not caught by prompt_feedback (less common now but good to keep)
            logging.warning(f"Accessing response.text failed. Response structure might be unexpected or blocked. Response: {response}", exc_info=True)
            return jsonify({"error": "The AI could not process the request due to content restrictions or an unexpected response format."}), 500
        except AttributeError:
             logging.error(f"AttributeError accessing response data. Response: {response}", exc_info=True)
             return jsonify({"error": "Internal server error: Failed to parse AI response structure."}), 500
        except Exception as e:
            # Catch any other unexpected errors during response processing
            logging.error(f"Unexpected error extracting text from Gemini response: {e}", exc_info=True)
            return jsonify({"error": "Internal server error: Could not parse AI response."}), 500


        if not check_result or not check_result.strip():
            logging.warning("Gemini returned an empty solution.")
            # Check finish reason if available
            try:
                finish_reason = response.candidates[0].finish_reason.name
                if finish_reason != "STOP":
                    logging.warning(f"Generation finished unexpectedly: {finish_reason}")
                    return jsonify({"error": f"AI generation stopped unexpectedly ({finish_reason}). The solution might be incomplete or unavailable."}), 500
            except (IndexError, AttributeError, TypeError):
                 logging.warning("Could not determine finish reason from response candidates.")
            # Generic message if empty
            return jsonify({"error": "The AI returned an empty response. Please try rephrasing or check your input."}), 500

        # Return the solution
        return jsonify({"check_result": check_result})


    except Exception as e:
        # General exception handler for the route
        logging.error(f"An unexpected error occurred in /solve-math route: {e}", exc_info=True)
        # Basic error message classification
        error_message = f"An internal server error occurred: {type(e).__name__}" # Provide type for debugging
        err_str = str(e).lower()
        if "api key" in err_str or "permission denied" in err_str or "authentication" in err_str:
            error_message = "AI service authentication failed. Check server configuration."
        elif "quota" in err_str:
            error_message = "AI service quota exceeded. Please try again later."
        elif "rate limit" in err_str:
            error_message = "Rate limit exceeded with the AI service. Please try again later."
        elif "invalid argument" in err_str or "bad request" in err_str:
             error_message = f"Invalid request sent to AI service. Please check input. ({type(e).__name__})"


        return jsonify({"error": error_message}), 500

    finally:
        # --- Cleanup ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                logging.info(f"Cleaning up temporary file: {temp_file_path}")
                os.remove(temp_file_path)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temporary file {temp_file_path}: {cleanup_error}")

def build_clarification_prompt_parts(original_problem_text, original_file_uri, full_solution, step_number, user_question):
    """Builds prompt parts for the clarification request."""
    # Modified prompt to request Hindi clarification and English math
    prompt = f"""You are MathMind AI, an expert AI Math Tutor.
A user received the following solution to a math problem and needs clarification on a specific step.
Provide the clarification and explanation in Hindi. Mathematical formulas and expressions should be in English using LaTeX ($...$ or $$...$$).

--- Original Problem Context ---
Text Input: {original_problem_text or 'None provided.'}
"""
    if original_file_uri:
        prompt += f"File Input Reference URI: {original_file_uri} (Content provided separately if needed)\n"
        try:
             # Re-attach the file using the URI for context if needed
             # Note: Ensure the file hasn't expired on the File API side.
             # Gemini might infer from URI, but attaching ensures it's considered.
             # Need MIME type - this is a limitation; ideally, MIME type was stored too.
             # For now, let's assume the model can use the URI or we omit the Part.
             # A better approach is to store URI and MIME type together.
             # parts.append(genai.types.Part.from_uri(uri=original_file_uri, mime_type="image/png")) # Example MIME
             logging.info(f"Including file URI {original_file_uri} in clarification prompt text.")
        except Exception as e:
            logging.warning(f"Could not create Part from URI {original_file_uri} for clarification: {e}")
            prompt += "(Could not re-attach file content for clarification.)\n"

    else:
        prompt += "File Input: None provided.\n"

    prompt += f"""
--- Full Original Solution Provided ---
{full_solution}

--- User's Question ---
The user is asking for clarification specifically about **Step {step_number}**.
User's Question: "{user_question}"

--- Your Task ---
Provide a clear and detailed explanation addressing the user's question about Step {step_number}.
Focus *only* on clarifying that specific step in the context of the overall solution.
Do not re-solve the entire problem. Use Markdown and LaTeX ($...$ or $$...$$) for formatting as needed.
Use Markdown for formatting:
- Use `### Heading Name` for main section titles and `#### Subheading Name` for subsections within your clarification.
- Use blank lines between paragraphs.
- Use standard Markdown for bulleted lists (`* item` or `- item`) and numbered lists (`1. item`).
- Use Markdown for **bold** and *italic** text where appropriate.
- Use LaTeX ($...$ or $$...$$) for mathematical expressions.
Explain the reasoning, definitions, or formulas relevant to that particular step and the user's query.
Make the explanation easy to read and understand.
"""
    # Add the text part first
    parts = [types.Part.from_text(text=prompt)]
    # parts.insert(0, prompt)

    # Add the file part if URI exists and we have a way to get MIME type (or guess)
    # This part is tricky without storing MIME type initially.
    # Let's rely on the URI being mentioned in the text prompt for now.

    return parts


# --- NEW ENDPOINT for Step Clarification ---
@app.route('/clarify-step', methods=['POST'])
def clarify_step():


    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request body. Expected JSON."}), 400

        # Extract data from request
        original_problem_text = data.get('originalProblemText', '')
        original_file_uri = data.get('originalProblemFileURI') # Get the URI stored by frontend
        full_solution = data.get('fullSolution', '')
        step_number = data.get('stepNumber')
        user_question = data.get('userQuestion', '').strip()

        # Validation
        if not full_solution or step_number is None or not user_question:
            return jsonify({"error": "Missing required fields: fullSolution, stepNumber, or userQuestion."}), 400
        if not isinstance(step_number, int) or step_number <= 0:
             return jsonify({"error": "Invalid stepNumber."}), 400

        # --- Construct Clarification Prompt Parts ---
        prompt_parts = build_clarification_prompt_parts(
            original_problem_text, original_file_uri, full_solution, step_number, user_question
        )

        # --- Call the Gemini API ---
        # logging.info(f"Sending clarification request to Gemini model ({model.model_name})...")
        # response = model.generate_content(prompt_parts)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=prompt_parts)],
            # stream=False # Default is False, explicitly set if needed
        )
         # --- Process Response ---
        try:
            if not response.candidates:
                 logging.error(f"Clarification response blocked or empty. Prompt Feedback: {response.prompt_feedback}")
                 block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "Unknown"
                 return jsonify({"error": f"Clarification request blocked by safety filter: {block_reason}."}), 400

            clarification_text = response.text
            logging.info(f"Received clarification text length: {len(clarification_text)}")

        except (ValueError, AttributeError, IndexError) as e:
            logging.error(f"Error extracting text from clarification response: {e}. Response: {response}", exc_info=True)
            return jsonify({"error": "Internal server error: Could not parse AI clarification response."}), 500

        if not clarification_text or not clarification_text.strip():
            logging.warning("Gemini returned an empty clarification.")
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            if finish_reason != "STOP":
                logging.warning(f"Clarification generation finished unexpectedly: {finish_reason}")
                return jsonify({"error": f"AI clarification generation stopped unexpectedly ({finish_reason})."}), 500
            return jsonify({"error": "The AI returned an empty clarification response."}), 500


        return jsonify({"clarification": clarification_text})


    except Exception as e:
        logging.error(f"An unexpected error occurred in /clarify-step route: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during clarification: {type(e).__name__}"}), 500





@app.route('/practice', methods=['POST'])
def practice():


    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request body. Expected JSON."}), 400

        # Extract data from request
        original_problem_text = data.get('contextText', '')
        no_probs = data.get('numberOfProblems') # Get the URI stored by frontend

        print("solution generated : ",original_problem_text)
        print("------------------------------------------------")
        print("number of problems : ",no_probs)



        # --- Construct Clarification Prompt Parts ---
        prompt_parts = build_pratice_prompt_parts(
            original_problem_text, no_probs )

        # --- Call the Gemini API ---
        # logging.info(f"Sending clarification request to Gemini model ({model.model_name})...")
        # response = model.generate_content(prompt_parts)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=prompt_parts)]
            # stream=False # Default is False, explicitly set if needed
        )
         # --- Process Response ---
        try:

            practice_problems = response.text
            logging.info(f"Received practice text length: {len(practice_problems)}")
            logging.info(f"Practice text : {practice_problems}")

        except (ValueError, AttributeError, IndexError) as e:
            logging.error(f"Error extracting text from clarification response: {e}. Response: {response}", exc_info=True)
            return jsonify({"error": "Internal server error: Could not parse AI clarification response."}), 500


        # --- Return the clarification ---
        # Convert basic markdown/latex from clarification to HTML before sending?
        # Or let frontend handle it? Let's assume frontend MathJax handles LaTeX.
        # We might want basic Markdown conversion here later if needed.
        return jsonify({"practice_text": practice_problems})


    except Exception as e:
        logging.error(f"An unexpected error occurred in /clarify-step route: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred during clarification: {type(e).__name__}"}), 500




# --- NEW ENDPOINT for Step Clarification ---
@app.route('/refresher', methods=['POST'])
def refresher():

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body. Expected JSON."}), 400

    # Extract data from request
    identified_concept = data.get('concepts', '')
    print("identified_concept :\n",identified_concept)
    # Modified refresher_prompt to request Hindi explanations and English math within JSON values
    refresher_prompt ="""You are an AI assistant designed to generate technical refreshers in a strict JSON format.

    Based on the mathematical concept identified below, generate a JSON object containing the following information: concept name, a detailed description, an example, and a list of solved questions.
    Provide the description, example explanations, and solved question explanations/solutions in Hindi. Mathematical formulas and expressions within these values should be in English using LaTeX ($...$ or $$...$$).
    STRICT JSON FORMATTING RULES:
    Your response MUST contain ONLY a single, valid JSON object.
    DO NOT include any text, markdown, or characters before or after the JSON object.
    DO NOT wrap the JSON object in markdown code blocks (e.g., DO NOT use json ...).
    The JSON object MUST have these top-level keys: "concept_name", "concept_description", "example", "solved_questions".
    Ensure ALL string values within the JSON (including values for keys like "example", "question", and "solution") are properly escaped for JSON (e.g., internal double quotes must be " , newlines \n, backslashes \).
    For the "example" and "solution" keys, their values MUST be plain strings. DO NOT include markdown code blocks (like python or) inside these string values. Provide the code or math examples directly as strings.
    The "solved_questions" key MUST contain a JSON array [...]. Each item in this array MUST be a JSON object {{...}} with exactly two keys: "question" (string) and "solution" (string).
    Use this exact structure example (replace content with actual data):
    {{
    "concept_name": "<Identified Concept Name>",
    "concept_description": "<Detailed description in Hindi with English math>",
    "example": "<Example as plain string in Hindi with English math>",
    "solved_questions": [
    {{"question": "<Problem 1 in Hindi with English math>", "solution": "<Solution 1 as plain string in Hindi with English math>"}},
    {{"question": "<Problem 2 in Hindi with English math>", "solution": "<Solution 2 as plain string in Hindi with English math>"}}
    // Add more solved questions here
    ]
}}"""
    refresher_prompt+=refresher_prompt + "identified concept : "+ identified_concept

    print("refresher prompt : \n",refresher_prompt)

    response = call_gemini(refresher_prompt)


    concept_desc = response.text # This is the AI response object from llm.invoke
    raw_ai_content = concept_desc.strip() # Get the raw text content and strip whitespace

    if not raw_ai_content:
        logging.error("Refresher AI returned empty content.")
        return jsonify({"error": "AI returned empty refresher content. Cannot display."}), 500 # Return error if empty

    logging.info(f"Raw AI Refresher Content: {raw_ai_content}") # Log raw content for debugging

    # --- Attempt to extract JSON from markdown code block ---
    match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_ai_content) # Non-greedy match for content inside ```json ```

    json_content_to_parse = ""
    if match:
        json_content_to_parse = match.group(1).strip() # Extract content from the first capturing group
        logging.info("Extracted JSON from markdown block.")
    else:
        # If no markdown block found, assume the entire content MIGHT be JSON
        json_content_to_parse = raw_ai_content
        logging.warning("No ```json ``` block found. Attempting to parse raw content as JSON.")

    if not json_content_to_parse:
        logging.error("Extracted or raw content to parse is empty.")
        return jsonify({"error": "AI response was not in expected JSON format."}), 500

    try:
        resp = json.loads(json_content_to_parse) # <-- NOW parse the EXTRACTED content
        concept_name = resp.get("concept_name", "Concept Details") # Use .get() with default to prevent KeyError
        concept_description = resp.get("concept_description", "Description not available.")
        example = resp.get("example", "Example not available.")
        # Ensure solved_questions is a list, even if AI returns something else or None
        solved_questions_raw = resp.get("solved_questions", [])
        solved_questions = solved_questions_raw if isinstance(solved_questions_raw, list) else []

        logging.info(f"Parsed concept output: {resp}")

        # Return the parsed data
        return jsonify({
            "concept_name": concept_name,
            "concept_description": concept_description,
            "example": example,
            "solved_questions": solved_questions
        })

    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error in /refresher after extraction attempt: {e}")
        logging.error(f"Attempted to parse: {json_content_to_parse}")
        # Include the raw AI content in the error response for better debugging on the frontend
        return jsonify({"error": f"Failed to parse AI response as JSON: {e}. Raw content received: {raw_ai_content[:500]}..."}), 500 # Return specific error
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON processing: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred processing AI response: {type(e).__name__}"}), 500


@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    if db is None:
        print("Database not initialized.")
        return jsonify({"error": "Database service unavailable."}), 500 # Internal Server Error

    try:
        # Get data from the incoming JSON request
        data = request.json
        rating = data.get('rating')
        email = data.get('email')
        feedback = data.get('feedback')

        # Basic Server-Side Validation
        if rating is None or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({"error": "Invalid or missing rating."}), 400 # Bad Request
        if email is None or not isinstance(email, str) or "@" not in email: # More robust email validation possible
            return jsonify({"error": "Invalid or missing email address."}), 400 # Bad Request
        # Feedback is optional, no validation needed beyond type check
        if feedback is not None and not isinstance(feedback, str):
             return jsonify({"error": "Invalid feedback format."}), 400


        # Data structure to store in Firestore
        feedback_entry = {
            'rating': rating,
            'email': email,
            'feedback': feedback if feedback is not None else '', # Store as empty string if optional field is missing
            'timestamp': datetime.datetime.utcnow() # Add a server-side timestamp
        }

        # Get a reference to the 'feedback' collection and add a new document
        # Firestore automatically generates a unique ID for the document
        doc_ref = db.collection('feedback').add(feedback_entry)

        print(f"Feedback successfully written to Firestore. Document ID: {doc_ref[1].id}")

        # Return a success response
        return jsonify({"message": "Feedback submitted successfully!", "id": doc_ref[1].id}), 200 # OK

    except Exception as e:
        print(f"Error submitting feedback: {e}")
        # Log the error properly in a real application
        return jsonify({"error": "An error occurred while saving feedback."}), 500 # Internal Server 


@app.route('/')
def index():
    return "Hello World"


# --- Run the Flask App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) # Use PORT env var or default to 8080
   
    app.run(host='0.0.0.0', port=port, debug=True)
