# AI Resume Enhancer

A web application that enhances resumes using the Google Gemini API, tailoring them to a specific job description. It supports PDF, DOCX, DOC, and TXT resume formats.

## Features

*   Upload resumes in PDF, DOCX, DOC, and TXT formats.
*   Enhance resumes based on a provided job description using AI.
*   Receive an ATS (Applicant Tracking System) score for your resume against the job description.
*   Get detailed feedback on your resume's content and formatting.
*   Download the enhanced resume.
*   Copy the enhanced resume to the clipboard.

## Technologies Used

*   **Backend**: FastAPI (Python)
*   **Frontend**: React (JavaScript), Three.js, postprocessing, jspdf
*   **AI**: Google Gemini API
*   **Other Python Libraries**: `uvicorn`, `PyMuPDF`, `python-docx`, `python-dotenv`, `aiofiles`, `python-multipart`, `pytest`, `pytest-asyncio`, `httpx`

## API Endpoints

The backend provides the following endpoints:

*   `POST /resume/upload_resume`: Upload a resume file.
*   `POST /resume/enhance_resume`: Enhance a resume using either a newly uploaded file or a previously uploaded one.
*   `GET /resume/download/{filename}`: Download an enhanced resume.
*   `GET /health`: Health check endpoint.

## Setup and Installation (Local)

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd ai-resume-enhancer
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create a `.env` file**:
    In the root directory of your project, create a file named `.env` and add your Google Gemini API key:
    ```
    GEMINI_API_KEY=YOUR_API_KEY
    ```

6.  **Run the application**:
    ```bash
    uvicorn main:app --reload
    ```
    The application will be accessible at `http://127.0.0.1:8000`.

## Deployment (Render)

This project is configured for easy deployment on Render using a `render.yaml` blueprint.

1.  **Prepare your repository**: Ensure your project is pushed to a Git repository (GitHub, GitLab, or Bitbucket).

2.  **Create a new service on Render**: Go to the [Render dashboard](https://dashboard.render.com/) and create a new **Blueprint** service. Connect your Git repository.

3.  **Environment Variables**: Set the `GEMINI_API_KEY` as a secret environment variable in the Render dashboard for your service. This is crucial for the application to function.

4.  **Automatic Deployment**: Render will automatically detect and use your `render.yaml` file to build and deploy your application.

## Usage

1.  Open the web application in your browser (locally at `http://127.0.0.1:8000` or your deployed Render URL).
2.  Upload your resume file (PDF, DOCX, DOC, or TXT).
3.  Paste the job description into the provided text area.
4.  Click the "Enhance Resume" button to get an AI-optimized version of your resume.
5.  The enhanced resume will be displayed on the page, along with options to download it or copy its content to the clipboard.

## Testing

To run the unit and integration tests for the project, navigate to the root directory and execute:

```bash
python -m pytest
```
