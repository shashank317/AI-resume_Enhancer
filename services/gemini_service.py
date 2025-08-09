# services/gemini_service.py
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME
import logging

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are an expert resume writer and technical recruiter. "
    "Given the candidate's resume and the job description, do the following:\n"
    "1. Remove all existing resume content that is unrelated to the job description.\n"
    "2. After removing unrelated content, add the skills, keywords, and requirements explicitly asked for in the job description.\n"
    "3. Review each project/experience:\n"
    "   - If a project aligns well with the job description, keep it as is.\n"
    "   - If a project does not fit, suggest exactly one example project that would be a better match.\n"
    "4. Keep all information truthful. Do NOT invent job experience, certifications, or dates.\n"
    "5. Output the improved resume in plain text with clear sections: Contact (if present), Summary, Skills, Experience (bulleted), Education, Certifications.\n"
    "6. After the resume, add a short 2-3 sentence interview pitch tailored for this job.\n"
    "Output ONLY the resume and the pitch separated by a line '----'."
)

def call_gemini_optimize_resume(resume_text: str, job_description: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """
    Calls the Gemini API to produce an enhanced resume.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        user_content = (
            f"Resume:\n<<RESUME>>\n{resume_text}\n<<END RESUME>>\n\n"
            f"Job Description:\n<<JOB>>\n{job_description}\n<<END JOB>>\n\n"
            "Please strictly follow these instructions:\n"
            "- Remove all unrelated content from the resume that does not align with the job description.\n"
            "- After removal, add or emphasize the skills and requirements explicitly mentioned in the job description.\n"
            "- For projects/experience, keep the ones that fit the job. If any project doesn't fit, suggest exactly one example project that fits better.\n"
            "- Do NOT invent any job experience, dates, or certifications.\n"
            "- Output ONLY the cleaned and enhanced resume and a brief interview pitch separated by '----'."
        )

        response = model.generate_content(
            user_content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )

        return response.text.strip()
    except Exception as e:
        logging.exception("Gemini call failed")
        raise
