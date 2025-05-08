import logging
from typing import List, Dict, Any, Set
import re
import json

from models.resume_schema import Resume, JobOffer, OptimizationResponse

logger = logging.getLogger(__name__)

def optimize_resume(resume: Resume, job_offer: JobOffer) -> OptimizationResponse:
    """
    Optimize a resume based on a job offer.
    
    Args:
        resume: The original resume
        job_offer: The job offer to optimize for
        
    Returns:
        OptimizationResponse: Optimized resume and recommendations
    """
    try:
        # Try using LangChain if available
        return llm_optimize_resume(resume, job_offer)
    except ImportError:
        # Fall back to rule-based optimization
        logger.info("LangChain not available, using rule-based optimization")
        return rule_based_optimize_resume(resume, job_offer)


def llm_optimize_resume(resume: Resume, job_offer: JobOffer) -> OptimizationResponse:
    """
    Use LLM to optimize the resume.
    
    Args:
        resume: The original resume
        job_offer: The job offer to optimize for
        
    Returns:
        OptimizationResponse: Optimized resume and recommendations
    """
    try:
        from langchain_ollama.llms import OllamaLLM
        from langchain_core.prompts import ChatPromptTemplate
        
        # Convert resume and job offer to readable format for the LLM
        resume_json = json.dumps(resume.model_dump(), indent=2)
        job_offer_json = json.dumps(job_offer.model_dump(), indent=2)
        
        # Create a prompt
        template = """
        You are a professional resume optimizer. Your task is to optimize a resume for a specific job offer.
        
        # Resume
        ```json
        {resume_json}
        ```
        
        # Job Offer
        ```json
        {job_offer_json}
        ```
        
        Please optimize the resume for this job offer. Focus on:
        1. Highlighting relevant skills and experiences
        2. Adjusting the summary to match the job requirements
        3. Rephrasing experience descriptions to better match the job
        4. Reordering skills to prioritize those mentioned in the job description
        
        Return your answer as a JSON object with the following structure:
        {{
            "optimized_resume": {{...}}, // The optimized resume in the same structure as the original
            "recommendations": [...],    // List of recommendations for further improvement
            "matching_score": 0.0,       // A score from 0.0 to 1.0 indicating how well the resume matches the job offer
            "skills_to_highlight": [...],// List of skills in the resume that match the job requirements
            "skills_to_acquire": [...]   // List of skills mentioned in the job description that are missing in the resume
        }}
        
        IMPORTANT: Maintain the exact same structure of the original resume. Don't add or remove fields, just optimize the content.
        """
        
        # Prepare the prompt
        prompt = ChatPromptTemplate.from_template(template)
        prompt_value = prompt.format_messages(
            resume_json=resume_json,
            job_offer_json=job_offer_json
        )
        
        # Use Ollama for processing
        model = OllamaLLM(
            model="mistral:latest",
            temperature=0.2,
            system="You are a precision resume optimization system that outputs only valid, well-structured JSON."
        )
        
        # Get the model's response
        raw_response = model.invoke(prompt_value)
        
        # Save the raw response for debugging
        with open("temp_files/optimizer_raw_response.txt", "w", encoding="utf-8") as f:
            f.write(raw_response)
        
        # Extract JSON from the response
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}")
        if json_start >= 0 and json_end >= 0:
            json_str = raw_response[json_start:json_end+1]
            result = json.loads(json_str)
            return OptimizationResponse.model_validate(result)
        else:
            # If JSON extraction fails, fall back to rule-based optimization
            logger.warning("Failed to extract JSON from LLM response, falling back to rule-based optimization")
            return rule_based_optimize_resume(resume, job_offer)
        
    except Exception as e:
        logger.error(f"LLM optimization failed: {e}")
        return rule_based_optimize_resume(resume, job_offer)


def rule_based_optimize_resume(resume: Resume, job_offer: JobOffer) -> OptimizationResponse:
    """
    Apply rule-based optimization to the resume based on job offer.
    
    Args:
        resume: The original resume
        job_offer: The job offer to optimize for
        
    Returns:
        OptimizationResponse: Optimized resume and recommendations
    """
    # Clone the resume to avoid modifying the original
    optimized_resume = Resume.model_validate(resume.model_dump())
    recommendations = []
    
    # Extract keywords from job offer
    job_keywords = extract_keywords(job_offer)
    resume_keywords = extract_keywords_from_resume(resume)
    
    # Calculate matching skills and missing skills
    matching_skills = resume_keywords.intersection(job_keywords)
    missing_skills = job_keywords - resume_keywords
    
    # Calculate a matching score (simple version)
    if len(job_keywords) > 0:
        matching_score = len(matching_skills) / len(job_keywords)
    else:
        matching_score = 0.0
    
    # Optimize summary
    if optimized_resume.summary:
        optimized_resume.summary = optimize_summary(optimized_resume.summary, job_offer, matching_skills)
    else:
        new_summary = generate_summary(optimized_resume, job_offer)
        optimized_resume.summary = new_summary
        recommendations.append("Added a professional summary tailored to the job")
    
    # Optimize skills - reorder to prioritize matching skills
    if optimized_resume.skills and optimized_resume.skills.technical_skills:
        tech_skills = optimized_resume.skills.technical_skills
        # Move matching skills to the front
        reordered_skills = [s for s in tech_skills if any(k.lower() in s.lower() for k in matching_skills)]
        reordered_skills += [s for s in tech_skills if not any(k.lower() in s.lower() for k in matching_skills)]
        optimized_resume.skills.technical_skills = reordered_skills
        recommendations.append("Reordered skills to highlight those relevant to the job")
    
    # Optimize experience descriptions
    if optimized_resume.experience:
        for i, exp in enumerate(optimized_resume.experience):
            if exp.description:
                optimized_resume.experience[i].description = optimize_experience_description(
                    exp.description, job_offer, matching_skills
                )
        recommendations.append("Enhanced experience descriptions to align with job requirements")
    
    # Generate additional recommendations
    if len(missing_skills) > 0:
        skills_str = ", ".join(list(missing_skills)[:5])
        if len(missing_skills) > 5:
            skills_str += f", and {len(missing_skills) - 5} more"
        recommendations.append(f"Consider acquiring these skills mentioned in the job description: {skills_str}")
    
    if not optimized_resume.skills.soft_skills:
        recommendations.append("Add soft skills to your resume as they are often valued by employers")
    
    return OptimizationResponse(
        optimized_resume=optimized_resume,
        recommendations=recommendations,
        matching_score=round(matching_score, 2),
        skills_to_highlight=list(matching_skills)[:10],  # Limit to top 10
        skills_to_acquire=list(missing_skills)[:10]      # Limit to top 10
    )


def extract_keywords(job_offer: JobOffer) -> Set[str]:
    """
    Extract keywords from job offer.
    
    Args:
        job_offer: The job offer
        
    Returns:
        Set[str]: Set of keywords
    """
    # Combine all text fields
    text = f"{job_offer.title} {job_offer.description}"
    if job_offer.requirements:
        text += " " + " ".join(job_offer.requirements)
    if job_offer.preferences:
        text += " " + " ".join(job_offer.preferences)
    
    # Extract potential keywords (simple version)
    text = text.lower()
    
    # Common technical skills to look for
    tech_terms = [
        "python", "java", "javascript", "js", "typescript", "ts", "c\\+\\+", "c#", "ruby", "php", "go", "rust",
        "react", "angular", "vue", "node", "django", "flask", "spring", "express", "tensorflow", "pytorch",
        "ai", "ml", "machine learning", "deep learning", "nlp", "data science", "analytics", "statistics",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis", "elasticsearch",
        "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "jenkins", "cicd", "devops",
        "agile", "scrum", "kanban", "jira", "git", "github", "gitlab"
    ]
    
    # Compile a regex pattern to find these terms
    pattern = r'\b(' + '|'.join(tech_terms) + r')\b'
    tech_matches = set(re.findall(pattern, text))
    
    # Also look for phrases with "experience with/in" or "knowledge of"
    skill_phrases = re.findall(r'experience (?:with|in|of) ([\w\s,]+)', text)
    for phrase in skill_phrases:
        skills = [s.strip() for s in re.split(r'[,]', phrase)]
        tech_matches.update(skills)
    
    return tech_matches


def extract_keywords_from_resume(resume: Resume) -> Set[str]:
    """
    Extract keywords from resume.
    
    Args:
        resume: The resume
        
    Returns:
        Set[str]: Set of keywords
    """
    keywords = set()
    
    # Extract from skills
    if resume.skills:
        if resume.skills.technical_skills:
            for skill in resume.skills.technical_skills:
                keywords.add(skill.lower())
        if resume.skills.soft_skills:
            for skill in resume.skills.soft_skills:
                keywords.add(skill.lower())
    
    # Extract from experience descriptions
    if resume.experience:
        for exp in resume.experience:
            if exp.description:
                # Simple keyword extraction
                words = re.findall(r'\b\w+\b', exp.description.lower())
                keywords.update(words)
    
    return keywords


def optimize_summary(summary: str, job_offer: JobOffer, matching_skills: Set[str]) -> str:
    """
    Optimize the summary to better match the job offer.
    
    Args:
        summary: Original summary
        job_offer: The job offer
        matching_skills: Matching skills between resume and job offer
        
    Returns:
        str: Optimized summary
    """
    # Simple rule-based optimization - mention job title and top skills
    job_title = job_offer.title.lower()
    
    # Check if the job title is already mentioned
    if job_title not in summary.lower():
        # Try to add job title to the summary
        if summary.strip().endswith('.'):
            summary = summary[:-1]
        
        # Add a sentence mentioning the job and skills
        if matching_skills:
            top_skills = list(matching_skills)[:3]  # Top 3 matching skills
            skills_str = ", ".join(top_skills[:-1]) + (" and " + top_skills[-1] if len(top_skills) > 1 else top_skills[0])
            summary += f" Seeking a {job_offer.title} position where I can utilize my expertise in {skills_str}."
        else:
            summary += f" Seeking a {job_offer.title} position to contribute to organizational success."
    
    return summary


def generate_summary(resume: Resume, job_offer: JobOffer) -> str:
    """
    Generate a summary for the resume based on the job offer.
    
    Args:
        resume: The resume
        job_offer: The job offer
        
    Returns:
        str: Generated summary
    """
    # Get years of experience
    years_exp = calculate_years_experience(resume)
    
    # Get area of expertise
    expertise = ""
    if resume.skills and resume.skills.technical_skills:
        top_skills = resume.skills.technical_skills[:3]
        expertise = ", ".join(top_skills)
    
    # Generate a basic summary
    name = "Professional" if not resume.personal_info or not resume.personal_info.name else resume.personal_info.name.split()[0]
    summary = f"Experienced {job_offer.title} with {years_exp}+ years of experience in {expertise}. "
    summary += f"Seeking to leverage my skills in a {job_offer.title} role at {job_offer.company}."
    
    return summary


def calculate_years_experience(resume: Resume) -> int:
    """
    Calculate total years of experience from the resume.
    
    Args:
        resume: The resume
        
    Returns:
        int: Years of experience
    """
    total_years = 0
    current_year = 2025  # Using a hardcoded current year since we don't need precise calculations
    
    if resume.experience:
        for exp in resume.experience:
            start_year = None
            end_year = None
            
            # Extract years from dates
            if exp.start_date:
                match = re.search(r'(\d{4})', exp.start_date)
                if match:
                    start_year = int(match.group(1))
            
            if exp.end_date:
                if exp.end_date.lower() == "present":
                    end_year = current_year
                else:
                    match = re.search(r'(\d{4})', exp.end_date)
                    if match:
                        end_year = int(match.group(1))
            
            # Calculate duration if both start and end are available
            if start_year and end_year:
                duration = end_year - start_year
                total_years += duration
    
    # Return at least 1 year of experience
    return max(1, total_years)


def optimize_experience_description(description: str, job_offer: JobOffer, matching_skills: Set[str]) -> str:
    """
    Optimize experience description to better match job requirements.
    
    Args:
        description: Original description
        job_offer: The job offer
        matching_skills: Matching skills between resume and job offer
        
    Returns:
        str: Optimized description
    """
    optimized = description
    
    # Add emphasis on matching skills if they appear in the description
    for skill in matching_skills:
        # Only process skills that are actual words (not single letters)
        if len(skill) > 1:
            # Use word boundaries to ensure we're matching whole words
            pattern = r'\b' + re.escape(skill) + r'\b'
            # Case-insensitive replacement while preserving original case
            optimized = re.sub(
                pattern, 
                lambda m: m.group(0),  # Preserve original case
                optimized, 
                flags=re.IGNORECASE
            )
    
    # If any job requirements contain "result" or "impact", try to add metrics if missing
    if any("result" in req.lower() or "impact" in req.lower() for req in job_offer.requirements):
        if not re.search(r'\d+%|increased|decreased|improved|reduced|by \d+', optimized.lower()):
            # If no metrics are found, add a generic one
            if not optimized.endswith('.'):
                optimized += '.'
            optimized += " Contributed to team efficiency and project success."
    
    return optimized