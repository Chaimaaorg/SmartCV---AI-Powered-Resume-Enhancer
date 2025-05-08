import json
import logging
from typing import Any, Dict
import os

from models.resume_schema import Resume
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json

logger = logging.getLogger(__name__)

def extract_resume_from_text(extracted_text: str) -> Resume:
    """
    Extract structured resume data from raw text using LLM.
    
    Args:
        extracted_text: Raw text extracted from CV
        
    Returns:
        Resume: Structured resume data
    """
    try:
        try:
            # Try using Ollama for local LLM processing
            from langchain_ollama.llms import OllamaLLM
            
            # Set up parser with the Pydantic model
            parser = PydanticOutputParser(pydantic_object=Resume)
            
            # Create a prompt with explicit formatting instructions
            template = """
            You are a CV/resume parsing expert with exceptional attention to detail. Your task is to extract structured information from the CV text below and format it according to the specified JSON schema.

            Follow these strict guidelines:

            1. Extract ALL information present in the CV - don't miss any details
            2. Format dates consistently as YYYY-MM or YYYY
            3. For ongoing positions, use "Present" as the end date
            4. Ensure all URLs are properly formatted
            5. Return ONLY valid JSON that strictly conforms to the schema
            6. Don't invent information that's not in the CV
            7. Include empty arrays [] for list fields that have no data
            8. Use null for optional fields that are not present

            {format_instructions}

            ## CV TEXT:

            {cv_text}

            OUTPUT: Respond with ONLY a valid JSON object according to the schema, nothing else.
            """

            # Create the prompt with the template
            prompt = ChatPromptTemplate.from_template(template)
            prompt_value = prompt.format_messages(
                cv_text=extracted_text,
                format_instructions=parser.get_format_instructions()
            )

            # Use a more specific system message and temperature for structured output
            model = OllamaLLM(
                model="mistral:latest",
                temperature=0.1,  # Lower temperature for more consistent outputs
                system="You are a precision data extraction system that outputs only valid, well-structured JSON."
            )

            # Get the model's response
            raw_response = model.invoke(prompt_value)
            with open("temp_files/raw_llm_response.txt", "w", encoding="utf-8") as f:
                f.write(raw_response)

            # Try to parse the response as JSON
            cleaned_response = raw_response
            if "`json" in raw_response:
                # Extract content between json code blocks
                start = raw_response.find("`json") + 7
                end = raw_response.find("`", start)
                if end != -1:
                    cleaned_response = raw_response[start:end].strip()
            elif "```" in raw_response:
                # Extract content between generic code blocks
                start = raw_response.find("```") + 3
                if "json" in raw_response[start:start+10]:
                    start += 4  # Skip "json" if present
                end = raw_response.find("```", start)
                if end != -1:
                    cleaned_response = raw_response[start:end].strip()
            
            # Try direct parsing
            try:
                parsed_response = parser.parse(cleaned_response)
                logger.info("✅ Successfully parsed using PydanticOutputParser")
            except Exception as e:
                # Fallback: Try to parse as raw JSON if the parser fails
                logger.warning(f"PydanticOutputParser failed: {e}")
                parsed_json = json.loads(cleaned_response)
                # Convert parsed JSON to Pydantic model
                parsed_response = Resume.model_validate(parsed_json)
                logger.info("✅ Successfully parsed using direct JSON parsing")
            
            # Save to JSON file for debugging
            with open("temp_files/parsed_resume.json", "w", encoding="utf-8") as f:
                json.dump(parsed_response.model_dump(), f, ensure_ascii=False, indent=4)
            
            logger.info("✅ Resume saved to parsed_resume.json")
            return parsed_response
            
        except ImportError:
            # If Ollama is not available, try a simpler approach
            logger.warning("Langchain/Ollama not available, using fallback parsing method")
            return fallback_resume_extraction(extracted_text)
            
    except Exception as e:
        logger.error(f"❌ Failed to parse resume: {e}")
        # Return a basic resume structure if parsing fails
        return Resume(
            personal_info={"name": "Extraction Failed"},
            summary=f"Failed to parse resume. Error: {str(e)}. Please try again with a clearer document."
        )


def fallback_resume_extraction(text: str) -> Resume:
    """
    A simple fallback method for extracting resume information when LLM is not available.
    This is a very basic implementation and won't be as accurate as LLM-based extraction.
    
    Args:
        text: Raw text extracted from CV
        
    Returns:
        Resume: Basic structured resume data
    """
    import re
    
    # Very basic extraction
    lines = text.split('\n')
    
    # Try to find a name (usually at the top)
    name = lines[0].strip() if lines and lines[0].strip() else "Unknown Name"
    
    # Try to find email with regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, text)
    email = email_matches[0] if email_matches else None
    
    # Try to find phone with regex
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, text)
    phone = phone_matches[0] if phone_matches else None
    
    # Extract some skills (very basic)
    skills = []
    skill_indicators = ["proficient in", "skilled in", "skills:", "technologies:", "languages:"]
    for line in lines:
        for indicator in skill_indicators:
            if indicator.lower() in line.lower():
                potential_skills = line.split(indicator.lower())[1].split(',')
                skills.extend([s.strip() for s in potential_skills if s.strip()])
    
    # Create basic resume
    resume = Resume(
        personal_info={
            "name": name,
            "email": email,
            "phone": phone
        },
        skills={"technical_skills": skills[:10]},  # Limit to first 10 found
        summary=f"This is a basic extraction of the resume. Full text contains {len(text)} characters."
    )
    
    return resume