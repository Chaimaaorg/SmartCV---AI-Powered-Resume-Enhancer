from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator

# Define Pydantic models with examples and descriptions for better parsing
class PersonalInfo(BaseModel):
    name: str = Field(None, description="The person's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    address: Optional[str] = Field(None, description="Physical address")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    website: Optional[str] = Field(None, description="Personal website URL")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    nationality: Optional[str] = Field(None, description="Nationality")

class Skills(BaseModel):
    technical_skills: List[str] = Field(default_factory=list, description="List of skills")
    soft_skills: List[str] = Field(default_factory=list, description="List of soft skills")
        
class EducationItem(BaseModel):
    degree: str = Field(None, description="Degree obtained or in progress")
    field_of_study: str = Field(None, description="Field or major")
    institution: str = Field(None, description="Name of the institution")
    start_date: str = Field(None, description="Start date (YYYY-MM or YYYY)")
    end_date: str = Field(None, description="End date (YYYY-MM or YYYY or 'Present')")
    location: str = Field(None, description="Location of the institution")
    grade: Optional[str] = Field(None, description="Grade, GPA, or distinction")

class ExperienceItem(BaseModel):
    job_title: str = Field(None, description="Job title or position")
    company: str = Field(None, description="Company or organization name")
    start_date: str = Field(None, description="Start date (YYYY-MM or YYYY)")
    end_date: str = Field(None, description="End date (YYYY-MM or YYYY or 'Present')")
    location: str = Field(None, description="Location of the job")
    description: Optional[str] = Field(None, description="Description of responsibilities and achievements")
    description: Optional[Union[str, List[str]]] = Field(
            None,
            description="Description of responsibilities and achievements"
        )
    @field_validator('description', mode="before")
    def format_description(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return "\n".join(v)  
        return str(v)  
    
class CertificationItem(BaseModel):
    title: str = Field(None, description="Title of the certification")
    issuer: str = Field(None, description="Certification issuing organization")
    date: str = Field(None, description="Date obtained (YYYY-MM or YYYY)")
    url: Optional[str] = Field(None, description="URL of certification",alias=['url','link'])

class ProjectItem(BaseModel):
    title: str = Field(None, description="Project title")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used in the project")
    url: Optional[str] = Field(None, description="URL of the project",alias=["url"])

class PublicationItem(BaseModel):
    title: str = Field(None, description="Publication title")
    publisher: str = Field(None, description="Publisher name")
    date: str = Field(None, description="Publication date (YYYY-MM or YYYY)")
    url: Optional[str] = Field(None, description="URL to the publication")

class AwardItem(BaseModel):
    title: str = Field(None, description="Award title")
    issuer: str = Field(None, description="Award issuing organization")
    date: str = Field(None, description="Date received (YYYY-MM or YYYY)")
    description: Optional[str] = Field(None, description="Description of the award")

class ReferenceItem(BaseModel):
    name: str = Field(None, description="Reference's full name")
    position: Optional[str] = Field(None, description="Reference's position")
    company: Optional[str] = Field(None, description="Reference's company")
    email: Optional[str] = Field(None, description="Reference's email")
    phone: Optional[str] = Field(None, description="Reference's phone")

class Resume(BaseModel):
    personal_info: PersonalInfo = Field(None, description="Personal and contact information", alias='PersonalInfo')
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    skills: Optional[Union[Skills,Any]] = Field(default_factory=Skills, description="Skills categorized by type",alias=['skills'])
    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Professional certifications")
    projects: List[ProjectItem] = Field(default_factory=list, description="Projects")
    publications: List[PublicationItem] = Field(default_factory=list, description="Publications")
    awards: List[AwardItem] = Field(default_factory=list, description="Awards and recognitions")
    languages: List[str] = Field(default_factory=list, description="Languages the person can speak",alias=['languages'])    
    interests: List[str] = Field(default_factory=list, description="Personal interests or hobbies")
    references: List[ReferenceItem] = Field(default_factory=list, description="Professional references")
    
    @field_validator('skills', mode='before')
    def normalize_skills(cls, v: Any) -> Union[Skills, Any]:
        """Normalisation pendant la validation"""
        if v is None:
            return Skills()
        
        if isinstance(v, Skills):
            return v
        
        if isinstance(v, dict):
            tech_skills = []
            for skills in v.values():
                if isinstance(skills, list):
                    tech_skills.extend(skills)
                elif isinstance(skills, str):
                    tech_skills.append(skills)
            return Skills(technical_skills=tech_skills)
        
        if isinstance(v, (list, str)):
            return Skills(technical_skills=v if isinstance(v, list) else [s.strip() for s in v.split(",")])
        
        return v

class JobOffer(BaseModel):
    title: str
    company: str
    description: str
    requirements: List[str]
    preferences: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = None
    
    
class OptimizationRequest(BaseModel):
    resume: Resume
    job_offer: JobOffer
    
    
class OptimizationResponse(BaseModel):
    optimized_resume: Resume
    recommendations: List[str]
    matching_score: float
    skills_to_highlight: List[str]
    skills_to_acquire: List[str]