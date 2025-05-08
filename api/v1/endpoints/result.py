from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import json
import os

from models.resume_schema import OptimizationRequest, OptimizationResponse
from services.cv_optimizer import optimize_resume

router = APIRouter()
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("temp_files", exist_ok=True)


@router.post("/optimize", response_model=OptimizationResponse)
async def get_result(request: OptimizationRequest):
    """
    Optimize a resume based on a job offer.
    
    Args:
        request: OptimizationRequest containing resume and job offer data
        
    Returns:
        OptimizationResponse: Optimized resume and recommendations
    """
    try:
        # Log inputs for debugging
        request_id = abs(hash(str(request.resume) + str(request.job_offer)))
        debug_file = f"temp_files/optimization_request_{request_id}.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(request.model_dump(), f, ensure_ascii=False, indent=4)
        
        # Process the optimization
        optimization_result = optimize_resume(request.resume, request.job_offer)
        
        # Log outputs for debugging
        result_file = f"temp_files/optimization_result_{request_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(optimization_result.model_dump(), f, ensure_ascii=False, indent=4)
        
        return optimization_result
    
    except Exception as e:
        logger.error(f"Failed to optimize resume: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")