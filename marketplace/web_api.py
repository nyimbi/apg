#!/usr/bin/env python3
"""
APG Capability Marketplace Web API
==================================

FastAPI-based web interface for the capability marketplace.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import uuid

def uuid7str() -> str:
	"""Generate a UUID7-style string using uuid4 as fallback"""
	return str(uuid.uuid4())

from capability_marketplace import (
	CapabilityMarketplace,
	MarketplaceCapability,
	CapabilityCategory,
	CapabilityStatus,
	LicenseType,
	CapabilityRating,
	CapabilityDependency,
	CapabilityVersion
)

# Pydantic models for API
class CapabilityCreateRequest(BaseModel):
	"""Request model for creating a new capability"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	name: str = Field(..., min_length=3, max_length=100)
	display_name: str = Field(..., min_length=3, max_length=100)
	description: str = Field(..., min_length=50, max_length=500)
	detailed_description: str = Field(default="", max_length=5000)
	
	category: CapabilityCategory = CapabilityCategory.CUSTOM
	tags: List[str] = Field(default_factory=list, max_items=10)
	keywords: List[str] = Field(default_factory=list, max_items=15)
	
	author: str = Field(..., min_length=2, max_length=100)
	author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
	organization: str = Field(default="", max_length=100)
	license: LicenseType = LicenseType.MIT
	homepage: str = Field(default="", max_length=500)
	repository: str = Field(default="", max_length=500)
	
	capability_code: str = Field(..., min_length=100)
	example_usage: str = Field(default="", max_length=2000)
	documentation: str = Field(default="", max_length=10000)
	
	dependencies: List[Dict[str, Any]] = Field(default_factory=list, max_items=50)
	platforms: List[str] = Field(default_factory=lambda: ["linux", "windows", "macos"])

class CapabilityUpdateRequest(BaseModel):
	"""Request model for updating a capability"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	display_name: Optional[str] = Field(None, min_length=3, max_length=100)
	description: Optional[str] = Field(None, min_length=50, max_length=500)
	detailed_description: Optional[str] = Field(None, max_length=5000)
	
	tags: Optional[List[str]] = Field(None, max_items=10)
	keywords: Optional[List[str]] = Field(None, max_items=15)
	
	homepage: Optional[str] = Field(None, max_length=500)
	repository: Optional[str] = Field(None, max_length=500)
	
	capability_code: Optional[str] = Field(None, min_length=100)
	example_usage: Optional[str] = Field(None, max_length=2000)
	documentation: Optional[str] = Field(None, max_length=10000)

class CapabilityResponse(BaseModel):
	"""Response model for capability data"""
	model_config = ConfigDict(extra='forbid')
	
	id: str
	name: str
	display_name: str
	description: str
	detailed_description: str
	category: str
	tags: List[str]
	keywords: List[str]
	author: str
	organization: str
	license: str
	homepage: str
	repository: str
	current_version: str
	status: str
	featured: bool
	verified: bool
	premium: bool
	price: float
	average_rating: float
	rating_count: int
	download_count: int
	created_at: str
	updated_at: str

class RatingCreateRequest(BaseModel):
	"""Request model for creating a rating"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	rating: int = Field(..., ge=1, le=5)
	review: str = Field(default="", max_length=1000)
	user_id: str = Field(..., min_length=1)

class SearchRequest(BaseModel):
	"""Request model for capability search"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	query: str = Field(default="", max_length=200)
	category: Optional[CapabilityCategory] = None
	tags: List[str] = Field(default_factory=list, max_items=5)
	min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
	max_results: int = Field(default=50, ge=1, le=100)

class RecommendationRequest(BaseModel):
	"""Request model for capability recommendations"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	based_on_capability: Optional[str] = None
	user_history: List[str] = Field(default_factory=list, max_items=100)
	project_context: Dict[str, Any] = Field(default_factory=dict)
	limit: int = Field(default=10, ge=1, le=50)

# FastAPI app setup
app = FastAPI(
	title="APG Capability Marketplace API",
	description="Community-driven marketplace for APG capabilities",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Global marketplace instance
marketplace: Optional[CapabilityMarketplace] = None

async def get_marketplace() -> CapabilityMarketplace:
	"""Dependency to get marketplace instance"""
	global marketplace
	if marketplace is None:
		marketplace = CapabilityMarketplace()
		await asyncio.sleep(0.1)  # Allow initialization
	return marketplace

def capability_to_response(capability: MarketplaceCapability) -> CapabilityResponse:
	"""Convert marketplace capability to API response model"""
	return CapabilityResponse(
		id=capability.id,
		name=capability.name,
		display_name=capability.display_name,
		description=capability.description,
		detailed_description=capability.detailed_description,
		category=capability.category.value,
		tags=capability.tags,
		keywords=capability.keywords,
		author=capability.author,
		organization=capability.organization,
		license=capability.license.value,
		homepage=capability.homepage,
		repository=capability.repository,
		current_version=capability.current_version,
		status=capability.status.value,
		featured=capability.featured,
		verified=capability.verified,
		premium=capability.premium,
		price=capability.price,
		average_rating=capability.metrics.average_rating,
		rating_count=capability.metrics.rating_count,
		download_count=capability.metrics.download_count,
		created_at=capability.created_at.isoformat(),
		updated_at=capability.updated_at.isoformat()
	)

# API Endpoints

@app.get("/")
async def root():
	"""Root endpoint with API information"""
	return {
		"name": "APG Capability Marketplace API",
		"version": "1.0.0",
		"description": "Community-driven marketplace for APG capabilities",
		"endpoints": {
			"docs": "/docs",
			"capabilities": "/capabilities",
			"search": "/search",
			"recommendations": "/recommendations",
			"stats": "/stats"
		}
	}

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	marketplace_instance = await get_marketplace()
	return {
		"status": "healthy",
		"timestamp": datetime.utcnow().isoformat(),
		"capabilities_count": len(marketplace_instance.capabilities)
	}

# Capability Management

@app.post("/capabilities", response_model=Dict[str, Any])
async def create_capability(
	request: CapabilityCreateRequest,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Create a new capability"""
	try:
		# Convert dependencies
		dependencies = []
		for dep_data in request.dependencies:
			dependencies.append(CapabilityDependency(
				name=dep_data.get("name", ""),
				version_constraint=dep_data.get("version_constraint", "*"),
				optional=dep_data.get("optional", False),
				description=dep_data.get("description", "")
			))
		
		# Create capability
		capability = MarketplaceCapability(
			name=request.name,
			display_name=request.display_name,
			description=request.description,
			detailed_description=request.detailed_description,
			category=request.category,
			tags=request.tags,
			keywords=request.keywords,
			author=request.author,
			author_email=request.author_email,
			organization=request.organization,
			license=request.license,
			homepage=request.homepage,
			repository=request.repository,
			capability_code=request.capability_code,
			example_usage=request.example_usage,
			documentation=request.documentation,
			dependencies=dependencies,
			platforms=request.platforms
		)
		
		# Submit to marketplace
		result = await marketplace_instance.submit_capability(capability)
		
		if result['success']:
			return {
				"success": True,
				"capability_id": result['capability_id'],
				"message": "Capability created successfully",
				"validation_score": result['validation_results']['score']
			}
		else:
			raise HTTPException(
				status_code=400,
				detail={
					"message": "Capability validation failed",
					"errors": result['errors'],
					"validation_results": result['validation_results']
				}
			)
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/capabilities", response_model=List[CapabilityResponse])
async def list_capabilities(
	status: Optional[str] = Query(None, description="Filter by status"),
	category: Optional[str] = Query(None, description="Filter by category"),
	author: Optional[str] = Query(None, description="Filter by author"),
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""List capabilities with optional filters"""
	try:
		# Convert string parameters to enums
		status_filter = None
		if status:
			try:
				status_filter = CapabilityStatus(status)
			except ValueError:
				raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
		
		category_filter = None
		if category:
			try:
				category_filter = CapabilityCategory(category)
			except ValueError:
				raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
		
		capabilities = await marketplace_instance.list_capabilities(
			status=status_filter,
			category=category_filter,
			author=author
		)
		
		return [capability_to_response(cap) for cap in capabilities]
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/capabilities/{capability_id}", response_model=CapabilityResponse)
async def get_capability(
	capability_id: str,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Get a specific capability by ID"""
	try:
		capability = await marketplace_instance.get_capability(capability_id)
		if not capability:
			raise HTTPException(status_code=404, detail="Capability not found")
		
		return capability_to_response(capability)
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.put("/capabilities/{capability_id}", response_model=Dict[str, Any])
async def update_capability(
	capability_id: str,
	request: CapabilityUpdateRequest,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Update an existing capability"""
	try:
		capability = await marketplace_instance.get_capability(capability_id)
		if not capability:
			raise HTTPException(status_code=404, detail="Capability not found")
		
		# Update fields if provided
		if request.display_name is not None:
			capability.display_name = request.display_name
		if request.description is not None:
			capability.description = request.description
		if request.detailed_description is not None:
			capability.detailed_description = request.detailed_description
		if request.tags is not None:
			capability.tags = request.tags
		if request.keywords is not None:
			capability.keywords = request.keywords
		if request.homepage is not None:
			capability.homepage = request.homepage
		if request.repository is not None:
			capability.repository = request.repository
		if request.capability_code is not None:
			capability.capability_code = request.capability_code
		if request.example_usage is not None:
			capability.example_usage = request.example_usage
		if request.documentation is not None:
			capability.documentation = request.documentation
		
		# Update timestamp
		capability.updated_at = datetime.utcnow()
		
		# Re-validate
		validation_results = await marketplace_instance.validator.validate_capability(capability)
		
		if not validation_results['valid']:
			raise HTTPException(
				status_code=400,
				detail={
					"message": "Updated capability validation failed",
					"errors": validation_results['errors']
				}
			)
		
		# Save changes
		await marketplace_instance._save_capabilities()
		
		return {
			"success": True,
			"message": "Capability updated successfully",
			"validation_score": validation_results['score']
		}
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/capabilities/{capability_id}/publish", response_model=Dict[str, Any])
async def publish_capability(
	capability_id: str,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Publish a capability to make it publicly available"""
	try:
		success = await marketplace_instance.publish_capability(capability_id)
		if not success:
			raise HTTPException(
				status_code=400,
				detail="Failed to publish capability. Check that capability exists and passes validation."
			)
		
		return {
			"success": True,
			"message": "Capability published successfully"
		}
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/capabilities/{capability_id}/download", response_model=Dict[str, Any])
async def download_capability(
	capability_id: str,
	user_id: str = Query(..., description="User ID for download tracking"),
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Download a capability package"""
	try:
		package = await marketplace_instance.download_capability(capability_id, user_id)
		if not package:
			raise HTTPException(
				status_code=404,
				detail="Capability not found or not available for download"
			)
		
		# Return capability package without the full capability object
		return {
			"success": True,
			"capability_id": capability_id,
			"code": package['code'],
			"documentation": package['documentation'],
			"example_usage": package['example_usage'],
			"test_cases": package['test_cases'],
			"dependencies": [
				{
					"name": dep.name,
					"version_constraint": dep.version_constraint,
					"optional": dep.optional,
					"description": dep.description
				}
				for dep in package['dependencies']
			]
		}
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Search and Discovery

@app.post("/search", response_model=List[CapabilityResponse])
async def search_capabilities(
	request: SearchRequest,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Search capabilities with intelligent ranking"""
	try:
		capabilities = await marketplace_instance.search_capabilities(
			query=request.query,
			category=request.category,
			tags=request.tags,
			min_rating=request.min_rating,
			max_results=request.max_results
		)
		
		return [capability_to_response(cap) for cap in capabilities]
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recommendations", response_model=List[CapabilityResponse])
async def get_recommendations(
	request: RecommendationRequest,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Get personalized capability recommendations"""
	try:
		capabilities = await marketplace_instance.get_recommendations(
			based_on_capability=request.based_on_capability,
			user_history=request.user_history,
			project_context=request.project_context,
			limit=request.limit
		)
		
		return [capability_to_response(cap) for cap in capabilities]
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Ratings and Reviews

@app.post("/capabilities/{capability_id}/ratings", response_model=Dict[str, Any])
async def add_rating(
	capability_id: str,
	request: RatingCreateRequest,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Add a rating for a capability"""
	try:
		rating = CapabilityRating(
			user_id=request.user_id,
			capability_id=capability_id,
			rating=request.rating,
			review=request.review
		)
		
		success = await marketplace_instance.add_rating(rating)
		if not success:
			raise HTTPException(status_code=404, detail="Capability not found")
		
		return {
			"success": True,
			"message": "Rating added successfully",
			"rating_id": rating.id
		}
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/capabilities/{capability_id}/ratings", response_model=List[Dict[str, Any]])
async def get_capability_ratings(
	capability_id: str,
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Get ratings for a capability"""
	try:
		capability = await marketplace_instance.get_capability(capability_id)
		if not capability:
			raise HTTPException(status_code=404, detail="Capability not found")
		
		return [
			{
				"id": rating.id,
				"user_id": rating.user_id,
				"rating": rating.rating,
				"review": rating.review,
				"helpful_votes": rating.helpful_votes,
				"created_at": rating.created_at.isoformat(),
				"verified_purchase": rating.verified_purchase
			}
			for rating in capability.ratings
		]
	
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Analytics and Statistics

@app.get("/stats", response_model=Dict[str, Any])
async def get_marketplace_stats(
	marketplace_instance: CapabilityMarketplace = Depends(get_marketplace)
):
	"""Get marketplace statistics"""
	try:
		return marketplace_instance.get_marketplace_stats()
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/categories", response_model=List[Dict[str, str]])
async def get_categories():
	"""Get available capability categories"""
	return [
		{
			"value": category.value,
			"name": category.name.replace('_', ' ').title()
		}
		for category in CapabilityCategory
	]

@app.get("/licenses", response_model=List[Dict[str, str]])
async def get_licenses():
	"""Get available license types"""
	return [
		{
			"value": license_type.value,
			"name": license_type.name.replace('_', ' ').title()
		}
		for license_type in LicenseType
	]

# Error handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
	return JSONResponse(
		status_code=404,
		content={"error": "Not found", "detail": "The requested resource was not found"}
	)

@app.exception_handler(500)
async def internal_error_handler(request, exc):
	return JSONResponse(
		status_code=500,
		content={"error": "Internal server error", "detail": "An unexpected error occurred"}
	)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)