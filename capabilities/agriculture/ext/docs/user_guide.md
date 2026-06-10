# Extension Services — User Guide

## Overview

agr_ext digitises the agricultural extension programme: logging advisor-farmer interactions
across all delivery channels, managing demonstration plots, tracking training events, and
maintaining a searchable knowledge base for farmers and extension workers.

## Key Use Cases

- **Advisory Delivery**: Log every advisory interaction (field visit, SMS, voice, group meeting)
  with topic, message, and follow-up flags. Track pending follow-ups per worker.
- **Demo Plots**: Register demonstration plots for specific crops and topics; track farmer
  visits and outcomes.
- **Training Records**: Schedule and record training events with participant lists and
  actual attendance.
- **Knowledge Base**: Build a multilingual, searchable library of agronomic articles
  tagged by crop type and category.

## Example Workflows

### Log a Field Advisory
```
POST /api/agriculture/ext/advisories
{
  "farmer_id": "farmer-001",
  "extension_worker_id": "worker-001",
  "channel": "field_visit",
  "topic": "Fall Armyworm management",
  "message": "Apply Coragen 20SC at 150ml/ha at first sign of infestation",
  "crop_type": "maize",
  "follow_up_required": true
}
```

### Create a Demo Plot
```
POST /api/agriculture/ext/demo-plots
{
  "name": "Conservation Ag Demo - Nakuru",
  "farm_parcel_id": "par-001",
  "extension_worker_id": "worker-001",
  "crop_type": "maize",
  "demonstration_topic": "Zero tillage maize production",
  "start_date": "2025-03-01"
}
```

### Add a Knowledge Article
```
POST /api/agriculture/ext/knowledge
{
  "title": "Managing Striga in Maize",
  "category": "pest_disease",
  "content": "Striga (witchweed) is a parasitic weed...",
  "crop_types": ["maize", "sorghum"],
  "tags": ["striga", "weed", "parasite"],
  "language": "en"
}
```
