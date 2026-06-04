"""
NLPC Data Models — Natural Language Processing Core

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Website: www.datacraft.co.ke

Pydantic v2 models for all NLPC entities. All models share a common base
with tenant isolation, soft-delete, and audit columns.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, AfterValidator, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_confidence(v: float) -> float:
	if not 0.0 <= v <= 1.0:
		raise ValueError("confidence must be in [0, 1]")
	return v


def _validate_non_empty(v: str) -> str:
	if not v or not v.strip():
		raise ValueError("value cannot be empty")
	return v.strip()


def _validate_non_negative(v: float) -> float:
	if v < 0:
		raise ValueError("value must be non-negative")
	return v


# ---------------------------------------------------------------------------
# Status / Type enums
# ---------------------------------------------------------------------------

class ProcessingStatus(str, Enum):
	PENDING    = "pending"
	QUEUED     = "queued"
	PROCESSING = "processing"
	COMPLETED  = "completed"
	FAILED     = "failed"
	CANCELLED  = "cancelled"
	RETRYING   = "retrying"


class NLPTask(str, Enum):
	LANGUAGE_DETECTION      = "language_detection"
	ENTITY_EXTRACTION       = "entity_extraction"
	SENTIMENT_ANALYSIS      = "sentiment_analysis"
	INTENT_CLASSIFICATION   = "intent_classification"
	TEXT_SUMMARISATION      = "text_summarisation"
	TRANSLATION             = "translation"
	TEXT_EMBEDDING          = "text_embedding"
	DOCUMENT_CLASSIFICATION = "document_classification"
	KEYWORD_EXTRACTION      = "keyword_extraction"
	NAMED_ENTITY_LINKING    = "named_entity_linking"
	RELATION_EXTRACTION     = "relation_extraction"
	COREFERENCE_RESOLUTION  = "coreference_resolution"
	POS_TAGGING             = "pos_tagging"
	DEPENDENCY_PARSING      = "dependency_parsing"
	TOPIC_MODELLING         = "topic_modelling"
	TEXT_SIMILARITY         = "text_similarity"
	PII_DETECTION           = "pii_detection"
	QUESTION_ANSWERING      = "question_answering"
	TEXT_GENERATION         = "text_generation"


class EntityType(str, Enum):
	PERSON       = "PERSON"
	ORGANISATION = "ORG"
	LOCATION     = "LOC"
	DATE         = "DATE"
	TIME         = "TIME"
	MONEY        = "MONEY"
	PERCENT      = "PERCENT"
	PRODUCT      = "PRODUCT"
	EVENT        = "EVENT"
	LAW          = "LAW"
	LANGUAGE     = "LANGUAGE"
	WORK_OF_ART  = "WORK_OF_ART"
	FACILITY     = "FAC"
	GPE          = "GPE"    # geopolitical entity
	NORP         = "NORP"   # nationalities / religions / political groups
	QUANTITY     = "QUANTITY"
	ORDINAL      = "ORDINAL"
	CARDINAL     = "CARDINAL"
	MISC         = "MISC"


class SentimentLabel(str, Enum):
	POSITIVE = "positive"
	NEGATIVE = "negative"
	NEUTRAL  = "neutral"
	MIXED    = "mixed"


class LanguageCode(str, Enum):
	AUTO  = "auto"
	EN    = "en"
	ES    = "es"
	FR    = "fr"
	DE    = "de"
	IT    = "it"
	PT    = "pt"
	RU    = "ru"
	ZH    = "zh"
	JA    = "ja"
	KO    = "ko"
	AR    = "ar"
	HI    = "hi"
	MULTI = "multi"
	# African languages
	AF  = "af"
	AM  = "am"
	BM  = "bm"
	EE  = "ee"
	FF  = "ff"
	HA  = "ha"
	IG  = "ig"
	KI  = "ki"
	RW  = "rw"
	RN  = "rn"
	LN  = "ln"
	LG  = "lg"
	MG  = "mg"
	NY  = "ny"
	OM  = "om"
	SN  = "sn"
	SO  = "so"
	ST  = "st"
	SW  = "sw"
	TI  = "ti"
	TS  = "ts"
	TN  = "tn"
	VE  = "ve"
	WO  = "wo"
	XH  = "xh"
	YO  = "yo"
	ZU  = "zu"
	KAB = "kab"
	KAM = "kam"
	LUO = "luo"
	MAS = "mas"
	MER = "mer"
	MOS = "mos"
	NUS = "nus"
	SUK = "suk"
	TZM = "tzm"
	TIG = "tig"
	UMB = "umb"


AFRICAN_LANGUAGE_CODES: frozenset[str] = frozenset({
	"af", "am", "bm", "ee", "ff", "ha", "ig", "ki", "rw", "rn", "ln", "lg",
	"mg", "ny", "om", "sn", "so", "st", "sw", "ti", "ts", "tn", "ve", "wo",
	"xh", "yo", "zu", "kab", "kam", "luo", "mas", "mer", "mos", "nus", "suk",
	"tzm", "tig", "umb",
})


class ModelProvider(str, Enum):
	OLLAMA       = "ollama"
	TRANSFORMERS = "transformers"
	SPACY        = "spacy"
	LANGDETECT   = "langdetect"
	CUSTOM       = "custom"


class PriorityLevel(str, Enum):
	LOW      = "low"
	NORMAL   = "normal"
	HIGH     = "high"
	CRITICAL = "critical"


class DocumentType(str, Enum):
	PLAIN_TEXT = "text/plain"
	HTML       = "text/html"
	MARKDOWN   = "text/markdown"
	PDF        = "application/pdf"
	JSON       = "application/json"
	XML        = "application/xml"


class ClassificationTaxonomy(str, Enum):
	TOPICS    = "topics"
	SENTIMENT = "sentiment"
	INTENT    = "intent"
	LANGUAGE  = "language"
	CUSTOM    = "custom"


class SummaryMethod(str, Enum):
	EXTRACTIVE  = "extractive"
	ABSTRACTIVE = "abstractive"
	HYBRID      = "hybrid"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class NLPCBase(BaseModel):
	"""Shared base for every NLPC entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True,
	)

	id:         str      = Field(default_factory=uuid7str, description="UUID-7 primary key")
	tenant_id:  str      = Field(description="Tenant owning this record")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str      = Field(default="system")
	is_deleted: bool     = Field(default=False)
	version:    int      = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# NLPDocument
# ---------------------------------------------------------------------------

class NLPDocument(NLPCBase):
	"""A text document submitted for NLP processing."""

	content:        Annotated[str, AfterValidator(_validate_non_empty)] = Field(max_length=10_000_000)
	title:          str | None           = Field(default=None, max_length=500)
	source:         str | None           = None
	source_id:      str | None           = None
	language:       LanguageCode | None  = None
	content_type:   DocumentType         = DocumentType.PLAIN_TEXT
	content_hash:   str | None           = None
	word_count:     int | None           = Field(default=None, ge=0)
	char_count:     int | None           = Field(default=None, ge=0)
	is_sensitive:   bool                 = False
	retention_days: int | None           = Field(default=None, ge=1)
	metadata:       dict[str, Any]       = Field(default_factory=dict)


class NLPDocumentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:      str
	created_by:     str                  = "system"
	content:        str
	title:          str | None           = None
	source:         str | None           = None
	source_id:      str | None           = None
	language:       LanguageCode | None  = None
	content_type:   DocumentType         = DocumentType.PLAIN_TEXT
	is_sensitive:   bool                 = False
	retention_days: int | None           = None
	metadata:       dict[str, Any]       = Field(default_factory=dict)


class NLPDocumentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	title:          str | None           = None
	language:       LanguageCode | None  = None
	is_sensitive:   bool | None          = None
	metadata:       dict[str, Any] | None = None


class NLPDocumentResponse(NLPDocument):
	pass


# ---------------------------------------------------------------------------
# NLPEntity
# ---------------------------------------------------------------------------

class NLPEntity(NLPCBase):
	"""Named entity extracted from a document."""

	document_id:  str
	text:         Annotated[str, AfterValidator(_validate_non_empty)]
	entity_type:  EntityType
	start_char:   int                = Field(ge=0)
	end_char:     int                = Field(ge=0)
	confidence:   Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	canonical:    str | None         = None
	kb_id:        str | None         = None
	kb_url:       str | None         = None
	sentence_idx: int | None         = None
	metadata:     dict[str, Any]     = Field(default_factory=dict)

	@model_validator(mode="after")
	def _end_after_start(self) -> "NLPEntity":
		if self.end_char < self.start_char:
			raise ValueError("end_char must be >= start_char")
		return self


class NLPEntityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:   str
	created_by:  str        = "system"
	document_id: str
	text:        str
	entity_type: EntityType
	start_char:  int
	end_char:    int
	confidence:  float      = 0.0
	canonical:   str | None = None
	kb_id:       str | None = None
	kb_url:      str | None = None


class NLPEntityResponse(NLPEntity):
	pass


# ---------------------------------------------------------------------------
# NLPIntent
# ---------------------------------------------------------------------------

class NLPIntent(NLPCBase):
	"""Detected user intent for a document or utterance."""

	document_id:  str
	intent_label: Annotated[str, AfterValidator(_validate_non_empty)]
	confidence:   Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	all_scores:   dict[str, float] = Field(default_factory=dict)
	model_used:   str | None       = None
	utterance:    str | None       = None
	metadata:     dict[str, Any]   = Field(default_factory=dict)


class NLPIntentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:    str
	created_by:   str             = "system"
	document_id:  str
	intent_label: str
	confidence:   float           = 0.0
	all_scores:   dict[str, float] = Field(default_factory=dict)
	model_used:   str | None      = None
	utterance:    str | None      = None


class NLPIntentResponse(NLPIntent):
	pass


# ---------------------------------------------------------------------------
# NLPSentiment
# ---------------------------------------------------------------------------

class EmotionScores(BaseModel):
	"""Fine-grained Ekman-6 + contempt emotion scores."""
	model_config = ConfigDict(extra="allow", validate_by_name=True)

	joy:      float = Field(default=0.0, ge=0.0, le=1.0)
	sadness:  float = Field(default=0.0, ge=0.0, le=1.0)
	anger:    float = Field(default=0.0, ge=0.0, le=1.0)
	fear:     float = Field(default=0.0, ge=0.0, le=1.0)
	surprise: float = Field(default=0.0, ge=0.0, le=1.0)
	disgust:  float = Field(default=0.0, ge=0.0, le=1.0)
	contempt: float = Field(default=0.0, ge=0.0, le=1.0)


class NLPSentiment(NLPCBase):
	"""Sentiment analysis result for a document."""

	document_id:   str
	label:         SentimentLabel
	score:         Annotated[float, AfterValidator(_validate_confidence)]
	positive:      Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	negative:      Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	neutral:       Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	compound:      float          = Field(default=0.0, ge=-1.0, le=1.0)
	emotions:      EmotionScores  = Field(default_factory=EmotionScores)
	model_used:    str | None     = None
	aspect_scores: list[dict[str, Any]] = Field(default_factory=list)
	metadata:      dict[str, Any] = Field(default_factory=dict)


class NLPSentimentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:   str
	created_by:  str           = "system"
	document_id: str
	label:       SentimentLabel
	score:       float
	positive:    float         = 0.0
	negative:    float         = 0.0
	neutral:     float         = 0.0
	compound:    float         = 0.0
	emotions:    EmotionScores = Field(default_factory=EmotionScores)
	model_used:  str | None    = None


class NLPSentimentResponse(NLPSentiment):
	pass


# ---------------------------------------------------------------------------
# NLPLanguage
# ---------------------------------------------------------------------------

class LanguageCandidate(BaseModel):
	"""Single language detection candidate."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	code:        str
	name:        str | None = None
	probability: Annotated[float, AfterValidator(_validate_confidence)] = 0.0


class NLPLanguage(NLPCBase):
	"""Language identification result for a document."""

	document_id: str
	detected:    LanguageCode
	confidence:  Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	candidates:  list[LanguageCandidate] = Field(default_factory=list)
	script:      str | None  = None
	is_african:  bool        = False
	model_used:  str | None  = None
	metadata:    dict[str, Any] = Field(default_factory=dict)


class NLPLanguageCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:  str
	created_by: str          = "system"
	document_id: str
	detected:   LanguageCode
	confidence: float        = 0.0
	candidates: list[LanguageCandidate] = Field(default_factory=list)
	script:     str | None   = None
	is_african: bool         = False
	model_used: str | None   = None


class NLPLanguageResponse(NLPLanguage):
	pass


# ---------------------------------------------------------------------------
# NLPSummary
# ---------------------------------------------------------------------------

class NLPSummary(NLPCBase):
	"""Extractive or abstractive summary of a document."""

	document_id:       str
	summary_text:      Annotated[str, AfterValidator(_validate_non_empty)]
	method:            SummaryMethod = SummaryMethod.EXTRACTIVE
	max_words:         int | None    = Field(default=None, ge=1)
	actual_word_count: int           = Field(default=0, ge=0)
	compression_ratio: Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	key_sentences:     list[str]     = Field(default_factory=list)
	model_used:        str | None    = None
	metadata:          dict[str, Any] = Field(default_factory=dict)


class NLPSummaryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:    str
	created_by:   str          = "system"
	document_id:  str
	summary_text: str
	method:       SummaryMethod = SummaryMethod.EXTRACTIVE
	max_words:    int | None   = None
	model_used:   str | None   = None


class NLPSummaryResponse(NLPSummary):
	pass


# ---------------------------------------------------------------------------
# NLPTranslation
# ---------------------------------------------------------------------------

class NLPTranslation(NLPCBase):
	"""Translation of document text to a target language."""

	document_id:     str
	source_language: LanguageCode
	target_language: LanguageCode
	translated_text: Annotated[str, AfterValidator(_validate_non_empty)]
	confidence:      Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	model_used:      str | None     = None
	char_count:      int            = Field(default=0, ge=0)
	metadata:        dict[str, Any] = Field(default_factory=dict)

	@field_validator("target_language")
	@classmethod
	def _target_not_auto(cls, v: str) -> str:
		if v == LanguageCode.AUTO.value or v == "auto":
			raise ValueError("target_language must be a specific language code, not 'auto'")
		return v


class NLPTranslationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:       str
	created_by:      str         = "system"
	document_id:     str
	source_language: LanguageCode
	target_language: LanguageCode
	translated_text: str
	confidence:      float       = 0.0
	model_used:      str | None  = None


class NLPTranslationResponse(NLPTranslation):
	pass


# ---------------------------------------------------------------------------
# NLPEmbedding
# ---------------------------------------------------------------------------

class NLPEmbedding(NLPCBase):
	"""Dense vector embedding of document text."""

	document_id:    str
	vector:         list[float]   = Field(description="Embedding vector")
	dimensions:     int           = Field(ge=1)
	model_used:     str
	model_provider: ModelProvider = ModelProvider.OLLAMA
	norm:           float | None  = None
	chunk_index:    int | None    = Field(default=None, ge=0)
	chunk_text:     str | None    = None
	metadata:       dict[str, Any] = Field(default_factory=dict)

	@model_validator(mode="after")
	def _dimensions_match(self) -> "NLPEmbedding":
		if len(self.vector) != self.dimensions:
			raise ValueError(f"vector length {len(self.vector)} != dimensions {self.dimensions}")
		return self


class NLPEmbeddingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:      str
	created_by:     str           = "system"
	document_id:    str
	vector:         list[float]
	dimensions:     int
	model_used:     str
	model_provider: ModelProvider = ModelProvider.OLLAMA
	chunk_index:    int | None    = None
	chunk_text:     str | None    = None


class NLPEmbeddingResponse(NLPEmbedding):
	"""Bandwidth-friendly response — full vector excluded by default."""
	vector:         list[float] = Field(default_factory=list, exclude=True)
	vector_preview: list[float] = Field(default_factory=list, description="First 8 dims")

	@model_validator(mode="after")
	def _fill_preview(self) -> "NLPEmbeddingResponse":
		if not self.vector_preview and self.vector:
			self.vector_preview = self.vector[:8]
		return self


# ---------------------------------------------------------------------------
# NLPClassification
# ---------------------------------------------------------------------------

class NLPClassification(NLPCBase):
	"""Document classification result against a taxonomy."""

	document_id: str
	taxonomy:    ClassificationTaxonomy
	label:       Annotated[str, AfterValidator(_validate_non_empty)]
	confidence:  Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	all_scores:  dict[str, float] = Field(default_factory=dict)
	hierarchy:   list[str]        = Field(default_factory=list)
	model_used:  str | None       = None
	metadata:    dict[str, Any]   = Field(default_factory=dict)


class NLPClassificationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, str_strip_whitespace=True, use_enum_values=True)

	tenant_id:   str
	created_by:  str                  = "system"
	document_id: str
	taxonomy:    ClassificationTaxonomy
	label:       str
	confidence:  float                = 0.0
	all_scores:  dict[str, float]     = Field(default_factory=dict)
	hierarchy:   list[str]            = Field(default_factory=list)
	model_used:  str | None           = None


class NLPClassificationResponse(NLPClassification):
	pass


# ---------------------------------------------------------------------------
# NLPRelation
# ---------------------------------------------------------------------------

class NLPRelation(NLPCBase):
	"""Directed relation between two entities in a document."""

	document_id:  str
	subject_id:   str
	object_id:    str
	relation:     str
	confidence:   Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	sentence_idx: int | None     = None
	model_used:   str | None     = None
	metadata:     dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# NLPCoreferenceChain
# ---------------------------------------------------------------------------

class NLPCoreferenceChain(NLPCBase):
	"""Coreference chain — set of mentions referring to the same entity."""

	document_id:    str
	cluster_id:     int           = Field(ge=0)
	mentions:       list[dict[str, Any]] = Field(min_length=1)
	representative: str | None    = None
	entity_id:      str | None    = None
	model_used:     str | None    = None


# ---------------------------------------------------------------------------
# NLPKeyPhrase
# ---------------------------------------------------------------------------

class NLPKeyPhrase(NLPCBase):
	"""Key phrase extracted from a document."""

	document_id: str
	phrase:      Annotated[str, AfterValidator(_validate_non_empty)]
	score:       Annotated[float, AfterValidator(_validate_confidence)] = 0.0
	frequency:   int          = Field(default=1, ge=1)
	method:      str          = "tfidf"
	metadata:    dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# NLPModelConfig
# ---------------------------------------------------------------------------

class NLPModelConfig(NLPCBase):
	"""Runtime configuration for a loaded NLP model."""

	name:                Annotated[str, AfterValidator(_validate_non_empty)]
	provider:            ModelProvider
	model_name:          str
	supported_tasks:     list[NLPTask]        = Field(min_length=1)
	supported_languages: list[LanguageCode]   = Field(default_factory=lambda: [LanguageCode.EN])
	max_input_chars:     int                  = Field(default=100_000, ge=1)
	gpu_required:        bool                 = False
	memory_mb:           int | None           = Field(default=None, ge=0)
	is_active:           bool                 = True
	load_priority:       int                  = Field(default=50, ge=1, le=100)
	configuration:       dict[str, Any]       = Field(default_factory=dict)
	performance_metrics: dict[str, float]     = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# NLPBatchJob
# ---------------------------------------------------------------------------

class NLPBatchJob(NLPCBase):
	"""Batch processing job spanning multiple documents."""

	name:                 Annotated[str, AfterValidator(_validate_non_empty)]
	document_ids:         list[str]        = Field(min_length=1)
	tasks:                list[NLPTask]    = Field(min_length=1)
	status:               ProcessingStatus = ProcessingStatus.PENDING
	priority:             PriorityLevel    = PriorityLevel.NORMAL
	progress:             float            = Field(default=0.0, ge=0.0, le=100.0)
	total_documents:      int              = Field(ge=1)
	processed_documents:  int              = Field(default=0, ge=0)
	failed_documents:     int              = Field(default=0, ge=0)
	started_at:           datetime | None  = None
	completed_at:         datetime | None  = None
	error_summary:        str | None       = None
	configuration:        dict[str, Any]   = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Processing request / result
# ---------------------------------------------------------------------------

class NLPProcessingRequest(NLPCBase):
	"""Request to run one or more NLP tasks against a document."""

	document_id:     str
	tasks:           list[NLPTask]   = Field(min_length=1)
	priority:        PriorityLevel   = PriorityLevel.NORMAL
	parameters:      dict[str, Any]  = Field(default_factory=dict)
	callback_url:    str | None      = None
	batch_id:        str | None      = None
	max_seconds:     int | None      = Field(default=None, ge=1)
	require_explain: bool            = False

	@field_validator("tasks")
	@classmethod
	def _no_duplicate_tasks(cls, v: list[NLPTask]) -> list[NLPTask]:
		seen: set[str] = set()
		out = []
		for t in v:
			if t not in seen:
				seen.add(t)
				out.append(t)
		return out


class NLPProcessingResult(NLPCBase):
	"""Result of a single NLP task execution."""

	request_id:     str
	document_id:    str
	task:           NLPTask
	status:         ProcessingStatus      = ProcessingStatus.COMPLETED
	confidence:     float | None          = Field(default=None, ge=0.0, le=1.0)
	processing_ms:  Annotated[float, AfterValidator(_validate_non_negative)] = 0.0
	model_used:     str | None            = None
	model_provider: ModelProvider | None  = None
	result_data:    dict[str, Any]        = Field(default_factory=dict)
	error_message:  str | None            = None
	cache_hit:      bool                  = False
	explanation:    dict[str, Any] | None = None

	@property
	def succeeded(self) -> bool:
		return self.status == ProcessingStatus.COMPLETED


# ---------------------------------------------------------------------------
# Analytics / report
# ---------------------------------------------------------------------------

class NLPUsageReport(BaseModel):
	"""Aggregated usage statistics for a tenant within a time window."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id:              str
	period_start:           datetime
	period_end:             datetime
	total_requests:         int   = 0
	total_documents:        int   = 0
	total_tokens_processed: int   = 0
	task_breakdown:         dict[str, int]   = Field(default_factory=dict)
	model_breakdown:        dict[str, int]   = Field(default_factory=dict)
	language_breakdown:     dict[str, int]   = Field(default_factory=dict)
	avg_processing_ms:      float = 0.0
	p95_processing_ms:      float = 0.0
	error_rate:             float = Field(default=0.0, ge=0.0, le=1.0)
	cache_hit_rate:         float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Forward-ref rebuild
# ---------------------------------------------------------------------------

NLPDocument.model_rebuild()
NLPEntity.model_rebuild()
NLPSentiment.model_rebuild()
NLPEmbedding.model_rebuild()
NLPEmbeddingResponse.model_rebuild()
NLPProcessingRequest.model_rebuild()
NLPProcessingResult.model_rebuild()
NLPBatchJob.model_rebuild()

# Backward-compatibility alias
ProcessingRequest = NLPProcessingRequest
ProcessingResult = NLPProcessingResult
ProcessingRecord = NLPDocument  # backward-compat alias

