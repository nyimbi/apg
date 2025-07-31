"""
Conflict Keyword Manager
========================

Manages and updates conflict-related keywords for search operations.
Provides dynamic keyword generation and relevance scoring.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeywordSet:
    """A set of related keywords with metadata."""
    name: str
    keywords: List[str]
    weight: float = 1.0
    category: str = "general"
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "manual"
    confidence: float = 1.0


@dataclass
class SearchQuery:
    """A structured search query with metadata."""
    query: str
    keywords: List[str]
    expected_relevance: float
    target_engines: List[str] = field(default_factory=lambda: ["google", "bing"])
    priority: int = 1  # 1=high, 2=medium, 3=low
    created_at: datetime = field(default_factory=datetime.now)


class ConflictKeywordManager:
    """Manages conflict-related keywords for search operations."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core keyword sets
        self.keyword_sets: Dict[str, KeywordSet] = {}
        self.query_cache: Dict[str, List[SearchQuery]] = {}
        self.performance_stats = {
            'queries_generated': 0,
            'successful_matches': 0,
            'last_update': datetime.now()
        }

        self._initialize_core_keywords()

    def _initialize_core_keywords(self):
        """Initialize core conflict keyword sets."""

        # Violence and conflict
        self.add_keyword_set(KeywordSet(
            name="violence",
            keywords=[
                "attack", "attacks", "violence", "violent", "fighting", "clash", "clashes",
                "battle", "battles", "armed conflict", "civil war", "war", "warfare",
                "combat", "conflict", "conflicts", "insurgency", "rebellion", "uprising",
                "militancy", "militant", "militants", "raid", "raids", "assault",
                "skirmish", "skirmishes", "confrontation", "confrontations", "hostilities",
                "aggression", "aggressive", "offensive", "strike", "strikes", "ambush",
                "ambushes", "firefight", "gunfight", "shootout", "armed confrontation",
                "military operation", "counteroffensive", "siege", "blockade", "invasion",
                "occupation", "ethnic violence", "sectarian violence", "communal violence",
                "gang violence", "tribal conflict", "inter-communal conflict", "vendetta",
                "retaliation", "revenge attacks", "crackdown", "military action",
                "security operation", "peacekeeping", "intervention", "escalation",
                "hostility", "armed resistance", "guerrilla", "guerrilla warfare",
                "paramilitary", "militia", "armed groups", "faction", "factions",
                "warring parties", "belligerents", "combatants", "armed forces",
                "government forces", "opposition forces", "rebel forces", "insurgent groups"
            ],
            weight=0.9,
            category="high_priority",
            source="expert_curated"
        ))

        # Casualties and impact
        self.add_keyword_set(KeywordSet(
            name="casualties",
            keywords=[
                "killed", "death", "deaths", "casualties", "wounded", "injured",
                "victim", "victims", "fatalities", "dead", "dying", "massacre",
                "slaughter", "bloodshed", "execution", "murder", "assassinate",
                "assassinated", "assassination", "murdered", "homicide", "homicides",
                "body count", "death toll", "casualty figures", "loss of life",
                "perished", "perish", "deceased", "mortal", "mortality", "lethal",
                "fatal", "fatally", "critically injured", "seriously wounded",
                "life-threatening", "intensive care", "hospitalized", "medical emergency",
                "trauma", "traumatic injuries", "survivors", "survivor", "maimed",
                "dismembered", "mutilated", "tortured", "torture", "beaten to death",
                "lynched", "lynching", "stoned to death", "burned alive", "beheaded",
                "beheading", "shot dead", "gunned down", "stabbed to death",
                "mass killing", "mass murder", "genocide", "ethnic cleansing",
                "war crimes", "crimes against humanity", "extermination",
                "systematic killing", "targeted killing", "summary execution",
                "extrajudicial killing", "civilian casualties", "collateral damage",
                "friendly fire", "crossfire", "caught in crossfire", "human cost",
                "burial", "burials", "funeral", "funerals", "morgue", "mortuary",
                "missing persons", "disappeared", "enforced disappearance",
                "presumed dead", "confirmed dead", "identified bodies",
                "unidentified remains", "mass grave", "mass graves"
            ],
            weight=0.95,
            category="high_priority",
            source="expert_curated"
        ))

        # Security and stability
        self.add_keyword_set(KeywordSet(
            name="security",
            keywords=[
                "security", "insecurity", "instability", "unrest", "riots", "riot",
                "protest", "protests", "demonstration", "demonstrations", "emergency",
                "crisis", "threat", "threats", "danger", "dangerous", "unsafe",
                "martial law", "curfew", "lockdown", "quarantine", "restrictions",
                "surveillance", "monitoring", "intelligence", "counter-terrorism",
                "peacekeeping", "security forces", "police", "military", "army",
                "national guard", "border security", "checkpoint", "checkpoints",
                "roadblock", "roadblocks", "barricade", "barricades", "fortification",
                "defense", "defensive", "protection", "protective", "secure",
                "secured", "securing", "patrol", "patrols", "patrolling",
                "law and order", "public order", "civil disorder", "disorder",
                "disturbance", "disturbances", "upheaval", "turmoil", "chaos",
                "anarchy", "lawlessness", "breakdown", "collapse", "deterioration",
                "escalation", "tension", "tensions", "volatile", "volatility",
                "fragile", "fragility", "vulnerability", "vulnerable", "at risk",
                "high risk", "risk assessment", "threat level", "alert level",
                "warning", "warnings", "advisory", "advisories", "evacuation order",
                "state of emergency", "emergency powers", "emergency response",
                "security alert", "security warning", "security breach", "breach",
                "incident", "incidents", "situation", "developing situation",
                "containment", "control", "crowd control", "riot control",
                "tear gas", "pepper spray", "rubber bullets", "water cannon",
                "dispersal", "disperse", "scattered", "regrouped", "mobilized",
                "mobilization", "deployment", "deployed", "reinforcements",
                "backup", "support", "assistance", "intervention", "response",
                "crackdown", "sweep", "sweeps", "raid", "raids", "search",
                "searches", "seizure", "confiscation", "detention", "detainment",
                "arrest", "arrests", "custody", "interrogation", "questioning"
            ],
            weight=0.7,
            category="medium_priority",
            source="expert_curated"
        ))

        # Displacement and humanitarian
        self.add_keyword_set(KeywordSet(
            name="displacement",
            keywords=[
                "refugee", "refugees", "displaced", "displacement", "IDP", "IDPs",
                "internally displaced", "flee", "fled", "fleeing", "evacuation",
                "evacuated", "humanitarian crisis", "famine", "drought", "aid",
                "asylum seeker", "asylum seekers", "stateless", "stateless persons",
                "migrant", "migrants", "forced migration", "mass exodus", "exodus",
                "relocation", "relocated", "resettlement", "resettled", "repatriation",
                "returnee", "returnees", "voluntary return", "forced return",
                "deportation", "deported", "expulsion", "expelled", "banishment",
                "exile", "exiled", "diaspora", "dispersed", "scattered",
                "homeless", "homelessness", "shelter", "shelters", "camp", "camps",
                "refugee camp", "displacement camp", "temporary shelter",
                "emergency shelter", "makeshift shelter", "tent city", "settlement",
                "informal settlement", "squatter", "squatters", "slum", "slums",
                "host community", "host families", "reception center", "transit center",
                "border crossing", "border crossings", "crossing point", "entry point",
                "safe passage", "corridor", "humanitarian corridor", "escape route",
                "sanctuary", "safe haven", "protection", "international protection",
                "non-refoulement", "persecution", "persecuted", "ethnic cleansing",
                "population transfer", "forced relocation", "expulsion order",
                "humanitarian emergency", "emergency response", "relief operation",
                "humanitarian aid", "emergency aid", "disaster relief", "food aid",
                "medical aid", "shelter assistance", "cash assistance", "livelihood support",
                "protection services", "child protection", "gender-based violence",
                "trafficking", "human trafficking", "smuggling", "people smuggling",
                "vulnerable groups", "unaccompanied minors", "separated children",
                "elderly", "disabled", "pregnant women", "lactating mothers",
                "malnutrition", "malnourished", "stunting", "wasting", "underweight",
                "food insecurity", "food shortage", "hunger", "starvation", "starving",
                "water shortage", "water scarcity", "sanitation", "hygiene",
                "disease outbreak", "epidemic", "pandemic", "cholera", "measles",
                "acute watery diarrhea", "respiratory infections", "mental health",
                "trauma", "psychosocial support", "PTSD", "psychological distress",
                "documentation", "identity documents", "birth certificate", "passport",
                "legal status", "legal documentation", "registration", "civil registration",
                "UNHCR", "UNICEF", "WFP", "WHO", "IOM", "ICRC", "MSF", "NGO", "INGO",
                "humanitarian organization", "aid agency", "relief agency",
                "donor", "funding", "humanitarian funding", "emergency funding",
                "appeal", "humanitarian appeal", "flash appeal", "consolidated appeal",
                "cluster approach", "coordination", "humanitarian coordination",
                "needs assessment", "rapid assessment", "vulnerability assessment",
                "livelihood", "livelihoods", "income generation", "employment",
                "education", "schooling", "learning", "literacy", "skills training",
                "integration", "local integration", "self-reliance", "empowerment",
                "durable solutions", "voluntary repatriation", "local integration",
                "third country resettlement", "complementary pathways"
            ],
            weight=0.8,
            category="medium_priority",
            source="expert_curated"
        ))

        # Terrorism and extremism
        self.add_keyword_set(KeywordSet(
            name="terrorism",
            keywords=[
                "terrorism", "terrorist", "terrorists", "extremist", "extremists",
                "bomb", "bombing", "explosion", "blast", "suicide bomb", "IED",
                "al-shabaab", "al shabaab", "jihadist", "jihadists", "radicalization",
                "improvised explosive device", "car bomb", "truck bomb", "roadside bomb",
                "explosive device", "detonation", "detonated", "sabotage", "arson",
                "incendiary", "militant group", "armed group", "terror cell", "cell",
                "sleeper cell", "network", "terror network", "extremist network",
                "radical", "radicals", "fundamentalist", "fundamentalists", "fanatic",
                "fanatics", "ideological", "ideology", "violent extremism", "domestic terrorism",
                "international terrorism", "state terrorism", "cyber terrorism", "bioterrorism",
                "chemical attack", "biological attack", "nerve agent", "toxic", "poison",
                "hostage", "hostages", "kidnap", "kidnapped", "kidnapping", "abduction",
                "abducted", "ransom", "hijack", "hijacked", "hijacking", "plane hijacking",
                "siege", "barricade", "standoff", "takeover", "occupation", "infiltration",
                "recruitment", "recruiter", "indoctrination", "propaganda", "manifesto",
                "martyrdom", "martyr", "martyrs", "suicide bomber", "suicide attack",
                "kamikaze", "human bomb", "vest bomb", "explosive vest", "backpack bomb",
                "lone wolf", "self-radicalized", "homegrown", "sleeper", "operative",
                "handler", "coordinator", "facilitator", "financier", "supporter",
                "sympathizer", "accomplice", "conspirator", "plotter", "planning",
                "plot", "scheme", "conspiracy", "threat", "menace", "danger",
                "target", "targets", "targeting", "surveillance", "reconnaissance",
                "intelligence", "counter-terrorism", "anti-terrorism", "security alert",
                "terror alert", "threat level", "watch list", "no-fly list",
                "isis", "isil", "islamic state", "boko haram", "al-qaeda", "taliban",
                "hezbollah", "hamas", "plo", "ira", "eta", "pkk", "farc",
                "red army faction", "baader-meinhof", "weather underground", "aum shinrikyo",
                "sectarian", "ethnic violence", "hate crime", "hate group", "supremacist",
                "nazi", "fascist", "anarchist", "separatist", "nationalist", "militia",
                "paramilitary", "guerrilla", "insurgent", "rebel", "revolutionary",
                "underground", "clandestine", "covert", "secret", "hidden", "concealed",
                "weapon", "weapons", "arms", "ammunition", "firearm", "rifle", "pistol",
                "machine gun", "assault rifle", "sniper", "grenade", "mortar", "rocket",
                "missile", "launcher", "rpg", "explosive", "plastic explosive", "dynamite",
                "fertilizer bomb", "pressure cooker bomb", "pipe bomb", "nail bomb",
                "dirty bomb", "radiological", "nuclear", "uranium", "plutonium",
                "anthrax", "ricin", "sarin", "mustard gas", "chlorine", "cyanide"
            ],
            weight=1.0,
            category="high_priority",
            source="expert_curated"
        ))

        # Political instability
        self.add_keyword_set(KeywordSet(
            name="political",
            keywords=[
                "coup", "revolution", "overthrow", "government crisis", "regime change",
                "political crisis", "election violence", "electoral dispute",
                "constitutional crisis", "power struggle", "authoritarian",
                "dictatorship", "dictator", "autocracy", "autocrat", "totalitarian",
                "oppression", "oppressive", "repression", "repressive", "tyranny",
                "tyrant", "despotism", "despot", "military rule", "junta",
                "putsch", "palace coup", "soft coup", "constitutional coup",
                "coup attempt", "coup plot", "insurgency", "insurrection",
                "rebellion", "revolt", "uprising", "mutiny", "sedition",
                "subversion", "overthrow attempt", "power grab", "takeover",
                "regime collapse", "government collapse", "state failure",
                "failed state", "political vacuum", "power vacuum", "anarchy",
                "lawlessness", "breakdown of order", "institutional collapse",
                "democratic backsliding", "erosion of democracy", "authoritarianism",
                "one-party rule", "single-party state", "police state",
                "surveillance state", "censorship", "media suppression",
                "freedom of speech", "human rights violations", "civil liberties",
                "political prisoners", "arbitrary detention", "enforced disappearance",
                "extrajudicial execution", "state terrorism", "state violence",
                "crackdown", "purge", "purges", "political purge", "witch hunt",
                "show trial", "kangaroo court", "rigged election", "electoral fraud",
                "vote rigging", "ballot stuffing", "voter intimidation",
                "electoral manipulation", "gerrymandering", "disenfranchisement",
                "voter suppression", "poll violence", "campaign violence",
                "political assassination", "targeted killing", "hit squad",
                "death squad", "paramilitary", "militia", "private army",
                "warlord", "strongman", "caudillo", "kleptocracy", "corruption",
                "nepotism", "cronyism", "patronage", "clientelism", "rent-seeking",
                "embezzlement", "misappropriation", "graft", "bribery", "kickbacks",
                "political patronage", "spoils system", "political machine",
                "ruling party", "opposition party", "political opposition",
                "dissent", "dissident", "activist", "political activist",
                "protest movement", "civil disobedience", "resistance movement",
                "underground movement", "guerrilla movement", "separatist movement",
                "independence movement", "liberation movement", "freedom fighters",
                "political violence", "sectarian politics", "ethnic politics",
                "tribal politics", "identity politics", "polarization",
                "political polarization", "extremism", "radicalization",
                "fundamentalism", "nationalism", "ultranationalism", "populism",
                "demagogue", "propaganda", "disinformation", "fake news",
                "state media", "controlled media", "media manipulation",
                "information warfare", "psychological operations", "psyops",
                "political interference", "foreign interference", "election interference",
                "proxy war", "puppet government", "client state", "sphere of influence",
                "geopolitical rivalry", "great power competition", "cold war",
                "diplomatic crisis", "diplomatic breakdown", "sanctions",
                "economic sanctions", "trade war", "embargo", "blockade",
                "isolation", "pariah state", "rogue state", "axis of evil",
                "regime survival", "legitimacy crisis", "succession crisis",
                "dynastic rule", "hereditary rule", "personality cult",
                "cult of personality", "political dynasty", "ruling family",
                "political elite", "oligarchy", "plutocracy", "technocracy",
                "meritocracy", "bureaucracy", "deep state", "shadow government",
                "parallel government", "government-in-exile", "interim government",
                "transitional government", "provisional government", "caretaker government",
                "coalition government", "unity government", "national unity",
                "power sharing", "consociational democracy", "federal system",
                "devolution", "autonomy", "self-rule", "home rule", "independence",
                "secession", "breakaway", "partition", "balkanization", "fragmentation"
            ],
            weight=0.75,
            category="medium_priority",
            source="expert_curated"
        ))

        # Resource conflicts
        self.add_keyword_set(KeywordSet(
            name="resources",
            keywords=[
                "water conflict", "land dispute", "resource conflict", "grazing rights",
                "pastoralist conflict", "farmer herder", "drought conflict",
                "water scarcity", "land grabbing", "boundary dispute",
                "natural resources", "resource extraction", "mining conflict",
                "oil conflict", "gas pipeline", "water rights", "irrigation dispute",
                "fishing rights", "maritime boundary", "territorial waters",
                "exclusive economic zone", "continental shelf", "seabed mining",
                "freshwater resources", "groundwater", "aquifer", "river dispute",
                "dam construction", "hydroelectric", "water diversion",
                "desalination", "water allocation", "water sharing", "water treaty",
                "transboundary water", "upstream downstream", "water stress",
                "water crisis", "water war", "hydro-politics", "water diplomacy",
                "agricultural land", "arable land", "farmland", "cropland",
                "pasture land", "grazing land", "rangeland", "communal land",
                "private land", "state land", "public land", "indigenous land",
                "ancestral land", "traditional territory", "sacred land",
                "land ownership", "land tenure", "property rights", "title deed",
                "land registration", "cadastral", "surveying", "demarcation",
                "encroachment", "illegal occupation", "squatting", "eviction",
                "forced eviction", "land clearance", "deforestation", "logging",
                "timber rights", "forest concession", "protected area",
                "national park", "conservation", "biodiversity", "ecosystem",
                "environmental degradation", "habitat loss", "species extinction",
                "climate change", "global warming", "sea level rise",
                "coastal erosion", "desertification", "soil degradation",
                "overgrazing", "overfishing", "overhunting", "poaching",
                "wildlife trafficking", "illegal wildlife trade", "ivory trade",
                "rhino horn", "pangolin scales", "shark fins", "bushmeat",
                "mineral resources", "precious metals", "rare earth elements",
                "lithium", "cobalt", "coltan", "diamonds", "gold", "copper",
                "iron ore", "bauxite", "uranium", "coal", "petroleum",
                "natural gas", "shale gas", "fracking", "offshore drilling",
                "pipeline", "refinery", "petrochemicals", "energy security",
                "energy independence", "renewable energy", "solar", "wind",
                "hydropower", "geothermal", "biofuel", "ethanol", "palm oil",
                "food security", "food crisis", "agricultural productivity",
                "crop yield", "harvest", "livestock", "cattle", "sheep",
                "goats", "camels", "nomadic", "transhumance", "migration routes",
                "seasonal movement", "grazing patterns", "water points",
                "boreholes", "wells", "springs", "oases", "reservoirs",
                "dams", "irrigation systems", "canals", "aqueducts",
                "land use planning", "zoning", "urban expansion", "suburbanization",
                "infrastructure development", "road construction", "railways",
                "airports", "ports", "harbors", "coastal development",
                "industrial zones", "special economic zones", "free trade zones",
                "investment", "foreign investment", "multinational corporations",
                "extractive industries", "agribusiness", "plantation",
                "monoculture", "cash crops", "export crops", "subsistence farming",
                "smallholder farmers", "large-scale farming", "mechanization",
                "fertilizers", "pesticides", "herbicides", "genetically modified",
                "seeds", "hybrid varieties", "crop rotation", "intercropping",
                "agroforestry", "permaculture", "organic farming", "sustainable agriculture",
                "land reform", "redistribution", "restitution", "compensation",
                "resettlement", "relocation", "development-induced displacement",
                "involuntary resettlement", "economic displacement", "livelihood restoration",
                "community consultation", "free prior informed consent",
                "environmental impact assessment", "social impact assessment",
                "mitigation measures", "offset", "restoration", "rehabilitation",
                "remediation", "cleanup", "pollution", "contamination",
                "toxic waste", "chemical spills", "oil spills", "mining waste",
                "tailings", "slag", "acid mine drainage", "heavy metals",
                "environmental justice", "environmental racism", "sacrifice zones",
                "frontline communities", "indigenous rights", "tribal sovereignty",
                "customary law", "traditional knowledge", "local communities",
                "community-based management", "participatory management",
                "co-management", "collaborative governance", "stakeholder engagement",
                "conflict resolution", "mediation", "arbitration", "negotiation",
                "peace-building", "reconciliation", "truth and reconciliation",
                "transitional justice", "reparations", "acknowledgment",
                "resource sharing", "benefit sharing", "revenue sharing",
                "royalties", "taxes", "resource curse", "Dutch disease",
                "economic diversification", "value addition", "processing",
                "manufacturing", "industrialization", "technology transfer",
                "capacity building", "skills development", "education", "training"
            ],
            weight=0.6,
            category="low_priority",
            source="expert_curated"
        ))

        # Temporal indicators
        self.add_keyword_set(KeywordSet(
            name="temporal",
            keywords=[
                "breaking", "urgent", "alert", "latest", "developing", "ongoing",
                "recent", "today", "yesterday", "this week", "current", "now",
                "just in", "live", "immediate", "emergency", "flash", "bulletin",
                "update", "updated", "new", "fresh", "hot", "instant", "real-time",
                "this hour", "this morning", "this afternoon", "this evening", "tonight",
                "last night", "early morning", "dawn", "midnight", "late night",
                "moments ago", "minutes ago", "hours ago", "shortly", "soon",
                "imminent", "impending", "forthcoming", "approaching", "near",
                "within hours", "within minutes", "any moment", "expected soon",
                "unfolding", "in progress", "happening now", "as we speak",
                "live coverage", "eyewitness", "on scene", "first reports",
                "preliminary", "initial", "early reports", "unconfirmed",
                "confirmed", "verified", "authenticated", "official",
                "exclusive", "special report", "newsflash", "news alert",
                "crisis", "emergency situation", "critical", "priority",
                "high priority", "urgent update", "fast-moving", "rapidly developing",
                "evolving", "changing", "fluid situation", "dynamic",
                "last week", "past week", "recent days", "past few days",
                "over the weekend", "weekend", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday", "dawn raid",
                "morning attack", "evening assault", "overnight", "pre-dawn",
                "midday", "afternoon incident", "late evening", "early hours",
                "24 hours", "48 hours", "72 hours", "this month", "last month",
                "past month", "recent weeks", "coming days", "next few days",
                "in the coming hours", "later today", "earlier today",
                "this year", "2024", "2023", "January", "February", "March",
                "April", "May", "June", "July", "August", "September",
                "October", "November", "December", "Q1", "Q2", "Q3", "Q4",
                "first quarter", "second quarter", "third quarter", "fourth quarter",
                "year-to-date", "so far this year", "annual", "yearly",
                "seasonal", "during", "while", "when", "as", "since",
                "before", "after", "following", "preceding", "prior to",
                "in the wake of", "aftermath", "post-", "pre-", "mid-",
                "early", "late", "end of", "beginning of", "start of"
            ],
            weight=0.5,
            category="temporal",
            source="expert_curated"
        ))

    def add_keyword_set(self, keyword_set: KeywordSet):
        """Add a keyword set to the manager."""
        self.keyword_sets[keyword_set.name] = keyword_set
        self.logger.debug(f"Added keyword set: {keyword_set.name} with {len(keyword_set.keywords)} keywords")

    def get_keywords_by_category(self, category: str) -> List[str]:
        """Get all keywords for a specific category."""
        keywords = []
        for keyword_set in self.keyword_sets.values():
            if keyword_set.category == category:
                keywords.extend(keyword_set.keywords)
        return list(set(keywords))  # Remove duplicates

    def get_high_priority_keywords(self) -> List[str]:
        """Get high priority conflict keywords."""
        return self.get_keywords_by_category("high_priority")

    def get_weighted_keywords(self, min_weight: float = 0.7) -> Dict[str, float]:
        """Get keywords with their weights, filtered by minimum weight."""
        weighted_keywords = {}
        for keyword_set in self.keyword_sets.values():
            if keyword_set.weight >= min_weight:
                for keyword in keyword_set.keywords:
                    # Use the highest weight if keyword appears in multiple sets
                    if keyword in weighted_keywords:
                        weighted_keywords[keyword] = max(weighted_keywords[keyword], keyword_set.weight)
                    else:
                        weighted_keywords[keyword] = keyword_set.weight
        return weighted_keywords

    def generate_search_queries(
        self,
        location_keywords: List[str],
        max_queries: int = 30,
        include_temporal: bool = True,
        priority_filter: Optional[List[str]] = None
    ) -> List[SearchQuery]:
        """
        Generate optimized search queries combining conflict and location keywords.

        Args:
            location_keywords: Location-specific keywords
            max_queries: Maximum number of queries to generate
            include_temporal: Include temporal keywords
            priority_filter: Filter by keyword categories

        Returns:
            List of SearchQuery objects
        """
        if priority_filter is None:
            priority_filter = ["high_priority", "medium_priority"]

        queries = []

        # Get relevant keyword sets
        relevant_sets = [ks for ks in self.keyword_sets.values()
                        if ks.category in priority_filter]

        # Sort by weight (highest first)
        relevant_sets.sort(key=lambda x: x.weight, reverse=True)

        # Generate location + conflict combinations
        for location in location_keywords[:10]:  # Limit locations
            for keyword_set in relevant_sets:
                for keyword in keyword_set.keywords[:5]:  # Top 5 keywords per set
                    if len(queries) >= max_queries:
                        break

                    # Basic query: "location" + keyword
                    query_text = f'"{location}" {keyword}'
                    expected_relevance = keyword_set.weight * 0.8

                    # Add temporal modifier occasionally
                    if include_temporal and len(queries) % 4 == 0:
                        temporal_keywords = self.keyword_sets.get("temporal", KeywordSet("temporal", [])).keywords
                        if temporal_keywords:
                            query_text = f"{temporal_keywords[0]} {query_text}"
                            expected_relevance += 0.1

                    queries.append(SearchQuery(
                        query=query_text,
                        keywords=[location, keyword],
                        expected_relevance=min(expected_relevance, 1.0),
                        priority=1 if keyword_set.category == "high_priority" else 2
                    ))

                if len(queries) >= max_queries:
                    break

            if len(queries) >= max_queries:
                break

        # Add some broader regional queries
        if len(queries) < max_queries:
            high_priority_keywords = self.get_high_priority_keywords()
            for keyword in high_priority_keywords[:5]:
                if len(queries) >= max_queries:
                    break
                queries.append(SearchQuery(
                    query=f'"Horn of Africa" {keyword}',
                    keywords=["Horn of Africa", keyword],
                    expected_relevance=0.85,
                    priority=1
                ))

        # Sort by priority and expected relevance
        queries.sort(key=lambda x: (x.priority, -x.expected_relevance))

        # Update stats
        self.performance_stats['queries_generated'] += len(queries)
        self.performance_stats['last_update'] = datetime.now()

        return queries[:max_queries]

    def score_text_relevance(self, text: str, boost_recent: bool = True) -> float:
        """
        Score text relevance based on keyword matches.

        Args:
            text: Text to analyze
            boost_recent: Boost score for temporal indicators

        Returns:
            Relevance score between 0.0 and 1.0
        """
        text_lower = text.lower()
        total_score = 0.0
        max_possible_score = 0.0

        for keyword_set in self.keyword_sets.values():
            if keyword_set.category == "temporal" and not boost_recent:
                continue

            set_weight = keyword_set.weight
            max_possible_score += set_weight

            # Check for keyword matches
            matches = 0
            for keyword in keyword_set.keywords:
                if keyword.lower() in text_lower:
                    matches += 1

            # Calculate set score (diminishing returns for multiple matches)
            if matches > 0:
                set_score = set_weight * min(matches / 3.0, 1.0)  # Cap at 3 matches
                total_score += set_score

        return min(total_score / max_possible_score if max_possible_score > 0 else 0.0, 1.0)

    def extract_conflict_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract conflict indicators from text by category."""
        text_lower = text.lower()
        indicators = {}

        for keyword_set in self.keyword_sets.values():
            category_matches = []
            for keyword in keyword_set.keywords:
                if keyword.lower() in text_lower:
                    category_matches.append(keyword)

            if category_matches:
                indicators[keyword_set.category] = category_matches

        return indicators

    def update_keyword_performance(self, keyword: str, success: bool):
        """Update keyword performance statistics."""
        if success:
            self.performance_stats['successful_matches'] += 1

        # Could implement more sophisticated performance tracking here
        # e.g., adjust keyword weights based on success rates

    def get_trending_keywords(self, days: int = 7) -> List[str]:
        """Get trending keywords based on recent performance."""
        # Placeholder - in a real implementation, this would analyze
        # recent search results and news to identify trending terms

        # For now, return high-priority keywords
        return self.get_high_priority_keywords()[:10]

    def expand_query(self, base_query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand a base query with related keywords.

        Args:
            base_query: Original search query
            max_expansions: Maximum number of expanded queries

        Returns:
            List of expanded queries
        """
        expansions = [base_query]  # Include original

        # Find relevant keyword sets based on query content
        query_lower = base_query.lower()
        relevant_sets = []

        for keyword_set in self.keyword_sets.values():
            for keyword in keyword_set.keywords:
                if keyword.lower() in query_lower:
                    relevant_sets.append(keyword_set)
                    break

        # Add expansions using related keywords
        for keyword_set in relevant_sets[:2]:  # Limit to 2 sets
            for keyword in keyword_set.keywords[:2]:  # Top 2 keywords
                if len(expansions) >= max_expansions + 1:
                    break

                if keyword.lower() not in query_lower:
                    expanded = f"{base_query} {keyword}"
                    expansions.append(expanded)

        return expansions[:max_expansions + 1]

    def export_keywords(self, filepath: str):
        """Export keywords to JSON file."""
        export_data = {
            'keyword_sets': {},
            'performance_stats': self.performance_stats,
            'export_timestamp': datetime.now().isoformat()
        }

        for name, keyword_set in self.keyword_sets.items():
            export_data['keyword_sets'][name] = {
                'keywords': keyword_set.keywords,
                'weight': keyword_set.weight,
                'category': keyword_set.category,
                'source': keyword_set.source,
                'confidence': keyword_set.confidence
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Keywords exported to {filepath}")

    def import_keywords(self, filepath: str):
        """Import keywords from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            for name, set_data in data.get('keyword_sets', {}).items():
                keyword_set = KeywordSet(
                    name=name,
                    keywords=set_data['keywords'],
                    weight=set_data.get('weight', 1.0),
                    category=set_data.get('category', 'general'),
                    source=set_data.get('source', 'imported'),
                    confidence=set_data.get('confidence', 1.0)
                )
                self.add_keyword_set(keyword_set)

            self.logger.info(f"Keywords imported from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to import keywords: {e}")

    def get_stats(self) -> Dict:
        """Get keyword manager statistics."""
        total_keywords = sum(len(ks.keywords) for ks in self.keyword_sets.values())

        return {
            'total_keyword_sets': len(self.keyword_sets),
            'total_keywords': total_keywords,
            'performance_stats': self.performance_stats,
            'categories': list(set(ks.category for ks in self.keyword_sets.values())),
            'high_priority_count': len(self.get_high_priority_keywords())
        }
