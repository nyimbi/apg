"""
Horn of Africa Conflict Keywords
================================

Comprehensive keyword sets for conflict monitoring in Horn of Africa countries.
Includes conflict terms, location-specific keywords, and search query patterns.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CountryKeywords:
    """Keywords specific to a country."""
    country_name: str
    capital: str
    major_cities: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    ethnic_groups: List[str] = field(default_factory=list)
    political_entities: List[str] = field(default_factory=list)
    conflict_zones: List[str] = field(default_factory=list)


class HornOfAfricaKeywords:
    """Comprehensive keyword manager for Horn of Africa conflict monitoring."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_keywords()

    def _initialize_keywords(self):
        """Initialize all keyword sets."""

        # Core conflict terms
        self.conflict_terms = {
            'violence': [
                'violence', 'violent', 'attack', 'attacks', 'fighting', 'clash', 'clashes',
                'battle', 'battles', 'combat', 'conflict', 'conflicts', 'war', 'warfare',
                'armed conflict', 'civil war', 'insurgency', 'rebellion', 'uprising'
            ],
            'casualties': [
                'killed', 'death', 'deaths', 'casualties', 'wounded', 'injured', 'victim',
                'victims', 'fatalities', 'dead', 'massacre', 'slaughter', 'bloodshed'
            ],
            'displacement': [
                'displacement', 'displaced', 'refugee', 'refugees', 'IDP', 'IDPs',
                'internally displaced', 'flee', 'fled', 'fleeing', 'evacuation',
                'evacuated', 'migration', 'exodus'
            ],
            'security': [
                'security', 'insecurity', 'instability', 'unrest', 'riots', 'protest',
                'protests', 'demonstration', 'demonstrations', 'emergency', 'curfew',
                'martial law', 'state of emergency'
            ],
            'terrorism': [
                'terrorism', 'terrorist', 'terrorists', 'extremist', 'extremists',
                'militant', 'militants', 'al-shabab', 'al shabaab', 'jihadist',
                'jihadists', 'suicide bomb', 'bombing', 'explosion'
            ]
        }

        # Horn of Africa countries
        self.countries = {
            'somalia': CountryKeywords(
                country_name='Somalia',
                capital='Mogadishu',
                major_cities=['Hargeisa', 'Kismayo', 'Baidoa', 'Galkayo', 'Berbera', 'Bosaso', 'Garowe', 'Marka', 'Baraawe', 'Borama', 'Burao', 'Las Anod', 'Eyl', 'Qardho'],
                regions=['Puntland', 'Somaliland', 'Jubaland', 'Southwest State', 'Galmudug', 'Hirshabelle', 'Banadir', 'Awdal', 'Woqooyi Galbeed', 'Togdheer', 'Sanaag', 'Sool', 'Mudug', 'Nugaal', 'Bari', 'Galgaduud', 'Hiiraan', 'Middle Shabelle', 'Lower Shabelle', 'Bay', 'Bakool', 'Gedo', 'Middle Juba', 'Lower Juba'],
                ethnic_groups=['Somali', 'Bantu', 'Arab', 'Benadiri', 'Bajuni', 'Bravanese', 'Gibil Cad', 'Jareer', 'Reer Hamar', 'Ashraf', 'Madhiban', 'Tumal', 'Yibir', 'Gaboye'],
                political_entities=['Federal Government', 'Al-Shabaab', 'AMISOM'],
                conflict_zones=['Lower Shabelle', 'Middle Shabelle', 'Bay', 'Bakool', 'Gedo', 'Hiiraan', 'Mudug', 'Nugaal', 'Sool', 'Sanaag', 'Togdheer', 'Awdal', 'Balcad', 'Afgoye', 'Kurtunwaarey', 'Qoryoley', 'Marka', 'Baraawe', 'Jamaame', 'Dhobley', 'Luuq', 'Belet Weyne', 'Dhusamareb', 'Garowe', 'Borama', 'Zeila', 'Loyada']
            ),
            'ethiopia': CountryKeywords(
                country_name='Ethiopia',
                capital='Addis Ababa',
                major_cities=['Dire Dawa', 'Mekelle', 'Gondar', 'Hawassa', 'Bahir Dar', 'Dessie', 'Jimma', 'Jijiga', 'Shashamane', 'Nazret', 'Debre Markos', 'Harar', 'Dilla', 'Debre Berhan'],
                regions=['Tigray', 'Amhara', 'Oromia', 'Afar', 'Somali Region', 'Benishangul-Gumuz', 'SNNPR', 'Gambela', 'Harari', 'Addis Ababa', 'Dire Dawa', 'Sidama'],
                ethnic_groups=['Oromo', 'Amhara', 'Tigray', 'Sidama', 'Gurage', 'Welayta', 'Afar', 'Gamo', 'Gedeo', 'Silte', 'Kefficho', 'Hadiya', 'Somali', 'Agaw', 'Beta Israel', 'Harari', 'Kunama', 'Nuer', 'Anuak', 'Mursi'],
                political_entities=['EPRDF', 'TPLF', 'OLA', 'Federal Government'],
                conflict_zones=['Tigray', 'Oromia', 'Amhara', 'Benishangul-Gumuz', 'Gambela', 'SNNPR', 'Western Tigray', 'Wellega', 'Borena', 'Metekel', 'Assosa', 'Gedeo', 'West Guji', 'Jijiga', 'Moyale', 'Metema', 'Humera', 'Badme', 'Zalambessa', 'Bure', 'Dessie', 'Debre Markos', 'Nekemte', 'Jimma', 'Arba Minch', 'Weldiya']
            ),
            'eritrea': CountryKeywords(
                country_name='Eritrea',
                capital='Asmara',
                major_cities=['Keren', 'Massawa', 'Assab', 'Mendefera', 'Barentu', 'Adi Keih', 'Adi Quala', 'Dekemhare', 'Akordat', 'Tessenie', 'Ghinda', 'Nefasit', 'Senafe', 'Afabet'],
                regions=['Maekel', 'Debub', 'Gash-Barka', 'Anseba', 'Northern Red Sea', 'Southern Red Sea'],
                ethnic_groups=['Tigrinya', 'Tigre', 'Saho', 'Afar', 'Bilen', 'Hedareb', 'Kunama', 'Nara', 'Rashaida'],
                political_entities=['PFDJ', 'Government of Eritrea'],
                conflict_zones=['Border areas', 'Gash-Barka', 'Tesseney', 'Barentu', 'Agordat', 'Nakfa', 'Senafe', 'Zalambessa', 'Tsorona', 'Badme', 'Irob', 'Bada', 'Karora', 'Om Hajer', 'Forto']
            ),
            'djibouti': CountryKeywords(
                country_name='Djibouti',
                capital='Djibouti City',
                major_cities=['Ali Sabieh', 'Dikhil', 'Tadjourah', 'Obock', 'Arta', 'Holhol', 'Yoboki', 'As Eyla', 'Balho', 'Randa', 'Galafi', 'Loyada', 'Dewele', 'Khor Angar'],
                regions=['Djibouti Region', 'Ali Sabieh', 'Dikhil', 'Tadjourah', 'Obock', 'Arta'],
                ethnic_groups=['Somali', 'Afar', 'Arab', 'French', 'Issa', 'Gadabuursi', 'Ethiopian', 'Yemeni', 'Italian'],
                political_entities=['Government of Djibouti', 'FRUD'],
                conflict_zones=['Border areas', 'Ali Sabieh', 'Dikhil', 'Loyada', 'Galafi', 'Dewele', 'Holhol', 'Moulhoule', 'As Eyla', 'Randa', 'Khor Angar']
            ),
            'sudan': CountryKeywords(
                country_name='Sudan',
                capital='Khartoum',
                major_cities=['Omdurman', 'Port Sudan', 'Kassala', 'Gedaref', 'El Obeid', 'Nyala', 'El Fasher', 'Wad Medani', 'Al Qadarif', 'Kosti', 'Atbara', 'Dongola', 'Kadugli', 'Geneina'],
                regions=['Darfur', 'Kordofan', 'Blue Nile', 'White Nile', 'Red Sea', 'Kassala', 'Gedaref', 'Northern', 'River Nile', 'Sennar', 'West Darfur', 'North Darfur', 'South Darfur', 'Central Darfur', 'East Darfur', 'West Kordofan', 'North Kordofan', 'South Kordofan'],
                ethnic_groups=['Arab', 'Fur', 'Beja', 'Nuba', 'Fallata', 'Zaghawa', 'Masalit', 'Dinka', 'Shilluk', 'Nuer', 'Copts', 'Nubian', 'Hadendoa', 'Rashaida', 'Hausa', 'Darfuri', 'Funj', 'Ingassana'],
                political_entities=['SAF', 'RSF', 'TMC', 'Janjaweed', 'SLM'],
                conflict_zones=['Darfur', 'Blue Nile', 'South Kordofan', 'Kassala', 'West Darfur', 'North Darfur', 'South Darfur', 'Central Darfur', 'East Darfur', 'Abyei', 'Heglig', 'Kadugli', 'Damazin', 'Kurmuk', 'El Fasher', 'Nyala', 'Geneina', 'Zalingei', 'Ed Daein', 'Halayeb', 'Shalatin', 'Renk', 'Gallabat', 'Kusti']
            ),
            'south_sudan': CountryKeywords(
                country_name='South Sudan',
                capital='Juba',
                major_cities=['Wau', 'Malakal', 'Bentiu', 'Bor', 'Yei', 'Torit', 'Aweil', 'Kuacjok', 'Rumbek', 'Yambio', 'Kapoeta', 'Nasir', 'Akobo', 'Pochalla'],
                regions=['Upper Nile', 'Unity', 'Jonglei', 'Central Equatoria', 'Western Equatoria', 'Eastern Equatoria', 'Warrap', 'Northern Bahr el Ghazal', 'Western Bahr el Ghazal', 'Lakes'],
                ethnic_groups=['Dinka', 'Nuer', 'Shilluk', 'Azande', 'Bari', 'Kakwa', 'Kuku', 'Latuko', 'Acholi', 'Lotuho', 'Anyuak', 'Murle', 'Didinga', 'Toposa', 'Mundari', 'Pojulu', 'Avukaya', 'Moru'],
                political_entities=['SPLM', 'SPLM-IO', 'Government of South Sudan'],
                conflict_zones=['Unity', 'Upper Nile', 'Jonglei', 'Central Equatoria', 'Western Equatoria', 'Eastern Equatoria', 'Warrap', 'Northern Bahr el Ghazal', 'Western Bahr el Ghazal', 'Lakes', 'Abyei', 'Heglig', 'Panthou', 'Pibor', 'Nasir', 'Kodok', 'Pariang', 'Mayom', 'Rubkona', 'Koch', 'Leer', 'Mayendit', 'Panyijiar', 'Renk', 'Nimule', 'Kaya']
            ),
            'kenya': CountryKeywords(
                country_name='Kenya',
                capital='Nairobi',
                major_cities=['Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Meru', 'Thika', 'Kitale', 'Malindi', 'Garissa', 'Kakamega', 'Machakos', 'Nyeri', 'Kericho', 'Embu'],
                regions=['Coast', 'North Eastern', 'Eastern', 'Central', 'Rift Valley', 'Western', 'Nyanza', 'Nairobi'],
                ethnic_groups=['Kikuyu', 'Luhya', 'Luo', 'Kalenjin', 'Kamba', 'Kisii', 'Meru', 'Somali', 'Mijikenda', 'Maasai', 'Turkana', 'Embu', 'Taita', 'Kuria', 'Samburu', 'Teso', 'Borana', 'Galla', 'Pokot', 'Rendille'],
                political_entities=['Government of Kenya', 'ODM', 'Jubilee Party', 'UDA'],
                conflict_zones=['Mandera', 'Wajir', 'Garissa', 'Lamu', 'Tana River', 'Marsabit', 'Turkana', 'West Pokot', 'Samburu', 'Laikipia', 'Isiolo', 'Baringo', 'Elgeyo-Marakwet', 'Kapedo', 'Baragoi', 'Moyale', 'Liboi', 'Dadaab', 'Mandera Triangle', 'Banisa', 'Rhamu', 'El Wak', 'Takaba', 'Arabia', 'Bulla Hawa', 'Lokichoggio']
            ),
            'uganda': CountryKeywords(
                country_name='Uganda',
                capital='Kampala',
                major_cities=['Gulu', 'Lira', 'Mbarara', 'Jinja', 'Mbale', 'Fort Portal', 'Masaka', 'Kasese', 'Soroti', 'Arua', 'Kabale', 'Hoima', 'Moroto', 'Kitgum'],
                regions=['Northern', 'Eastern', 'Central', 'Western', 'West Nile', 'Karamoja', 'Buganda', 'Bunyoro', 'Toro', 'Ankole', 'Kigezi', 'Busoga', 'Bukedi', 'Teso', 'Lango', 'Acholi', 'Sebei', 'Rwenzori'],
                ethnic_groups=['Baganda', 'Banyankole', 'Basoga', 'Bakiga', 'Iteso', 'Langi', 'Acholi', 'Bagisu', 'Lugbara', 'Bunyoro', 'Batoro', 'Alur', 'Bagwere', 'Bakonjo', 'Jopadhola', 'Karimojong', 'Pokot', 'Sebei', 'Banyarwanda', 'Batwa'],
                political_entities=['NRM', 'Government of Uganda', 'FDC', 'NUP'],
                conflict_zones=['Karamoja', 'Northern Uganda', 'Rwenzori', 'West Nile', 'Acholi', 'Lango', 'Teso', 'Sebei', 'Bundibugyo', 'Kasese', 'Ntoroko', 'Kitgum', 'Pader', 'Amuru', 'Nwoya', 'Lamwo', 'Agago', 'Kotido', 'Kaabong', 'Abim', 'Moroto', 'Nakapiripirit', 'Amudat', 'Napak', 'Karenga', 'Busia', 'Malaba']
            ),
            'rwanda': CountryKeywords(
                country_name='Rwanda',
                capital='Kigali',
                major_cities=['Butare', 'Gitarama', 'Ruhengeri', 'Gisenyi', 'Cyangugu', 'Byumba', 'Kibungo', 'Kibuye', 'Nyagatare', 'Musanze', 'Rubavu', 'Rusizi', 'Muhanga', 'Huye'],
                regions=['Northern Province', 'Southern Province', 'Eastern Province', 'Western Province', 'Kigali City'],
                ethnic_groups=['Hutu', 'Tutsi', 'Twa'],
                political_entities=['RPF', 'Government of Rwanda', 'PSD', 'PL'],
                conflict_zones=['Western Province', 'Nyungwe Forest', 'Border areas', 'Cyangugu', 'Gisenyi', 'Ruhengeri', 'Butare', 'Byumba', 'Kibungo', 'Gikongoro', 'Kibuye', 'Umutara', 'Volcanoes National Park', 'Akagera National Park', 'Rusizi', 'Rubavu', 'Musanze', 'Nyagatare', 'Kirehe', 'Gatsibo', 'Kayonza']
            ),
            'burundi': CountryKeywords(
                country_name='Burundi',
                capital='Gitega',
                major_cities=['Bujumbura', 'Muyinga', 'Ruyigi', 'Kayanza', 'Ngozi', 'Bururi', 'Rutana', 'Makamba', 'Cankuzo', 'Karuzi', 'Kirundo', 'Muramvya', 'Mwaro', 'Bubanza'],
                regions=['Bujumbura Mairie', 'Bujumbura Rural', 'Bubanza', 'Cibitoke', 'Gitega', 'Kayanza', 'Kirundo', 'Makamba', 'Muramvya', 'Muyinga', 'Mwaro', 'Ngozi', 'Rutana', 'Ruyigi', 'Bururi', 'Cankuzo', 'Karuzi', 'Rumonge'],
                ethnic_groups=['Hutu', 'Tutsi', 'Twa', 'Ganwa'],
                political_entities=['CNDD-FDD', 'Government of Burundi', 'UPRONA', 'FNL'],
                conflict_zones=['Bujumbura', 'Cibitoke', 'Bubanza', 'Kayanza', 'Kirundo', 'Ngozi', 'Muyinga', 'Cankuzo', 'Ruyigi', 'Rutana', 'Bururi', 'Makamba', 'Rumonge', 'Bujumbura Rural', 'Gitega', 'Karuzi', 'Mwaro', 'Muramvya', 'Kobero', 'Gasenyi', 'Nemba', 'Kanyaru Haut', 'Muyinga-Gassarara', 'Busoni', 'Mabayi']
            ),
            'drc': CountryKeywords(
                country_name='Democratic Republic of Congo',
                capital='Kinshasa',
                major_cities=['Lubumbashi', 'Mbuji-Mayi', 'Kisangani', 'Kananga', 'Goma', 'Bukavu', 'Likasi', 'Kolwezi', 'Tshikapa', 'Beni', 'Butembo', 'Mwene-Ditu', 'Kikwit', 'Mbandaka'],
                regions=['Kinshasa', 'Kongo Central', 'Kwilu', 'Kwango', 'Mai-Ndombe', 'Kasai', 'Kasai-Central', 'Kasai-Oriental', 'Sankuru', 'Maniema', 'South Kivu', 'North Kivu', 'Ituri', 'Haut-Uele', 'Bas-Uele', 'Tshopo', 'Mongala', 'Nord-Ubangi', 'Sud-Ubangi', 'Equateur', 'Tshuapa', 'Lomami', 'Haut-Lomami', 'Lualaba', 'Haut-Katanga', 'Tanganyika'],
                ethnic_groups=['Luba', 'Mongo', 'Kongo', 'Mangbetu-Azande', 'Bantu', 'Nilotic', 'Sudanic', 'Shi', 'Nande', 'Nyanga', 'Havu', 'Hunde', 'Tembo', 'Lega', 'Fulero', 'Vira', 'Bembe', 'Banyamulenge', 'Tutsi', 'Hutu', 'Twa', 'Lendu', 'Hema', 'Alur', 'Lugbara', 'Kakwa'],
                political_entities=['Government of DRC', 'FARDC', 'MONUSCO', 'M23', 'ADF', 'FDLR'],
                conflict_zones=['North Kivu', 'South Kivu', 'Ituri', 'Kasai', 'Tanganyika', 'Maniema', 'Haut-Katanga', 'Lualaba', 'Haut-Lomami', 'Lomami', 'Sankuru', 'Kasai-Oriental', 'Kasai-Central', 'Tshopo', 'Bas-Uele', 'Haut-Uele', 'Equateur', 'Mai-Ndombe', 'Kwilu', 'Kwango', 'Kongo Central', 'Beni', 'Butembo', 'Rutshuru', 'Masisi', 'Walikale', 'Lubero', 'Nyiragongo', 'Uvira', 'Fizi', 'Shabunda', 'Kalehe', 'Walungu', 'Kabare', 'Mwenga', 'Bunyakiri', 'Minova', 'Mahagi', 'Djugu', 'Irumu', 'Aru', 'Bunia', 'Bunagana', 'Ishasha', 'Kasindi', 'Nobili', 'Vurondo']
            )
        }

        # Resource-related conflict terms
        self.resource_terms = [
            'water', 'drought', 'famine', 'food security', 'pasture', 'grazing',
            'livestock', 'cattle', 'nomadic', 'pastoralist', 'farmer', 'agriculture',
            'land dispute', 'boundary dispute', 'resource conflict'
        ]

        # International actors
        self.international_actors = [
            'UN', 'United Nations', 'UNHCR', 'UNICEF', 'WFP', 'WHO',
            'African Union', 'AU', 'IGAD', 'AMISOM', 'EU', 'European Union',
            'US', 'United States', 'China', 'Russia', 'Turkey', 'Egypt',
            'Saudi Arabia', 'UAE', 'Iran'
        ]

        # Time-sensitive terms
        self.temporal_terms = [
            'today', 'yesterday', 'this week', 'last week', 'recent', 'latest',
            'breaking', 'urgent', 'alert', 'developing', 'ongoing', 'current'
        ]

    def get_conflict_keywords(self, severity_level: str = 'all') -> List[str]:
        """
        Get conflict keywords by severity level.

        Args:
            severity_level: 'high', 'medium', 'low', or 'all'

        Returns:
            List of relevant conflict keywords
        """
        if severity_level == 'high':
            return (self.conflict_terms['violence'] +
                   self.conflict_terms['casualties'] +
                   self.conflict_terms['terrorism'])
        elif severity_level == 'medium':
            return (self.conflict_terms['displacement'] +
                   self.conflict_terms['security'])
        elif severity_level == 'low':
            return self.resource_terms
        else:
            # Return all conflict terms
            all_terms = []
            for term_list in self.conflict_terms.values():
                all_terms.extend(term_list)
            all_terms.extend(self.resource_terms)
            return all_terms

    def get_country_keywords(self, country: str) -> List[str]:
        """Get all keywords for a specific country."""
        if country.lower() not in self.countries:
            self.logger.warning(f"Country '{country}' not found in keyword database")
            return []

        country_data = self.countries[country.lower()]
        keywords = [country_data.country_name, country_data.capital]
        keywords.extend(country_data.major_cities)
        keywords.extend(country_data.regions)
        keywords.extend(country_data.ethnic_groups)
        keywords.extend(country_data.political_entities)
        keywords.extend(country_data.conflict_zones)

        return keywords

    def get_all_location_keywords(self) -> List[str]:
        """Get all location-related keywords for Horn of Africa."""
        all_locations = []
        for country in self.countries.values():
            all_locations.extend(self.get_country_keywords(country.country_name))
        return list(set(all_locations))  # Remove duplicates

    def generate_search_queries(
        self,
        countries: List[str] = None,
        conflict_level: str = 'all',
        include_temporal: bool = True,
        max_queries: int = 50
    ) -> List[str]:
        """
        Generate search queries for conflict monitoring.

        Args:
            countries: List of countries to focus on (None for all)
            conflict_level: Severity level of conflicts to search for
            include_temporal: Include time-sensitive terms
            max_queries: Maximum number of queries to generate

        Returns:
            List of optimized search queries
        """
        if countries is None:
            countries = list(self.countries.keys())

        conflict_keywords = self.get_conflict_keywords(conflict_level)
        queries = []

        # Generate country + conflict term combinations
        for country in countries:
            country_keywords = self.get_country_keywords(country)

            # Main country name + conflict terms
            for conflict_term in conflict_keywords[:10]:  # Limit to top conflict terms
                query = f'"{self.countries[country].country_name}" {conflict_term}'
                if include_temporal:
                    # Add recent temporal modifier to some queries
                    if len(queries) % 3 == 0 and self.temporal_terms:
                        query = f"{self.temporal_terms[0]} {query}"
                queries.append(query)

                if len(queries) >= max_queries:
                    break

            if len(queries) >= max_queries:
                break

            # Major cities + conflict terms
            for city in country_keywords[2:5]:  # Take first 3 major cities
                for conflict_term in conflict_keywords[:5]:  # Top 5 conflict terms
                    queries.append(f'"{city}" {conflict_term}')
                    if len(queries) >= max_queries:
                        break
                if len(queries) >= max_queries:
                    break

        # Add regional Horn of Africa queries
        horn_queries = [
            '"Horn of Africa" conflict',
            '"Horn of Africa" violence',
            '"East Africa" security crisis',
            '"Horn of Africa" humanitarian crisis'
        ]
        queries.extend(horn_queries[:max_queries - len(queries)])

        return queries[:max_queries]

    def get_priority_keywords(self) -> Dict[str, List[str]]:
        """Get high-priority keywords for real-time monitoring."""
        return {
            'urgent_conflict': [
                'breaking', 'urgent', 'attack', 'bombing', 'killed', 'massacre',
                'al-shabaab', 'emergency', 'evacuation'
            ],
            'displacement': [
                'refugee', 'displaced', 'flee', 'fleeing', 'IDP', 'camp'
            ],
            'political': [
                'coup', 'election', 'government', 'president', 'prime minister',
                'parliament', 'protest', 'demonstration'
            ],
            'international': [
                'UN', 'African Union', 'IGAD', 'peacekeeping', 'humanitarian aid',
                'sanctions', 'intervention'
            ]
        }

    def score_keyword_relevance(self, text: str, country: str = None) -> float:
        """
        Score the relevance of text based on keyword matches.

        Args:
            text: Text to analyze
            country: Specific country context (optional)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        text_lower = text.lower()
        score = 0.0
        max_score = 0.0

        # Score conflict terms
        for category, terms in self.conflict_terms.items():
            category_weight = {
                'violence': 0.3,
                'casualties': 0.25,
                'terrorism': 0.25,
                'displacement': 0.15,
                'security': 0.05
            }.get(category, 0.1)

            for term in terms:
                max_score += category_weight
                if term in text_lower:
                    score += category_weight

        # Score location relevance
        if country:
            country_keywords = self.get_country_keywords(country)
            location_weight = 0.2
            max_score += location_weight

            for keyword in country_keywords:
                if keyword.lower() in text_lower:
                    score += location_weight / len(country_keywords)

        # Score temporal relevance
        temporal_weight = 0.1
        max_score += temporal_weight
        for term in self.temporal_terms:
            if term in text_lower:
                score += temporal_weight
                break

        return min(score / max_score if max_score > 0 else 0.0, 1.0)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract Horn of Africa entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with entity types and found entities
        """
        text_lower = text.lower()
        entities = {
            'countries': [],
            'cities': [],
            'regions': [],
            'ethnic_groups': [],
            'political_entities': [],
            'conflict_zones': []
        }

        for country, data in self.countries.items():
            # Check country name
            if data.country_name.lower() in text_lower:
                entities['countries'].append(data.country_name)

            # Check cities
            for city in data.major_cities:
                if city.lower() in text_lower:
                    entities['cities'].append(city)

            # Check regions
            for region in data.regions:
                if region.lower() in text_lower:
                    entities['regions'].append(region)

            # Check ethnic groups
            for group in data.ethnic_groups:
                if group.lower() in text_lower:
                    entities['ethnic_groups'].append(group)

            # Check political entities
            for entity in data.political_entities:
                if entity.lower() in text_lower:
                    entities['political_entities'].append(entity)

            # Check conflict zones
            for zone in data.conflict_zones:
                if zone.lower() in text_lower:
                    entities['conflict_zones'].append(zone)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities
