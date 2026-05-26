"""Regression coverage for NLPC language-code support."""

from ..models import LanguageCode
from .. import CAPABILITY_INFO


AFRICAN_LANGUAGE_CODES = {
	"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr",
	"ki", "rw", "rn", "kg", "ln", "lg", "mg", "ny", "om", "sg",
	"sn", "so", "st", "sw", "ss", "ti", "ts", "tn", "tw", "ve",
	"wo", "xh", "yo", "zu", "kab", "kam", "luo", "mas", "mer",
	"mos", "nus", "suk", "tzm", "tig", "umb"
}


def test_language_code_enum_includes_at_least_40_african_languages():
	enum_values = {language.value for language in LanguageCode}

	assert len(AFRICAN_LANGUAGE_CODES) >= 40
	assert AFRICAN_LANGUAGE_CODES <= enum_values


def test_capability_metadata_exposes_african_language_codes():
	supported_languages = set(CAPABILITY_INFO["supported_languages"])

	assert AFRICAN_LANGUAGE_CODES <= supported_languages
