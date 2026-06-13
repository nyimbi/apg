"""
USSD Menu DSL — declarative menu tree builder with I18N and 180-char rendering.

Design:
  - UsMenu is a pure dataclass-style Pydantic model
  - MenuBuilder provides a fluent builder for constructing trees
  - render(menu, language, variables) → str (truncated to 180 chars)
  - validate_input(menu, input) → bool
  - I18N: English (en), Swahili (sw), Amharic (am), French (fr)
  - 'Us' model prefix throughout
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

_log = logging.getLogger(__name__)

USSD_MAX_CHARS: int = 180       # safe GSM/USSD payload limit
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "sw", "am", "fr"})

# Built-in system phrase translations used in menu scaffolding
_SYSTEM_PHRASES: dict[str, dict[str, str]] = {
	"back": {"en": "0. Back", "sw": "0. Rudi", "am": "0. ተመለስ", "fr": "0. Retour"},
	"exit": {"en": "00. Exit", "sw": "00. Toka", "am": "00. ውጣ", "fr": "00. Quitter"},
	"invalid": {
		"en": "Invalid option. Try again.",
		"sw": "Chaguo batili. Jaribu tena.",
		"am": "ልክ ያልሆነ አማራጭ። እንደገና ይሞክሩ።",
		"fr": "Option invalide. Réessayez.",
	},
	"welcome": {
		"en": "Welcome",
		"sw": "Karibu",
		"am": "እንኳን ደህና መጡ",
		"fr": "Bienvenue",
	},
}


# ── Pydantic models ──────────────────────────────────────────────────────────

class UsMenuOption(BaseModel):
	"""Single selectable option inside a UsMenu."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	key: str                           # numeric key shown to user ("1", "2", …)
	label: str                         # display label in default language
	labels: dict[str, str] = Field(default_factory=dict)  # lang → translated label
	action: str = "navigate"           # navigate | execute | end | back | input
	target: str | None = None          # menu_id for navigate; var name for input
	handler: str | None = None         # callable reference name for execute
	condition: str | None = None       # simple condition expression


class UsMenu(BaseModel):
	"""A USSD screen/menu node."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	menu_id: str = Field(default_factory=uuid7str)
	title: str
	titles: dict[str, str] = Field(default_factory=dict)   # lang → translated title
	body: str = ""
	bodies: dict[str, str] = Field(default_factory=dict)   # lang → translated body
	options: list[UsMenuOption] = Field(default_factory=list)
	is_terminal: bool = False          # True → END (no further navigation)
	show_back: bool = False            # append back option automatically
	show_exit: bool = False            # append exit/quit option automatically
	metadata: dict[str, Any] = Field(default_factory=dict)


class UsMenuTree(BaseModel):
	"""Complete menu tree for a service code."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	service_code: str
	root_menu_id: str
	menus: dict[str, UsMenu] = Field(default_factory=dict)  # menu_id → UsMenu
	default_language: str = "en"
	supported_languages: list[str] = Field(default_factory=lambda: ["en"])
	metadata: dict[str, Any] = Field(default_factory=dict)


# ── Render engine ────────────────────────────────────────────────────────────

def _resolve_text(base: str, translations: dict[str, str], language: str) -> str:
	"""Return translated text or fall back to English then base."""
	if language in translations:
		return translations[language]
	if "en" in translations:
		return translations["en"]
	return base


def _substitute_variables(text: str, variables: dict[str, Any]) -> str:
	"""Replace {var_name} placeholders with session variable values."""
	for key, value in variables.items():
		text = text.replace(f"{{{key}}}", str(value))
	return text


def render(
	menu: UsMenu,
	language: str = "en",
	variables: dict[str, Any] | None = None,
	*,
	max_chars: int = USSD_MAX_CHARS,
) -> str:
	"""
	Render a UsMenu to a USSD-safe string.

	Applies translation, variable substitution, automatic back/exit injection,
	and hard truncation to max_chars (default 180).

	Args:
		menu: The menu node to render.
		language: BCP-47 language code. Falls back to 'en' → base text.
		variables: Session variables for placeholder substitution.
		max_chars: Hard character limit (default 180).

	Returns:
		Rendered string ready to send to subscriber.
	"""
	lang = language if language in SUPPORTED_LANGUAGES else "en"
	vars_ = variables or {}

	title = _resolve_text(menu.title, menu.titles, lang)
	title = _substitute_variables(title, vars_)

	body = _resolve_text(menu.body, menu.bodies, lang)
	body = _substitute_variables(body, vars_)

	lines: list[str] = []
	if title:
		lines.append(title)
	if body:
		lines.append(body)

	for opt in menu.options:
		cond = opt.condition
		if cond and not _eval_condition(cond, vars_):
			continue
		label = _resolve_text(opt.label, opt.labels, lang)
		label = _substitute_variables(label, vars_)
		lines.append(f"{opt.key}. {label}")

	if menu.show_back:
		lines.append(_SYSTEM_PHRASES["back"].get(lang, _SYSTEM_PHRASES["back"]["en"]))
	if menu.show_exit:
		lines.append(_SYSTEM_PHRASES["exit"].get(lang, _SYSTEM_PHRASES["exit"]["en"]))

	rendered = "\n".join(lines)

	if len(rendered) > max_chars:
		_log.debug(
			"menu '%s' rendered %d chars — truncating to %d",
			menu.menu_id, len(rendered), max_chars,
		)
		rendered = rendered[:max_chars]

	return rendered


def validate_input(menu: UsMenu, input_text: str) -> bool:
	"""
	Return True if input_text matches a valid option key on the menu
	(including reserved back "0" and exit "00" if enabled).
	"""
	valid_keys: set[str] = {opt.key for opt in menu.options}
	if menu.show_back:
		valid_keys.add("0")
	if menu.show_exit:
		valid_keys.add("00")
	return input_text.strip() in valid_keys


def _eval_condition(condition: str, variables: dict[str, Any]) -> bool:
	"""Evaluate a simple comparison condition against variables dict."""
	try:
		m = re.match(r"(\w+)\s*(==|!=|>|<|>=|<=)\s*(.+)", condition.strip())
		if not m:
			return True
		key, op, rhs = m.group(1), m.group(2), m.group(3).strip().strip("'\"")
		lhs = str(variables.get(key, ""))
		ops: dict[str, bool] = {
			"==": lhs == rhs, "!=": lhs != rhs,
			">": lhs > rhs, "<": lhs < rhs,
			">=": lhs >= rhs, "<=": lhs <= rhs,
		}
		return ops.get(op, True)
	except Exception as exc:
		_log.debug("condition eval error '%s': %s", condition, exc)
		return True


# ── Fluent builder ────────────────────────────────────────────────────────────

class MenuBuilder:
	"""
	Fluent builder for constructing UsMenu and UsMenuTree objects.

	Example::

		tree = (
			MenuBuilder("*123#")
			.menu("main", "Main Menu")
			    .option("1", "Check Balance", target="balance")
			    .option("2", "Send Money", target="send_money")
			    .show_back(False).show_exit()
			.menu("balance", "Your Balance", is_terminal=True)
			    .body("Balance: {balance}")
			.build()
		)
	"""

	def __init__(self, service_code: str, default_language: str = "en") -> None:
		self._service_code = service_code
		self._default_language = default_language
		self._menus: dict[str, UsMenu] = {}
		self._root_menu_id: str | None = None
		self._current: UsMenu | None = None
		self._supported_languages: list[str] = [default_language]

	# ── Menu operations ───────────────────────────────────────────────────────

	def menu(
		self,
		menu_id: str,
		title: str,
		*,
		is_terminal: bool = False,
	) -> "MenuBuilder":
		"""Add a new menu and make it the current context."""
		m = UsMenu(menu_id=menu_id, title=title, is_terminal=is_terminal)
		self._menus[menu_id] = m
		self._current = m
		if self._root_menu_id is None:
			self._root_menu_id = menu_id
		return self

	def body(self, text: str) -> "MenuBuilder":
		"""Set body text for the current menu."""
		assert self._current is not None, "call .menu() first"
		self._current.body = text
		return self

	def translate_title(self, language: str, text: str) -> "MenuBuilder":
		"""Add a translated title for the current menu."""
		assert self._current is not None, "call .menu() first"
		self._current.titles[language] = text
		if language not in self._supported_languages:
			self._supported_languages.append(language)
		return self

	def translate_body(self, language: str, text: str) -> "MenuBuilder":
		"""Add a translated body for the current menu."""
		assert self._current is not None, "call .menu() first"
		self._current.bodies[language] = text
		return self

	def show_back(self, enabled: bool = True) -> "MenuBuilder":
		"""Toggle automatic back option on the current menu."""
		assert self._current is not None, "call .menu() first"
		self._current.show_back = enabled
		return self

	def show_exit(self, enabled: bool = True) -> "MenuBuilder":
		"""Toggle automatic exit option on the current menu."""
		assert self._current is not None, "call .menu() first"
		self._current.show_exit = enabled
		return self

	# ── Option operations ─────────────────────────────────────────────────────

	def option(
		self,
		key: str,
		label: str,
		*,
		action: str = "navigate",
		target: str | None = None,
		handler: str | None = None,
		condition: str | None = None,
		translations: dict[str, str] | None = None,
	) -> "MenuBuilder":
		"""Append an option to the current menu."""
		assert self._current is not None, "call .menu() first"
		opt = UsMenuOption(
			key=key,
			label=label,
			labels=dict(translations or {}),
			action=action,
			target=target,
			handler=handler,
			condition=condition,
		)
		self._current.options.append(opt)
		return self

	def navigate(self, key: str, label: str, target: str, **kw: Any) -> "MenuBuilder":
		"""Shortcut: add a navigate option."""
		return self.option(key, label, action="navigate", target=target, **kw)

	def execute(self, key: str, label: str, handler: str, **kw: Any) -> "MenuBuilder":
		"""Shortcut: add an execute option."""
		return self.option(key, label, action="execute", handler=handler, **kw)

	def end(self, key: str, label: str, **kw: Any) -> "MenuBuilder":
		"""Shortcut: add a terminal option."""
		return self.option(key, label, action="end", **kw)

	def input_field(self, key: str, label: str, variable: str, handler: str | None = None) -> "MenuBuilder":
		"""Shortcut: add a free-text input option."""
		return self.option(key, label, action="input", target=variable, handler=handler)

	# ── Build ─────────────────────────────────────────────────────────────────

	def build(self) -> UsMenuTree:
		"""Finalise and return the complete UsMenuTree."""
		assert self._root_menu_id is not None, "at least one menu is required"
		return UsMenuTree(
			service_code=self._service_code,
			root_menu_id=self._root_menu_id,
			menus=dict(self._menus),
			default_language=self._default_language,
			supported_languages=list(self._supported_languages),
		)


# ── I18N helpers ──────────────────────────────────────────────────────────────

def system_phrase(key: str, language: str = "en") -> str:
	"""Return a built-in system phrase (back, exit, invalid, welcome) in the given language."""
	phrases = _SYSTEM_PHRASES.get(key, {})
	return phrases.get(language, phrases.get("en", key))


def add_translations(
	menu: UsMenu,
	language: str,
	*,
	title: str | None = None,
	body: str | None = None,
	option_labels: dict[str, str] | None = None,  # key → translated label
) -> UsMenu:
	"""
	Attach translations to an existing UsMenu in-place and return it.

	Args:
		menu: Menu to annotate.
		language: Target language code (en/sw/am/fr).
		title: Translated title string.
		body: Translated body string.
		option_labels: Mapping of option key → translated label.
	"""
	if language not in SUPPORTED_LANGUAGES:
		_log.warning("Language '%s' not in SUPPORTED_LANGUAGES %s", language, SUPPORTED_LANGUAGES)
	if title is not None:
		menu.titles[language] = title
	if body is not None:
		menu.bodies[language] = body
	if option_labels:
		for opt in menu.options:
			if opt.key in option_labels:
				opt.labels[language] = option_labels[opt.key]
	return menu
