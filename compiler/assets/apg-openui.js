/*
 * OpenUI Lang browser renderer.
 *
 * Pairs with compiler/openui_lang/render.py: the same component-to-class
 * mapping, so a screen painted by the server and one streamed into the browser
 * look identical. No framework, no build step, no CDN -- this file is vendored
 * into the generated app's static/ directory beside apg-sse.js.
 *
 * Usage, from a generated page:
 *
 *   var view = APGOpenUI.mount(document.getElementById("screen"), library);
 *   view.push(chunk);          // any number of times, as tokens arrive
 *   view.end();                // the host signals the end of the stream
 *
 * The parser is a port of compiler/openui_lang/{source,parser,evaluator}.py at
 * the level the browser needs: statements, expressions, the built-ins, actions,
 * and the recovery rules of spec section 8. It renders from the pending tail
 * after implicit closing, which is what makes a partial screen appear and grow.
 */
(function (global) {
	"use strict";

	// `Action(...)` is handled before this set is consulted, since it is an
	// action *list* rather than a statement that renders nothing.
	var RESERVED_CALL_HEADS = { Query: true, Mutation: true };
	var ACTION_NAMES = { Set: true, Reset: true, Run: true, ToAssistant: true, OpenUrl: true };
	var BUILTINS = {
		Count: true, First: true, Last: true, Sum: true, Avg: true, Min: true, Max: true,
		Filter: true, Sort: true, Round: true, Abs: true, Floor: true, Ceil: true, Each: true
	};

	// ── values (spec 3.1) ───────────────────────────────────────────────────

	function isNumber(value) {
		return typeof value === "number" && !isNaN(value);
	}

	function toNumber(value) {
		if (value === true) return 1;
		if (value === false || value === null || value === undefined) return 0;
		if (typeof value === "number") return isNaN(value) ? 0 : value;
		if (typeof value === "string") {
			var text = value.trim();
			if (text === "") return 0;
			var number = Number(text);
			return isNaN(number) ? 0 : number;
		}
		if (Array.isArray(value)) return toNumber(value.map(function (item) { return item === null ? "" : toString(item); }).join(","));
		return 0;
	}

	function toString(value) {
		if (value === null || value === undefined) return "null";
		if (value === true) return "true";
		if (value === false) return "false";
		if (typeof value === "number") return String(value);
		if (typeof value === "string") return value;
		if (Array.isArray(value)) return value.map(function (item) { return item === null ? "" : toString(item); }).join(",");
		return "[object Object]";
	}

	function toBoolean(value) {
		if (value === null || value === undefined || value === false) return false;
		if (value === true) return true;
		if (typeof value === "number") return value !== 0 && !isNaN(value);
		if (typeof value === "string") return value !== "";
		return true;
	}

	function looseEqual(left, right) {
		if (left === null || left === undefined) return right === null || right === undefined;
		if (right === null || right === undefined) return false;
		if (typeof left === typeof right && (typeof left === "string" || typeof left === "boolean")) return left === right;
		if (typeof left === "boolean") return looseEqual(toNumber(left), right);
		if (typeof right === "boolean") return looseEqual(left, toNumber(right));
		if (typeof left === "number" && typeof right === "number") return left === right;
		if (typeof left === "number" || typeof right === "number") return toNumber(left) === toNumber(right);
		if (typeof left === "string" && typeof right === "string") return left === right;
		if (Array.isArray(left) || typeof left === "object") return looseEqual(toString(left), right);
		if (Array.isArray(right) || typeof right === "object") return looseEqual(left, toString(right));
		return left === right;
	}

	function add(left, right) {
		if (typeof left === "string" || typeof right === "string") {
			return (left === null || left === undefined ? "" : toString(left)) +
				(right === null || right === undefined ? "" : toString(right));
		}
		return toNumber(left) + toNumber(right);
	}

	function divide(left, right) {
		var divisor = toNumber(right);
		return divisor === 0 ? 0 : toNumber(left) / divisor;
	}

	function modulo(left, right) {
		var divisor = toNumber(right);
		return divisor === 0 ? 0 : toNumber(left) % divisor;
	}

	function compare(left, right, op) {
		var a = toNumber(left), b = toNumber(right);
		if (op === ">") return a > b;
		if (op === "<") return a < b;
		if (op === ">=") return a >= b;
		return a <= b;
	}

	function pluckField(item, field) {
		// `a.b` where `a` is an array element, including a digits field (3.3).
		if (Array.isArray(item)) {
			if (/^\d+$/.test(field)) {
				var index = Number(field);
				return index < item.length ? item[index] : null;
			}
			return item.map(function (element) { return pluckField(element, field); });
		}
		if (item && typeof item === "object") return Object.prototype.hasOwnProperty.call(item, field) ? item[field] : null;
		return null;
	}

	function memberOf(target, field) {
		// `a.b` where `a` is the whole array: the field is plucked from each
		// element, so `m.0` on a list of lists is the first column, not the first
		// row (grammar/009).
		if (Array.isArray(target)) {
			if (field === "length") return target.length;
			return target.map(function (element) { return pluckField(element, field); });
		}
		if (target && typeof target === "object") return Object.prototype.hasOwnProperty.call(target, field) ? target[field] : null;
		return null;
	}

	// ── source preparation (spec 1.2, 1.4) ──────────────────────────────────

	function stripMarkers(text) {
		return text.split("\n").filter(function (line) { return line.indexOf("]]>openui:") !== 0; }).join("\n");
	}

	function extractFences(text) {
		var lines = text.split("\n"), fences = [], current = null;
		lines.forEach(function (line) {
			if (line.indexOf("```") === 0) {
				if (current === null) { current = []; } else { fences.push(current.join("\n")); current = null; }
				return;
			}
			if (current !== null) current.push(line);
		});
		if (current !== null) fences.push(current.join("\n"));
		return fences.length ? fences.join("\n") : text;
	}

	function stripComments(text) {
		var output = "", quote = "", index = 0;
		while (index < text.length) {
			var char = text[index];
			if (quote) {
				output += char;
				if (char === "\\" && index + 1 < text.length) { output += text[index + 1]; index += 2; continue; }
				if (char === quote) quote = "";
				index += 1;
				continue;
			}
			if (char === "\"" || char === "'") { quote = char; output += char; index += 1; continue; }
			if (char === "#" || (char === "/" && text[index + 1] === "/")) {
				while (index < text.length && text[index] !== "\n") index += 1;
				continue;
			}
			output += char;
			index += 1;
		}
		return output;
	}

	function prepare(text) {
		return stripComments(extractFences(stripMarkers(text))).trim();
	}

	function endsIncomplete(text) {
		var depth = 0, quote = "", index = 0, ternaryOpen = false, ternaryColon = false;
		while (index < text.length) {
			var char = text[index];
			if (quote) {
				if (char === "\\") { index += 2; continue; }
				if (char === quote) quote = "";
				index += 1;
				continue;
			}
			if (char === "\"" || char === "'") { quote = char; index += 1; continue; }
			if ("([{".indexOf(char) >= 0) depth += 1;
			else if (")]}".indexOf(char) >= 0) depth = Math.max(0, depth - 1);
			else if (char === "?" && depth === 0) { ternaryOpen = true; ternaryColon = false; }
			else if (char === ":" && depth === 0 && ternaryOpen) ternaryColon = true;
			index += 1;
		}
		return !!quote || depth > 0 || (ternaryOpen && !ternaryColon);
	}

	function scanLine(line, state) {
		var index = 0;
		while (index < line.length) {
			var char = line[index];
			if (state.quote) {
				if (char === "\\") { index += 2; continue; }
				if (char === state.quote) state.quote = "";
				index += 1;
				continue;
			}
			if (char === "\"" || char === "'") state.quote = char;
			else if ("([{".indexOf(char) >= 0) state.depth += 1;
			else if (")]}".indexOf(char) >= 0) state.depth = Math.max(0, state.depth - 1);
			else if (char === "?" && state.depth === 0) { state.ternaryOpen = true; state.ternaryColon = false; }
			else if (char === ":" && state.depth === 0 && state.ternaryOpen) state.ternaryColon = true;
			index += 1;
		}
		if (state.depth === 0 && state.ternaryOpen && state.ternaryColon) {
			state.ternaryOpen = false; state.ternaryColon = false;
		}
	}

	function statementName(raw) {
		var first = raw.split("\n")[0].trim();
		var equals = first.indexOf("=");
		if (equals < 0) return "";
		var head = first.slice(0, equals).trim();
		if (!head) return "";
		if (head.charAt(0) === "$") {
			var body = head.slice(1);
			return /^[A-Za-z_][A-Za-z0-9_]*$/.test(body) ? head : "";
		}
		return /^[A-Za-z_][A-Za-z0-9_]*$/.test(head) ? head : "";
	}

	function splitStatements(text, final) {
		var lines = text.split("\n"), out = [], buffer = [];
		var state = { depth: 0, quote: "", ternaryOpen: false, ternaryColon: false };
		var candidate = false;

		function flush(complete, neededClosing) {
			var raw = buffer.join("\n").trim();
			if (raw) {
				var name = statementName(raw);
				if (name) out.push({ name: name, text: raw, complete: complete, neededClosing: !!neededClosing });
			}
			buffer = [];
		}

		lines.forEach(function (line) {
			var stripped = line.replace(/^\s+/, "");
			if (buffer.length && candidate) {
				if (stripped.charAt(0) === "?") {
					candidate = false;
				} else if (statementName(buffer.join("\n").trim())) {
					flush(true, false);
					state = { depth: 0, quote: "", ternaryOpen: false, ternaryColon: false };
				} else {
					candidate = false;
				}
			}
			if (!buffer.length && !stripped) return;
			buffer.push(line);
			scanLine(line, state);
			if (state.quote || state.depth > 0 || (state.ternaryOpen && !state.ternaryColon)) {
				candidate = false;
				return;
			}
			candidate = true;
		});

		if (buffer.length) {
			var raw = buffer.join("\n").trim();
			if (raw) {
				var name = statementName(raw);
				if (name) out.push({ name: name, text: raw, complete: final && candidate, neededClosing: !candidate });
			}
		}
		return out;
	}

	function closeImplicitly(text) {
		var stack = [], quote = "", index = 0, midEscape = false;
		while (index < text.length) {
			var char = text[index];
			if (quote) {
				if (char === "\\") { midEscape = index + 1 >= text.length; index += 2; continue; }
				midEscape = false;
				if (char === quote) quote = "";
				index += 1;
				continue;
			}
			midEscape = false;
			if (char === "\"" || char === "'") quote = char;
			else if ("([{".indexOf(char) >= 0) stack.push(char);
			else if (")]}".indexOf(char) >= 0) {
				var expected = { ")": "(", "]": "[", "}": "{" }[char];
				if (stack.length && stack[stack.length - 1] === expected) stack.pop();
			}
			index += 1;
		}
		var result = text;
		if (quote) {
			if (midEscape) result += "\\";
			result += quote;
		}
		for (var position = stack.length - 1; position >= 0; position -= 1) {
			result += { "(": ")", "[": "]", "{": "}" }[stack[position]];
		}
		return result;
	}

	// ── lexer (spec 1.3) ────────────────────────────────────────────────────

	var OPERATORS = ["==", "!=", ">=", "<=", "&&", "||", "=", ">", "<", "+", "-", "*", "/", "%", "!", "?", ":", ".", ",", "(", ")", "[", "]", "{", "}"];

	function lex(source) {
		var tokens = [], index = 0;
		while (index < source.length) {
			var char = source[index];
			if (" \t\r\n".indexOf(char) >= 0) { index += 1; continue; }
			if (char === "#" || (char === "/" && source[index + 1] === "/")) {
				while (index < source.length && source[index] !== "\n") index += 1;
				continue;
			}
			if (char === "\"" || char === "'") {
				var quote = char, pieces = "";
				index += 1;
				while (index < source.length) {
					var current = source[index];
					if (current === "\\" && index + 1 < source.length) {
						var decoded = unescape(quote, source, index);
						pieces += decoded.text;
						index += decoded.length;
						continue;
					}
					if (current === quote) { index += 1; break; }
					pieces += current;
					index += 1;
				}
				tokens.push({ kind: "STRING", value: pieces });
				continue;
			}
			if (/[A-Za-z_]/.test(char)) {
				var nameStart = index;
				while (index < source.length && /[A-Za-z0-9_]/.test(source[index])) index += 1;
				tokens.push({ kind: "NAME", value: source.slice(nameStart, index) });
				continue;
			}
			if (char === "$" || char === "@") {
				var prefixedStart = index;
				index += 1;
				var bodyStart = index;
				while (index < source.length && /[A-Za-z0-9_]/.test(source[index])) index += 1;
				tokens.push({ kind: char === "$" ? "STATE" : "AT", value: source.slice(bodyStart, index), raw: source.slice(prefixedStart, index) });
				continue;
			}
			if (/[0-9]/.test(char) || (char === "-" && startsNumber(source, index, tokens))) {
				var numberStart = index;
				if (source[index] === "-") index += 1;
				while (index < source.length && /[0-9]/.test(source[index])) index += 1;
				if (source[index] === "." && /[0-9]/.test(source[index + 1] || "")) {
					index += 1;
					while (index < source.length && /[0-9]/.test(source[index])) index += 1;
				}
				if (/[eE]/.test(source[index] || "")) {
					var look = index + 1;
					if (/[+-]/.test(source[look] || "")) look += 1;
					if (/[0-9]/.test(source[look] || "")) {
						index = look;
						while (index < source.length && /[0-9]/.test(source[index])) index += 1;
					}
				}
				tokens.push({ kind: "NUMBER", value: Number(source.slice(numberStart, index)) });
				continue;
			}
			var matched = null;
			for (var position = 0; position < OPERATORS.length; position += 1) {
				if (source.substr(index, OPERATORS[position].length) === OPERATORS[position]) { matched = OPERATORS[position]; break; }
			}
			if (matched === null) { index += 1; continue; }
			tokens.push({ kind: "OP", value: matched });
			index += matched.length;
		}
		return tokens;
	}

	function unescape(quote, source, index) {
		var char = source[index + 1];
		if (quote === "\"") {
			var mapping = { n: "\n", t: "\t", r: "\r", b: "\b", f: "\f", "\"": "\"", "\\": "\\", "/": "/" };
			if (char === "u") {
				var digits = source.substr(index + 2, 4);
				if (/^[0-9a-fA-F]{4}$/.test(digits)) return { text: String.fromCharCode(parseInt(digits, 16)), length: 6 };
			}
			return { text: Object.prototype.hasOwnProperty.call(mapping, char) ? mapping[char] : char, length: 2 };
		}
		var single = { "'": "'", "\\": "\\", n: "\n", t: "\t" };
		return { text: Object.prototype.hasOwnProperty.call(single, char) ? single[char] : "\\" + char, length: 2 };
	}

	function startsNumber(source, index, tokens) {
		if (!/[0-9]/.test(source[index + 1] || "")) return false;
		if (!tokens.length) return true;
		var previous = tokens[tokens.length - 1];
		if (previous.kind === "NUMBER" || previous.kind === "STRING") return false;
		if (previous.kind === "NAME" || previous.kind === "STATE") return false;
		if (previous.kind === "OP" && (previous.value === ")" || previous.value === "]")) return false;
		return true;
	}

	// ── parser (spec 1.5, 1.6) ──────────────────────────────────────────────

	function Parser(tokens) {
		this.tokens = tokens;
		this.pos = 0;
	}

	Parser.prototype.peek = function () { return this.pos < this.tokens.length ? this.tokens[this.pos] : null; };
	Parser.prototype.atOp = function () {
		var token = this.peek();
		if (!token || token.kind !== "OP") return false;
		for (var index = 0; index < arguments.length; index += 1) if (token.value === arguments[index]) return true;
		return false;
	};
	Parser.prototype.acceptOp = function () {
		if (!this.atOp.apply(this, arguments)) return false;
		this.pos += 1;
		return true;
	};
	Parser.prototype.expectOp = function (value) {
		if (!this.acceptOp(value)) throw new Error("expected " + value);
	};

	Parser.prototype.parseExpression = function () { return this.parseTernary(); };

	Parser.prototype.parseTernary = function () {
		var condition = this.parseOr();
		if (this.acceptOp("?")) {
			var consequent = this.parseTernary();
			this.expectOp(":");
			return { kind: "ternary", condition: condition, consequent: consequent, alternate: this.parseTernary() };
		}
		return condition;
	};

	function binaryLevel(next, operators) {
		// The level below is looked up at call time, so these assignments can sit
		// above the definitions they delegate to.
		return function () {
			var node = this[next]();
			while (true) {
				var matched = null;
				for (var index = 0; index < operators.length; index += 1) if (this.atOp(operators[index])) { matched = operators[index]; break; }
				if (matched === null) return node;
				this.pos += 1;
				node = { kind: "binary", op: matched, left: node, right: this[next]() };
			}
		};
	}

	Parser.prototype.parseOr = binaryLevel("parseAnd", ["||"]);
	Parser.prototype.parseAnd = binaryLevel("parseEquality", ["&&"]);
	Parser.prototype.parseEquality = binaryLevel("parseComparison", ["==", "!="]);
	Parser.prototype.parseComparison = binaryLevel("parseAdditive", [">", "<", ">=", "<="]);
	Parser.prototype.parseAdditive = binaryLevel("parseMultiplicative", ["+", "-"]);
	Parser.prototype.parseMultiplicative = binaryLevel("parseUnary", ["*", "/", "%"]);

	Parser.prototype.parseUnary = function () {
		if (this.atOp("!", "-")) {
			var op = this.tokens[this.pos].value;
			this.pos += 1;
			return { kind: "unary", op: op, operand: this.parseUnary() };
		}
		return this.parsePostfix();
	};

	Parser.prototype.parsePostfix = function () {
		var node = this.parsePrimary();
		while (true) {
			if (this.acceptOp(".")) {
				var token = this.tokens[this.pos];
				this.pos += 1;
				node = { kind: "member", target: node, field: fieldName(token) };
				continue;
			}
			if (this.acceptOp("[")) {
				var index = this.parseExpression();
				this.expectOp("]");
				node = { kind: "index", target: node, index: index };
				continue;
			}
			return node;
		}
	};

	Parser.prototype.parsePrimary = function () {
		var token = this.peek();
		if (!token) throw new Error("unexpected end of input");
		if (token.kind === "NUMBER" || token.kind === "STRING") { this.pos += 1; return { kind: "literal", value: token.value }; }
		if (token.kind === "STATE") { this.pos += 1; return { kind: "state", name: token.value }; }
		if (token.kind === "AT") {
			this.pos += 1;
			var name = token.value;
			if (name === "Action") {
				var items = this.atOp("(") ? this.parseArguments() : [];
				return { kind: "actionList", args: items };
			}
			var args = this.atOp("(") ? this.parseArguments() : [];
			if (ACTION_NAMES[name]) return { kind: "action", name: name, args: args };
			return { kind: "builtin", name: name, args: args };
		}
		if (token.kind === "NAME") {
			this.pos += 1;
			var head = token.value;
			if (this.atOp("(")) {
				var callArgs = this.parseArguments();
				if (head === "Action") return { kind: "actionList", args: callArgs };
				if (RESERVED_CALL_HEADS[head]) return { kind: "reserved", name: head, args: callArgs };
				if (/^[A-Z]/.test(head)) return { kind: "component", name: head, args: callArgs };
				return { kind: "reference", name: head };
			}
			if (head === "true") return { kind: "literal", value: true };
			if (head === "false") return { kind: "literal", value: false };
			if (head === "null") return { kind: "literal", value: null };
			return { kind: "reference", name: head };
		}
		if (this.acceptOp("(")) { var inner = this.parseExpression(); this.expectOp(")"); return inner; }
		if (this.atOp("[")) return this.parseArray();
		if (this.atOp("{")) return this.parseObject();
		throw new Error("unexpected token " + token.value);
	};

	Parser.prototype.parseArguments = function () {
		this.expectOp("(");
		var args = [];
		while (!this.atOp(")") && this.peek()) {
			if (this.acceptOp(",")) continue;
			args.push(this.parseExpression());
		}
		this.expectOp(")");
		return args;
	};

	Parser.prototype.parseArray = function () {
		this.expectOp("[");
		var items = [];
		while (!this.atOp("]") && this.peek()) {
			if (this.acceptOp(",")) continue;
			items.push(this.parseExpression());
		}
		this.expectOp("]");
		return { kind: "array", items: items };
	};

	Parser.prototype.parseObject = function () {
		this.expectOp("{");
		var entries = [];
		while (!this.atOp("}") && this.peek()) {
			if (this.acceptOp(",")) continue;
			var token = this.tokens[this.pos];
			this.pos += 1;
			this.expectOp(":");
			entries.push([fieldName(token), this.parseExpression()]);
		}
		this.expectOp("}");
		return { kind: "object", entries: entries };
	};

	function fieldName(token) {
		if (!token) return "?";
		// A field accepts the same tokens as an object key, and both are bare:
		// `$s: 3` keys the object "s" (grammar/008).
		if (token.kind === "NUMBER") return String(token.value);
		return String(token.value);
	}

	function parseStatement(text) {
		var equals = text.indexOf("=");
		if (equals < 0) return null;
		var name = text.slice(0, equals).trim();
		var body = text.slice(equals + 1).trim();
		if (!name || !body) return null;
		var tokens = lex(body);
		if (!tokens.length) return null;
		var parser = new Parser(tokens);
		var expression;
		try {
			expression = parser.parseExpression();
		} catch (error) {
			return null;
		}
		if (parser.pos !== tokens.length) return null;
		var kind = "value";
		if (name.charAt(0) === "$") kind = "state";
		if (expression.kind === "reserved" && expression.name === "Query") kind = "query";
		if (expression.kind === "reserved" && expression.name === "Mutation") kind = "mutation";
		return { name: name, kind: kind, expression: expression, text: text };
	}

	// ── program and evaluation ──────────────────────────────────────────────

	function Program(source, final) {
		var prepared = prepare(source);
		var pieces = splitStatements(prepared, final);
		this.statements = [];
		this.pending = {};
		this.incomplete = endsIncomplete(prepared);
		for (var index = 0; index < pieces.length; index += 1) {
			var piece = pieces[index];
			var text = piece.neededClosing ? closeImplicitly(piece.text) : piece.text;
			var parsed = parseStatement(text);
			if (!parsed) continue;
			this.statements.push(parsed);
			if (piece.neededClosing) this.pending[parsed.name] = true;
		}
	}

	Program.prototype.effective = function () {
		var self = this;
		var completed = {};
		this.statements.forEach(function (statement) {
			if (!self.pending[statement.name]) completed[statement.name] = true;
		});
		return this.statements.filter(function (statement) {
			return !(self.pending[statement.name] && completed[statement.name]);
		});
	};

	function Evaluator(library, statements, functions) {
		this.library = library || {};
		this.functions = functions || {};
		this.byName = {};
		this.errors = [];
		this.unresolved = {};
		this.reachable = {};
		this.cache = {};
		this.resolving = {};
		this.state = {};
		this.current = "";
		this.entryName = "";
		var self = this;
		statements.forEach(function (statement) { self.byName[statement.name] = statement; });
		Object.keys(this.byName).forEach(function (name) {
			var statement = self.byName[name];
			if (statement.kind !== "state") return;
			var saved = self.errors;
			self.errors = [];
			var value = self.evaluate(statement.expression);
			self.errors = saved;
			self.state[name] = value;
		});
	}

	Evaluator.prototype.spec = function (name) {
		var components = this.library.components || {};
		return components[name] || null;
	};

	Evaluator.prototype.componentNames = function () {
		var schema = (this.library.schema || {});
		return Object.keys(schema.properties || {});
	};

	Evaluator.prototype.componentSpec = function (name) {
		var schema = (this.library.schema || {});
		var def = (schema.$defs || {})[name];
		if (!def) return null;
		var required = def.required || [];
		var props = Object.keys(def.properties || {}).map(function (propName, position) {
			return {
				name: propName,
				schema: def.properties[propName],
				required: required.indexOf(propName) >= 0,
				order: position
			};
		});
		return { name: name, props: props, required: required };
	};

	function refsOf(schema) {
		if (!schema) return [];
		if (schema.type === "array") return refsOf(schema.items);
		var refs = [];
		if (typeof schema.$ref === "string") refs.push(schema.$ref.split("/").pop());
		(schema.anyOf || []).forEach(function (option) {
			if (option && typeof option.$ref === "string") refs.push(option.$ref.split("/").pop());
		});
		return refs;
	}

	Evaluator.prototype.isComponentSlot = function (prop) {
		var refs = refsOf(prop.schema);
		if (!refs.length) return false;
		var names = this.componentNames();
		return refs.every(function (ref) { return names.indexOf(ref) >= 0; });
	};

	Evaluator.prototype.isActionSlot = function (prop) {
		return refsOf(prop.schema).some(function (ref) { return ref === "ActionExpression"; });
	};

	Evaluator.prototype.isList = function (prop) {
		return prop.schema && prop.schema.type === "array";
	};

	Evaluator.prototype.isBinding = function (componentName, propName) {
		var entry = (this.library.components || {})[componentName];
		var signature = entry && entry.signature ? entry.signature : "";
		if (signature.indexOf("$binding<") < 0) return false;
		var inside = signature.slice(signature.indexOf("(") + 1, signature.lastIndexOf(")"));
		var parts = inside.split(",");
		for (var index = 0; index < parts.length; index += 1) {
			var pieces = parts[index].split(":");
			if (pieces.length < 2) continue;
			if (pieces[0].trim().replace(/\?$/, "") === propName && pieces.slice(1).join(":").indexOf("$binding<") >= 0) return true;
		}
		return false;
	};

	Evaluator.prototype.evaluate = function (node) {
		if (!node) return null;
		switch (node.kind) {
			case "literal": return node.value;
			case "state":
				var key = "$" + node.name;
				if (!(key in this.state)) this.state[key] = null;
				return this.state[key];
			case "reference": return this.reference(node.name);
			case "component": return this.component(node);
			case "reserved": return null;
			case "builtin":
				// A registered action name evaluates to an action object, not a call
				// (spec 6.3); only a real function runs here.
				if ((this.library.actions || {})[node.name]) return { actions: [this.customAction(node)] };
				return this.builtin(node);
			case "action": return { actions: [this.action(node)] };
			case "actionList": return { actions: this.actionItems(node) };
			case "array": return this.array(node);
			case "object":
				var result = {}, self = this;
				node.entries.forEach(function (entry) { result[entry[0]] = self.evaluate(entry[1]); });
				return result;
			case "member": return memberOf(this.evaluate(node.target), node.field);
			case "index":
				var target = this.evaluate(node.target), index = this.evaluate(node.index);
				if (Array.isArray(target)) {
					var position = toNumber(index);
					return position >= 0 && position < target.length && position === Math.floor(position) ? target[position] : null;
				}
				if (target && typeof target === "object") return Object.prototype.hasOwnProperty.call(target, toString(index)) ? target[toString(index)] : null;
				return null;
			case "unary":
				var operand = this.evaluate(node.operand);
				return node.op === "!" ? !toBoolean(operand) : -toNumber(operand);
			case "binary": return this.binary(node);
			case "ternary": return this.evaluate(toBoolean(this.evaluate(node.condition)) ? node.consequent : node.alternate);
		}
		return null;
	};

	Evaluator.prototype.reference = function (name) {
		if (Object.prototype.hasOwnProperty.call(this.loop || {}, name)) return this.loop[name];
		var statement = this.byName[name];
		if (!statement) { this.unresolved[name] = true; return null; }
		this.reachable[name] = true;
		if (name in this.cache) return this.cache[name];
		if (this.resolving[name]) { this.unresolved[name] = true; return null; }
		this.resolving[name] = true;
		var previous = this.current;
		this.current = name;
		var value;
		if (statement.kind === "state") value = this.state[name];
		else if (statement.kind === "query") value = statement.expression.args.length >= 3 ? this.evaluate(statement.expression.args[2]) : null;
		else if (statement.kind === "mutation") value = { status: "idle", data: null, error: null };
		else value = this.evaluate(statement.expression);
		this.current = previous;
		delete this.resolving[name];
		this.cache[name] = value;
		return value;
	};

	Evaluator.prototype.array = function (node) {
		var items = [], self = this;
		node.items.forEach(function (item) {
			if (item.kind === "reference") {
				if (!self.byName[item.name] || self.resolving[item.name]) { self.unresolved[item.name] = true; return; }
			}
			if (item.kind === "reserved") { self.error("inline-reserved"); return; }
			var value = self.evaluate(item);
			if (value === NOTHING) return;
			items.push(value);
		});
		return items;
	};

	Evaluator.prototype.binary = function (node) {
		if (node.op === "&&") {
			var leftAnd = this.evaluate(node.left);
			return toBoolean(leftAnd) ? this.evaluate(node.right) : leftAnd;
		}
		if (node.op === "||") {
			var leftOr = this.evaluate(node.left);
			return toBoolean(leftOr) ? leftOr : this.evaluate(node.right);
		}
		var left = this.evaluate(node.left), right = this.evaluate(node.right);
		if (node.op === "+") return add(left, right);
		if (node.op === "-") return toNumber(left) - toNumber(right);
		if (node.op === "*") return toNumber(left) * toNumber(right);
		if (node.op === "/") return divide(left, right);
		if (node.op === "%") return modulo(left, right);
		if (node.op === "==") return looseEqual(left, right);
		if (node.op === "!=") return !looseEqual(left, right);
		return compare(left, right, node.op);
	};

	Evaluator.prototype.error = function (code, component) {
		this.errors.push({ code: code, statementId: this.current, component: component || "" });
	};

	Evaluator.prototype.component = function (node) {
		var spec = this.componentSpec(node.name);
		if (!spec) { this.error("unknown-component", node.name); return NOTHING; }
		var props = {}, self = this, dropped = false;
		if (node.args.length > spec.props.length) this.error("excess-args", node.name);
		node.args.forEach(function (arg, position) {
			if (dropped || position >= spec.props.length) return;
			if (self.assignProp(spec, spec.props[position], arg, props) === "drop") dropped = true;
		});
		if (dropped) return NOTHING;
		spec.props.forEach(function (prop) {
			if (Object.prototype.hasOwnProperty.call(props, prop.name)) return;
			if (Object.prototype.hasOwnProperty.call(prop.schema || {}, "default")) { props[prop.name] = prop.schema["default"]; return; }
			if (prop.required) { self.error("missing-required", spec.name); dropped = true; }
		});
		if (dropped) return NOTHING;
		return { type: node.name, props: props };
	};

	Evaluator.prototype.assignProp = function (spec, prop, arg, props) {
		var self = this;
		if (this.isBinding(spec.name, prop.name) && arg.kind === "state") {
			props[prop.name] = { $binding: "$" + arg.name };
			return "ok";
		}
		if (arg.kind === "reserved") {
			this.error("inline-reserved", spec.name);
			return prop.required ? "drop" : "ok";
		}
		var value = this.evaluate(arg);
		if (value === NOTHING) return prop.required ? "drop" : "ok";
		if (value === null) {
			if (prop.required) { this.error("null-required", spec.name); return "drop"; }
			return "ok";
		}
		if (this.isActionSlot(prop)) { props[prop.name] = value; return "ok"; }
		if (this.isComponentSlot(prop)) {
			if (this.isList(prop)) {
				if (!Array.isArray(value)) { this.error("type-mismatch", spec.name); return "ok"; }
				props[prop.name] = value.filter(function (item) {
					if (item === null || (item && typeof item === "object" && item.type && item.props)) return true;
					self.error("type-mismatch", spec.name);
					return false;
				});
				return "ok";
			}
			if (!(value && typeof value === "object" && value.type && value.props)) {
				this.error("type-mismatch", spec.name);
				return prop.required ? "drop" : "ok";
			}
			props[prop.name] = value;
			return "ok";
		}
		if (!this.typeOk(prop, value)) {
			this.error("type-mismatch", spec.name);
			if (Object.prototype.hasOwnProperty.call(prop.schema || {}, "default")) { props[prop.name] = prop.schema["default"]; return "ok"; }
			return prop.required ? "drop" : "ok";
		}
		props[prop.name] = value;
		return "ok";
	};

	Evaluator.prototype.typeOk = function (prop, value) {
		if (this.isList(prop)) return Array.isArray(value);
		if (prop.schema && Array.isArray(prop.schema.enum)) return typeof value === "string" && prop.schema.enum.indexOf(value) >= 0;
		var type = prop.schema && prop.schema.type;
		if (type === "string") return typeof value === "string";
		if (type === "number") return isNumber(value);
		if (type === "boolean") return typeof value === "boolean";
		if (type === "object") return !!value && typeof value === "object" && !Array.isArray(value);
		return true;
	};

	Evaluator.prototype.builtin = function (node) {
		var self = this;
		if (node.name === "Each") return this.each(node);
		if ((this.library.functions || {})[node.name]) return this.customFunction(node);
		if (!BUILTINS[node.name]) { this.error("unknown-function", node.name); return null; }
		var args = node.args.map(function (arg) { return self.evaluate(arg); });
		var first = args.length ? args[0] : null;
		if (node.name === "Filter") return filterValues(first, args.slice(1));
		if (node.name === "Sort") return sortValues(first, args.slice(1));
		if (node.name === "Count") return Array.isArray(first) ? first.length : 0;
		if (node.name === "First") return Array.isArray(first) && first.length ? first[0] : null;
		if (node.name === "Last") return Array.isArray(first) && first.length ? first[first.length - 1] : null;
		if (node.name === "Sum") return Array.isArray(first) ? first.reduce(function (sum, item) { return sum + toNumber(item); }, 0) : 0;
		if (node.name === "Avg") return Array.isArray(first) && first.length ? first.reduce(function (sum, item) { return sum + toNumber(item); }, 0) / first.length : 0;
		if (node.name === "Min" || node.name === "Max") {
			if (!Array.isArray(first) || !first.length) return 0;
			var numbers = first.map(toNumber);
			return node.name === "Min" ? Math.min.apply(null, numbers) : Math.max.apply(null, numbers);
		}
		if (node.name === "Round") return roundTo(toNumber(first), args.length > 1 ? toNumber(args[1]) : 0);
		if (node.name === "Abs") return Math.abs(toNumber(first));
		if (node.name === "Floor") return Math.floor(toNumber(first));
		if (node.name === "Ceil") return Math.ceil(toNumber(first));
		return null;
	};

	// The schema block is where a callable's params live; the top-level
	// `functions`/`actions` maps carry only the prompt-facing signature.
	Evaluator.prototype.callableSpec = function (kind, name) {
		var block = ((this.library.schema || {})[kind]) || {};
		return block[name] || null;
	};

	function paramSpecs(spec) {
		var params = (spec || {}).params || {};
		var properties = params.properties || {};
		var required = params.required || [];
		return Object.keys(properties).map(function (name) {
			return { name: name, schema: properties[name], required: required.indexOf(name) >= 0 };
		});
	}

	Evaluator.prototype.customFunction = function (node) {
		var params = paramSpecs(this.callableSpec("functions", node.name));
		var bound = this.bindArgs(params, node.args, node.name);
		if (bound === null) return null;
		var implementation = (this.functions || {})[node.name];
		if (typeof implementation !== "function") return null;
		// Positional, in declaration order: the browser-side convention, since a
		// JS host registers a plain function rather than a keyword-only one.
		var positional = params.map(function (param) { return bound[param.name]; });
		try { return implementation.apply(null, positional); }
		catch (error) { this.error("runtime-error", node.name); return null; }
	};

	Evaluator.prototype.bindArgs = function (params, args, callName) {
		// Positional binding with the component-prop recovery rules (spec 3.6).
		if (args.length > params.length) this.error("excess-args", callName);
		var bound = {}, attempted = {}, failed = false, self = this;
		args.slice(0, params.length).forEach(function (arg, position) {
			var param = params[position];
			attempted[param.name] = true;
			var value = self.evaluate(arg);
			if (value === null || value === undefined) {
				if (param.required && !hasDefault(param.schema)) failed = true;
				else if (hasDefault(param.schema)) bound[param.name] = param.schema["default"];
				return;
			}
			if (!self.typeOk(param, value)) {
				self.error("type-mismatch", callName);
				if (hasDefault(param.schema)) bound[param.name] = param.schema["default"];
				else if (param.required) failed = true;
				return;
			}
			bound[param.name] = value;
		});
		params.forEach(function (param) {
			if (Object.prototype.hasOwnProperty.call(bound, param.name) || attempted[param.name]) return;
			if (hasDefault(param.schema)) bound[param.name] = param.schema["default"];
			else if (param.required) { self.error("missing-required", callName); failed = true; }
		});
		return failed ? null : bound;
	};

	function hasDefault(schema) {
		return !!schema && Object.prototype.hasOwnProperty.call(schema, "default");
	}

	Evaluator.prototype.each = function (node) {
		if (node.args.length < 3) return [];
		var items = this.evaluate(node.args[0]);
		if (!Array.isArray(items)) return [];
		var varName = toString(this.evaluate(node.args[1]));
		var self = this, results = [];
		items.forEach(function (item) {
			self.loop = self.loop || {};
			var saved = self.loop[varName];
			self.loop[varName] = item;
			results.push(self.evaluate(node.args[2]));
			if (saved === undefined) delete self.loop[varName]; else self.loop[varName] = saved;
		});
		return results;
	};

	function roundTo(number, decimals) {
		var factor = Math.pow(10, decimals);
		return Math.round(number * factor) / factor;
	}

	function filterValues(items, rest) {
		if (!Array.isArray(items)) return [];
		var field = rest.length && rest[0] !== null && rest[0] !== undefined ? toString(rest[0]) : "";
		var op = rest.length > 1 && rest[1] !== null && rest[1] !== undefined ? toString(rest[1]) : "==";
		var target = rest.length > 2 ? rest[2] : null;
		return items.filter(function (item) {
			var candidate = field ? pluckField(item, field) : item;
			if (op === "contains") return toString(candidate).indexOf(toString(target)) >= 0;
			if (op === "==") return looseEqual(candidate, target);
			if (op === "!=") return !looseEqual(candidate, target);
			if (op === ">" || op === "<" || op === ">=" || op === "<=") return compare(candidate, target, op);
			return false;
		});
	}

	function sortValues(items, rest) {
		if (!Array.isArray(items)) return items;
		var field = rest.length && rest[0] !== null && rest[0] !== undefined ? toString(rest[0]) : "";
		var descending = rest.length > 1 && toString(rest[1]) === "desc";
		var ordered = items.slice().sort(function (a, b) {
			var left = field ? pluckField(a, field) : a, right = field ? pluckField(b, field) : b;
			var leftText = left === null || left === undefined ? "" : toString(left);
			var rightText = right === null || right === undefined ? "" : toString(right);
			var leftNumber = isNumber(left) ? left : (leftText !== "" && !isNaN(Number(leftText)) ? Number(leftText) : null);
			var rightNumber = isNumber(right) ? right : (rightText !== "" && !isNaN(Number(rightText)) ? Number(rightText) : null);
			if (leftNumber !== null && rightNumber !== null) return leftNumber - rightNumber;
			return leftText.localeCompare(rightText);
		});
		if (descending) ordered.reverse();
		return ordered;
	}

	Evaluator.prototype.actionItems = function (node) {
		var self = this, items = [];
		if (node.kind === "array") {
			node.items.forEach(function (item) { items = items.concat(self.actionItems(item)); });
			return items;
		}
		if (node.kind === "actionList") {
			node.args.forEach(function (item) { items = items.concat(self.actionItems(item)); });
			return items;
		}
		if (node.kind === "action" || node.kind === "builtin") {
			if ((this.library.actions || {})[node.name]) return [this.customAction(node)];
			if (ACTION_NAMES[node.name]) return [this.builtinAction(node)];
			this.error("unknown-function", node.name);
			return [];
		}
		var value = this.evaluate(node);
		return value && Array.isArray(value.actions) ? value.actions.slice() : [];
	};

	Evaluator.prototype.action = function (node) {
		var items = this.actionItems(node);
		return items.length ? items[0] : null;
	};

	function targetName(node) {
		if (!node) return "";
		if (node.kind === "state") return "$" + node.name;
		if (node.kind === "reference") return node.name;
		if (node.kind === "literal") return toString(node.value);
		return "";
	}

	Evaluator.prototype.builtinAction = function (node) {
		var self = this;
		if (node.name === "Set") {
			return { action: "Set", args: [targetName(node.args[0]), node.args.length > 1 ? this.evaluate(node.args[1]) : null] };
		}
		if (node.name === "Reset") return { action: "Reset", args: node.args.map(targetName) };
		if (node.name === "Run") return { action: "Run", args: node.args.length ? [targetName(node.args[0])] : [] };
		if (node.name === "ToAssistant") {
			var args = [node.args.length ? this.evaluate(node.args[0]) : ""];
			if (node.args.length > 1) args.push(this.evaluate(node.args[1]));
			return { action: "ToAssistant", args: args };
		}
		return { action: "OpenUrl", args: [node.args.length ? this.evaluate(node.args[0]) : ""] };
	};

	Evaluator.prototype.customAction = function (node) {
		var params = paramSpecs(this.callableSpec("actions", node.name));
		var bound = this.bindArgs(params, node.args, node.name);
		if (bound === null) return { action: node.name, args: [] };
		return { action: node.name, args: params.map(function (param) { return bound[param.name]; }) };
	};

	Evaluator.prototype.run = function () {
		var self = this;
		// Reachability is derived from the tree below, so start clean: the walk
		// must not be short-circuited by statements evaluation happened to visit.
		this.reachable = {};
		var root = null;
		var hasEntry = false;
		if (this.byName.root) {
			this.entryName = "root";
			hasEntry = true;
			var value = this.reference("root");
			if (value && value.type && value.props) root = value;
			else this.errors.push({ code: "no-root", statementId: "" });
		} else if (this.library.root) {
			var names = Object.keys(this.byName);
			for (var index = 0; index < names.length; index += 1) {
				var statement = this.byName[names[index]];
				if (statement.kind !== "value") continue;
				var candidate = this.reference(statement.name);
				if (candidate && candidate.type === this.library.root) {
					this.entryName = statement.name;
					hasEntry = true;
					root = candidate;
					this.errors.push({ code: "no-root", statementId: "", severity: "warning" });
					break;
				}
			}
		}
		if (!root && !this.errors.some(function (error) { return error.code === "no-root"; })) {
			this.errors.push({ code: "no-root", statementId: "" });
		}
		this.reachable = {};
		if (!hasEntry) {
			// No entry statement at all: nothing is reachable, and no statement is
			// an orphan (spec 8.4). An entry that failed to render still roots the
			// tree, so its own references are reachable and the rest are orphans.
			return {
				root: null, errors: this.errors, unresolved: Object.keys(this.unresolved),
				orphans: [], state: this.state
			};
		}
		var pending = [this.entryName];
		Object.keys(this.byName).forEach(function (name) {
			if (self.byName[name].kind === "query" || self.byName[name].kind === "mutation") pending.push(name);
		});
		while (pending.length) {
			var name = pending.pop();
			if (!name || this.reachable[name]) continue;
			var statement = this.byName[name];
			if (!statement) continue;
			this.reachable[name] = true;
			walk(statement.expression, function (node) {
				if (node.kind === "reference" && self.byName[node.name]) pending.push(node.name);
			});
		}
		var orphans = [];
		Object.keys(this.byName).forEach(function (name) {
			var statement = self.byName[name];
			if (statement.kind === "value" && !self.reachable[name]) orphans.push(name);
		});
		return {
			root: root,
			errors: this.errors,
			unresolved: Object.keys(this.unresolved),
			orphans: orphans,
			state: this.state
		};
	};

	var NOTHING = { __nothing: true };

	function walk(node, visit) {
		if (!node || typeof node !== "object") return;
		visit(node);
		var children = [];
		if (node.args) children = node.args;
		else if (node.items) children = node.items;
		else if (node.entries) children = node.entries.map(function (entry) { return entry[1]; });
		else if (node.kind === "member") children = [node.target];
		else if (node.kind === "index") children = [node.target, node.index];
		else if (node.kind === "unary") children = [node.operand];
		else if (node.kind === "binary") children = [node.left, node.right];
		else if (node.kind === "ternary") children = [node.condition, node.consequent, node.alternate];
		children.forEach(function (child) { walk(child, visit); });
	}

	// ── rendering ───────────────────────────────────────────────────────────

	function escapeHtml(value) {
		return String(value).replace(/[&<>"']/g, function (char) {
			return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[char];
		});
	}

	function textOf(value) {
		if (value === true) return "true";
		if (value === false) return "false";
		if (value === null || value === undefined) return "";
		return String(value);
	}

	function children(props) {
		return renderList(props.children);
	}

	function renderList(nodes) {
		if (!Array.isArray(nodes)) return renderNode(nodes);
		return nodes.map(renderNode).join("");
	}

	function renderNode(node) {
		if (node === null || node === undefined || node === NOTHING) return "";
		if (Array.isArray(node)) return node.map(renderNode).join("");
		if (typeof node !== "object" || !node.type || !node.props) {
			return '<span class="apg-openui-text">' + escapeHtml(textOf(node)) + "</span>";
		}
		var renderer = RENDERERS[node.type];
		if (!renderer) return "";
		return renderer(node.props);
	}

	var RENDERERS = {
		Page: function (props) {
			var header = props.title ? '<h1 class="apg-openui-heading">' + escapeHtml(textOf(props.title)) + "</h1>" : "";
			return '<div class="apg-openui-page">' + header + children(props) + "</div>";
		},
		Card: function (props) {
			var head = props.title
				? '<header class="apg-card-header"><h2>' + escapeHtml(textOf(props.title)) + "</h2></header>"
				: "";
			return '<section class="apg-card">' + head + children(props) + "</section>";
		},
		Stack: function (props) {
			var direction = props.direction === "row" ? "row" : "column";
			return '<div class="apg-openui-stack" data-direction="' + direction + '">' + children(props) + "</div>";
		},
		Text: function (props) {
			var text = escapeHtml(textOf(props.text));
			if (props.variant === "heading") return '<h2 class="apg-openui-heading">' + text + "</h2>";
			if (props.variant === "muted") return '<p class="apg-openui-muted">' + text + "</p>";
			return '<p class="apg-openui-body">' + text + "</p>";
		},
		Badge: badgeRenderer,
		Tag: badgeRenderer,
		Metric: function (props) {
			var value = escapeHtml(textOf(props.value)) + (props.unit ? escapeHtml(textOf(props.unit)) : "");
			var delta = "";
			if (isNumber(props.delta)) {
				delta = '<span class="apg-stat-delta ' + (props.delta >= 0 ? "up" : "down") + '">' +
					(props.delta >= 0 ? "+" : "") + escapeHtml(textOf(props.delta)) + "</span>";
			}
			return '<article class="apg-stat"><span class="apg-stat-label">' + escapeHtml(textOf(props.label)) +
				'</span><strong class="apg-stat-value">' + value + "</strong>" + delta + "</article>";
		},
		Table: function (props) {
			var labels = [], cells = [];
			(Array.isArray(props.columns) ? props.columns : []).forEach(function (column) {
				if (!column || column.type !== "Column") return;
				labels.push(escapeHtml(textOf(column.props.label)));
				cells.push(Array.isArray(column.props.data) ? column.props.data : []);
			});
			var rows = 0;
			cells.forEach(function (column) { rows = Math.max(rows, column.length); });
			var head = labels.map(function (label) { return "<th>" + label + "</th>"; }).join("");
			var body = "";
			for (var index = 0; index < rows; index += 1) {
				body += "<tr>" + cells.map(function (column) {
					var cell = index < column.length ? column[index] : null;
					return "<td>" + (cell === null || cell === undefined ? "" : escapeHtml(textOf(cell))) + "</td>";
				}).join("") + "</tr>";
			}
			var caption = props.caption ? "<caption>" + escapeHtml(textOf(props.caption)) + "</caption>" : "";
			return '<div class="apg-table-wrap"><table class="apg-table">' + caption +
				"<thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>";
		},
		Chart: function (props) {
			var points = Array.isArray(props.points) ? props.points : [];
			var header = props.title ? '<header class="apg-card-header"><h2>' + escapeHtml(textOf(props.title)) + "</h2></header>" : "";
			return '<figure class="apg-chart" data-apg-chart="' + escapeHtml(textOf(props.kind || "line")) +
				'" data-apg-chart-data="' + escapeHtml(JSON.stringify(points)) + '">' + header +
				'<figcaption class="apg-openui-muted">' + points.length + " points</figcaption></figure>";
		},
		Form: function (props) {
			return '<form class="apg-openui-form" data-apg-openui-form="' + escapeHtml(textOf(props.name)) + '">' +
				children(props) + "</form>";
		},
		Input: function (props) {
			var name = textOf(props.name);
			var label = textOf(props.label || props.placeholder || name);
			var binding = "", valueAttr = "";
			if (props.value && typeof props.value === "object" && props.value.$binding) binding = ' data-apg-bind="' + escapeHtml(props.value.$binding) + '"';
			else if (props.value !== undefined && props.value !== null) valueAttr = ' value="' + escapeHtml(textOf(props.value)) + '"';
			var rules = props.rules && typeof props.rules === "object" ? ' data-apg-rules="' + escapeHtml(JSON.stringify(props.rules)) + '"' : "";
			return '<label class="apg-field"><span>' + escapeHtml(label) + '</span><input type="' +
				escapeHtml(textOf(props.type || "text")) + '" name="' + escapeHtml(name) + '"' + valueAttr + binding + rules + "></label>";
		},
		Select: function (props) {
			var name = textOf(props.name);
			var label = textOf(props.label || name);
			var binding = props.value && typeof props.value === "object" && props.value.$binding
				? ' data-apg-bind="' + escapeHtml(props.value.$binding) + '"' : "";
			var options = (Array.isArray(props.items) ? props.items : []).filter(function (item) {
				return item && item.type === "SelectItem";
			}).map(function (item) {
				var optionValue = textOf(item.props.value);
				return '<option value="' + escapeHtml(optionValue) + '">' + escapeHtml(textOf(item.props.label || optionValue)) + "</option>";
			}).join("");
			return '<label class="apg-field"><span>' + escapeHtml(label) + '</span><select name="' +
				escapeHtml(name) + '"' + binding + ">" + options + "</select></label>";
		},
		Button: function (props) {
			var variant = props.variant === "secondary" ? " apg-btn-secondary" : (props.variant === "danger" ? " apg-btn-danger" : "");
			var payload = props.action !== undefined && props.action !== null
				? ' data-apg-action="' + escapeHtml(JSON.stringify(props.action)) + '"' : "";
			return '<button type="button" class="apg-btn' + variant + '"' + payload + ">" +
				escapeHtml(textOf(props.label)) + "</button>";
		},
		Link: function (props) {
			return '<a class="apg-openui-link" href="' + escapeHtml(textOf(props.href)) + '">' +
				escapeHtml(textOf(props.label)) + "</a>";
		}
	};

	function badgeRenderer(props) {
		var tone = props.tone || props.color || "neutral";
		var known = { neutral: true, success: true, warning: true, danger: true, info: true };
		if (tone === "gray") tone = "neutral";
		if (tone === "green") tone = "success";
		if (tone === "red") tone = "danger";
		return '<span class="apg-badge apg-badge-' + (known[tone] ? tone : "neutral") + '">' +
			escapeHtml(textOf(props.text)) + "</span>";
	}

	// ── view ────────────────────────────────────────────────────────────────

	function View(element, library, functions) {
		this.element = element;
		this.library = library || {};
		this.functions = functions || {};
		this.buffer = "";
		this.ended = false;
		this.program = null;
		this.result = null;
		this.element.classList.add("apg-openui-stream");
	}

	View.prototype.push = function (chunk) {
		if (this.ended) { this.buffer = ""; this.ended = false; }
		this.buffer += chunk;
		this.program = new Program(this.buffer, false);
		this.paint(false);
		return this.result;
	};

	View.prototype.end = function () {
		this.ended = true;
		this.program = new Program(this.buffer, true);
		this.paint(true);
		return this.result;
	};

	View.prototype.paint = function (final) {
		var statements = this.program.effective();
		var evaluator = new Evaluator(this.library, statements, this.functions);
		var result = evaluator.run();
		this.result = result;
		this.element.setAttribute("aria-busy", final ? "false" : "true");
		this.element.innerHTML = renderNode(result.root);
		this.element.dispatchEvent(new CustomEvent("apg:openui", { detail: result, bubbles: true }));
	};

	// ── actions ─────────────────────────────────────────────────────────────

	View.prototype.dispatch = function (action, context) {
		if (!action || !Array.isArray(action.actions)) return;
		var self = this;
		action.actions.forEach(function (item) {
			if (!item || !item.action) return;
			var name = item.action;
			var args = item.args || [];
			if (name === "Set") { self.set(args[0], args[1]); return; }
			if (name === "Reset") { args.forEach(function (target) { self.reset(target); }); return; }
			if (name === "Run") { self.emit({ type: "run", params: { statement: args[0] } }, context); return; }
			if (name === "ToAssistant") {
				self.emit({ type: "continue_conversation", humanFriendlyMessage: args[0], params: args.length > 1 ? { context: args[1] } : {} }, context);
				return;
			}
			if (name === "OpenUrl") { self.emit({ type: "open_url", params: { url: args[0] } }, context); return; }
			self.emit({ type: name, params: args }, context);
		});
	};

	View.prototype.set = function (target, value) {
		if (typeof target !== "string" || target.charAt(0) !== "$") return;
		var name = target.slice(1);
		this.state = this.state || {};
		this.state[target] = value;
		this.applyState();
	};

	View.prototype.reset = function (target) {
		if (typeof target !== "string" || target.charAt(0) !== "$") return;
		this.state = this.state || {};
		this.state[target] = null;
		this.applyState();
	};

	View.prototype.applyState = function () {
		if (!this.result) return;
		Object.keys(this.state || {}).forEach(function (key) { this.result.state[key] = this.state[key]; }, this);
		this.paint(true);
	};

	View.prototype.emit = function (event, context) {
		var detail = event;
		detail.formName = context && context.formName ? context.formName : null;
		detail.formState = context && context.formState ? context.formState : null;
		this.element.dispatchEvent(new CustomEvent("apg:openui-event", { detail: detail, bubbles: true }));
	};

	View.prototype.bind = function () {
		var self = this;
		this.element.addEventListener("click", function (event) {
			var button = event.target.closest("[data-apg-action]");
			if (!button) return;
			var action;
			try { action = JSON.parse(button.getAttribute("data-apg-action")); } catch (error) { return; }
			var form = button.closest("[data-apg-openui-form]");
			self.dispatch(action, self.formContext(form));
		});
		this.element.addEventListener("input", function (event) {
			var field = event.target.closest("[data-apg-bind]");
			if (!field) return;
			self.set(field.getAttribute("data-apg-bind"), field.value);
		});
	};

	View.prototype.formContext = function (form) {
		if (!form) return { formName: null, formState: this.result ? this.result.state : {} };
		var fields = {}, state = this.result ? this.result.state : {};
		form.querySelectorAll("[name]").forEach(function (field) {
			fields[field.getAttribute("name")] = { value: field.value, componentType: field.tagName.toLowerCase() };
		});
		return { formName: form.getAttribute("data-apg-openui-form"), formState: Object.assign({}, state, fields) };
	};

	// ── public surface ──────────────────────────────────────────────────────

	var APGOpenUI = {
		version: "1.0.0",
		mount: function (element, library, functions) {
			var view = new View(element, library, functions);
			view.bind();
			return view;
		},
		render: function (tree, library) {
			return renderNode(tree);
		},
		parse: function (source, library, functions) {
			var program = new Program(source, true);
			var evaluator = new Evaluator(library || {}, program.effective(), functions);
			return evaluator.run();
		},
		Program: Program,
		Evaluator: Evaluator
	};

	if (typeof module === "object" && module.exports) module.exports = APGOpenUI;
	else global.APGOpenUI = APGOpenUI;
})(typeof window !== "undefined" ? window : this);
