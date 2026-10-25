// Fixture-driven test of the browser renderer (compiler/assets/apg-openui.js).
//
// The browser renderer is a second implementation of the same language, so it
// is held to the same normative suite the Python parser is: every fixture case
// that does not require a patch merge (spec 7, which is applied server-side)
// must produce the same root, errors, unresolved set, and orphans.
//
// Run with `node tests/ci/openui_render_js.test.js`; the Python suite invokes
// it, so a broken port fails CI rather than a manual check.

"use strict";

var fs = require("fs");
var path = require("path");
var assert = require("assert");

var HERE = __dirname;
var ROOT = path.resolve(HERE, "..", "fixtures", "openui", "1.0");
var RENDERER = path.resolve(HERE, "..", "..", "compiler", "assets", "apg-openui.js");

var OpenUI = require(RENDERER);
var library = JSON.parse(fs.readFileSync(path.join(ROOT, "library.json"), "utf8"));

// The two callables the fixture library registers.
var functions = {
	Upper: function (text) { return String(text).toUpperCase(); }
};

function readJson(file) { return JSON.parse(fs.readFileSync(file, "utf8")); }

function cases() {
	var found = [];
	(function walk(directory) {
		fs.readdirSync(directory, { withFileTypes: true }).forEach(function (entry) {
			var full = path.join(directory, entry.name);
			if (entry.isDirectory()) return walk(full);
			if (entry.name === "expected.json") found.push(directory);
		});
	})(ROOT);
	return found.sort();
}

function normalise(value) {
	// Appendix B: absent and null compare equal.
	if (Array.isArray(value)) return value.map(normalise);
	if (value && typeof value === "object") {
		var out = {};
		Object.keys(value).forEach(function (key) {
			if (value[key] !== null && value[key] !== undefined) out[key] = normalise(value[key]);
		});
		return out;
	}
	return value;
}

function sourceOf(directory) {
	var chunks = path.join(directory, "chunks.json");
	if (fs.existsSync(chunks)) return readJson(chunks).join("");
	return fs.readFileSync(path.join(directory, "input.oui"), "utf8");
}

function compare(directory, expected, actual) {
	var problems = [];
	if ("root" in expected) {
		var got = JSON.stringify(normalise(actual.root));
		var want = JSON.stringify(normalise(expected.root));
		if (got !== want) problems.push("root\n    want " + want + "\n    got  " + got);
	}
	if ("errors" in expected) {
		var gotCodes = actual.errors.map(function (error) { return error.code; }).sort().join(",");
		var wantCodes = expected.errors.map(function (error) { return error.code; }).sort().join(",");
		if (gotCodes !== wantCodes) problems.push("errors want [" + wantCodes + "] got [" + gotCodes + "]");
	}
	["unresolved", "orphans"].forEach(function (key) {
		if (!(key in expected)) return;
		var got = actual[key].slice().sort().join(",");
		var want = expected[key].slice().sort().join(",");
		if (got !== want) problems.push(key + " want [" + want + "] got [" + got + "]");
	});
	return problems;
}

function run() {
	var passed = 0, failed = 0, skipped = 0;
	cases().forEach(function (directory) {
		var name = path.relative(ROOT, directory).split(path.sep).join("/");
		if (fs.existsSync(path.join(directory, "patch.oui"))) {
			// Spec 7 patch merging has no browser-side counterpart; the server
			// applies the patch and ships the merged program.
			skipped += 1;
			return;
		}
		var expected = readJson(path.join(directory, "expected.json"));
		var actual = OpenUI.parse(sourceOf(directory), library, functions);
		var problems = compare(directory, expected, actual);
		if (problems.length) {
			failed += 1;
			console.error("FAIL " + name + "\n  " + problems.join("\n  "));
		} else {
			passed += 1;
		}
	});
	console.log(passed + " passed, " + skipped + " skipped (patch merge), " + failed + " failed");
	assert.strictEqual(failed, 0, "browser renderer disagrees with " + failed + " fixture case(s)");
}

function testStreamingEqualsParsing() {
	// Spec 4: the streamed result at end of stream equals the batch parse.
	var chunks = ["root = Stack([a", ", b])\n", "a = Text(\"one\")\n", "b = Text(\"two\")"];
	var view = OpenUI.mount(fakeElement(), library);
	var streamed = null;
	chunks.forEach(function (chunk) { streamed = view.push(chunk); });
	streamed = view.end();
	var batch = OpenUI.parse(chunks.join(""), library);
	assert.deepStrictEqual(
		JSON.parse(JSON.stringify(streamed.root)),
		JSON.parse(JSON.stringify(batch.root)),
		"streaming must equal parsing (spec 4)"
	);
}

function testPartialPaintIsNotAnError() {
	// A truncated program still paints what it has; nothing throws.
	var view = OpenUI.mount(fakeElement(), library);
	var result = view.push('root = Stack([Text("hel');
	assert.ok(result.root, "a partial screen should still render a root");
}

function testEscapeInAttribute() {
	var result = OpenUI.parse('root = Stack([b])\nb = Link("x", "\\" onclick=\\"alert(1)")', library);
	var html = OpenUI.render(result.root);
	assert.ok(html.indexOf('onclick="alert(1)"') < 0, "attribute must be escaped");
}

function fakeElement() {
	var listeners = {};
	return {
		classList: { add: function () {} },
		attributes: {},
		innerHTML: "",
		setAttribute: function (name, value) { this.attributes[name] = value; },
		addEventListener: function (name, handler) { listeners[name] = handler; },
		dispatchEvent: function () {},
		querySelectorAll: function () { return []; }
	};
}

run();
testStreamingEqualsParsing();
testPartialPaintIsNotAnError();
testEscapeInAttribute();
console.log("browser renderer: ok");
