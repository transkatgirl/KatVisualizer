var cacheName = "katvisualizer-pwa";
var filesToCache = [
	"./",
	"./main.js",
	"./worklet.js",
	"./index.html",
	"./katvisualizer_wasm.js",
	"./katvisualizer_wasm_bg.wasm",
];

self.addEventListener("install", function (e) {
	e.waitUntil(
		caches.open(cacheName).then(function (cache) {
			return cache.addAll(filesToCache);
		})
	);
});

self.addEventListener("fetch", function (e) {
	e.respondWith(
		caches.match(e.request).then(function (response) {
			return response || fetch(e.request);
		})
	);
});
