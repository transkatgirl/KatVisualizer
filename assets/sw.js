var cacheName = "egui-template-pwa";
var filesToCache = [
	"./",
	"./index.html",
	"./eframe_template.js",
	"./eframe_template_bg.wasm",
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
