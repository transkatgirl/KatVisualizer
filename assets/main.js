addEventListener("TrunkApplicationStarted", (event) => {
	init();
});

if (window.wasmBindings) {
	init();
}

let initialized = false;

function init() {
	if (initialized) {
		return;
	} else {
		initialized = true;
	}
	console.log("application started - bindings:", window.wasmBindings);
}
