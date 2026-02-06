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

	const audioContext = new AudioContext();

	const audioElement = buildElements();
	const track = audioContext.createMediaElementSource(audioElement);

	audioElement.addEventListener("play", () => {
		if (audioContext.state === "suspended") {
			audioContext.resume();
		}
	});

	const scriptNode = audioContext.createScriptProcessor(512, 2, 2);
	scriptNode.addEventListener("audioprocess", (audioProcessingEvent) => {
		let inputBuffer = audioProcessingEvent.inputBuffer;
		let outputBuffer = audioProcessingEvent.outputBuffer;

		for (
			let channel = 0;
			channel < outputBuffer.numberOfChannels;
			channel++
		) {
			let inputData = inputBuffer.getChannelData(channel);
			let outputData = outputBuffer.getChannelData(channel);

			let wasmPosition = window.wasmBindings.get_position();

			let wasmBuffer;

			if (channel == 0) {
				wasmBuffer = window.wasmBindings.left_sample_buffer();
			} else if (channel == 1) {
				wasmBuffer = window.wasmBindings.right_sample_buffer();
			}

			for (let sample = 0; sample < inputBuffer.length; sample++) {
				outputData[sample] = inputData[sample];
				wasmBuffer[wasmPosition + sample] = inputData[sample];
			}

			if (channel == 1) {
				window.wasmBindings.set_position(
					wasmPosition + inputBuffer.length
				);
			}
		}
	});

	track.connect(scriptNode).connect(audioContext.destination);
}

function buildElements() {
	const leftControlDiv = document.createElement("div");
	leftControlDiv.style.zIndex = 1;
	leftControlDiv.style.position = "absolute";
	leftControlDiv.style.left = 0;
	leftControlDiv.style.bottom = 0;

	const rightControlDiv = document.createElement("div");
	rightControlDiv.style.zIndex = 1;
	rightControlDiv.style.position = "absolute";
	rightControlDiv.style.right = 0;
	rightControlDiv.style.bottom = 0;

	const audioElement = document.createElement("audio");
	audioElement.controls = true;

	const fileInput = document.createElement("input");
	fileInput.type = "file";
	fileInput.addEventListener("change", (event) => {
		if (!event.target.files.length) return;

		const urlObj = URL.createObjectURL(event.target.files[0]);

		audioElement.addEventListener("load", () => {
			URL.revokeObjectURL(urlObj);
		});
		audioElement.src = urlObj;
	});

	leftControlDiv.appendChild(fileInput);
	rightControlDiv.appendChild(audioElement);

	document.body.appendChild(leftControlDiv);
	document.body.appendChild(rightControlDiv);

	return audioElement;
}
