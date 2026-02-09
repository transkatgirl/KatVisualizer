addEventListener("TrunkApplicationStarted", async (event) => {
	await init();
});

async function init() {
	let wasm = window.wasmBindings;
	if (!wasm) {
		throw new Error("Webassembly bindings are undefined!");
	}

	const audioContext = new AudioContext();

	await audioContext.audioWorklet.addModule("./worklet.js");
	const workletNode = new AudioWorkletNode(audioContext, "copy-processor");

	let maxPosition = wasm.left_sample_buffer().length;

	workletNode.port.onmessage = (event) => {
		let input = event.data;

		if (input.length == 2) {
			wasm.set_stereo();
		} else if (input.length == 1) {
			wasm.set_mono();
		} else {
			throw new Error("Unsupported channel count!");
		}
		wasm.set_rate(audioContext.sampleRate);

		let position = wasm.get_position();

		for (let channel = 0; channel < input.length; ++channel) {
			let inputChannel = input[channel];

			let outputPosition = Math.min(
				position,
				Math.max(maxPosition - inputChannel.length, 0)
			);

			if (channel == 0) {
				wasm.left_sample_buffer().set(inputChannel, outputPosition);
			} else if (channel == 1) {
				wasm.right_sample_buffer().set(inputChannel, outputPosition);
			}
		}

		wasm.set_position(Math.min(position + input[0].length, maxPosition));
	};

	const audioElement = buildElements();
	const track = audioContext.createMediaElementSource(audioElement);

	audioElement.addEventListener("play", () => {
		if (audioContext.state === "suspended") {
			audioContext.resume();
		}
	});

	audioElement.addEventListener("playing", () => {
		if (audioContext.state === "suspended") {
			audioContext.resume();
		}
	});

	audioElement.addEventListener("pause", () => {
		audioContext.suspend();
	});

	audioElement.addEventListener("loadstart", () => {
		audioElement.play();
	});

	track.connect(workletNode).connect(audioContext.destination);

	audioContext.suspend();
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
	audioElement.style.display = "none";

	const message = document.createElement("p");
	message.style.color = "white";
	message.style.margin = "1em";
	message.style.fontFamily = "system-ui,sans-serif";
	message.innerText =
		"Note: Output device latency compensation is not supported.";

	const fileInput = document.createElement("input");
	fileInput.style.color = "white";
	fileInput.type = "file";
	fileInput.addEventListener("change", (event) => {
		if (!event.target.files.length) return;

		const urlObj = URL.createObjectURL(event.target.files[0]);

		audioElement.pause();
		audioElement.addEventListener("load", () => {
			URL.revokeObjectURL(urlObj);
		});
		audioElement.src = urlObj;
		audioElement.style.display = "inherit";
		message.style.display = "none";
	});

	leftControlDiv.appendChild(fileInput);
	rightControlDiv.appendChild(message);
	rightControlDiv.appendChild(audioElement);

	document.body.appendChild(leftControlDiv);
	document.body.appendChild(rightControlDiv);

	return audioElement;
}
