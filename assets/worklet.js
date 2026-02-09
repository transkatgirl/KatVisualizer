class CopyProcessor extends AudioWorkletProcessor {
	process(inputs, outputs, parameters) {
		const input = inputs[0];
		const output = outputs[0];
		for (let channel = 0; channel < input.length; ++channel) {
			output[channel].set(input[channel], 0);
		}

		this.port.postMessage(input);

		return true;
	}
}

registerProcessor("copy-processor", CopyProcessor);
