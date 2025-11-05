#!/usr/bin/env rust-script
//! WIP - NOT READY FOR USE YET
//!
//! This wrapper converts KatVisualizer's OSC output into a format directly usable within lighting software. Mapping this data to OSC controls is far from trivial, which is why this functionality is not built in to KatVisualizer.
//!
//! This script is meant as a *starting point* for using KatVisualizer for lighting control, and should be customized by the user to fit their needs.
//!
//! ```cargo
//! [dependencies]
//! anyhow = "1.0.100"
//! clap = { version = "4.5.51", features = ["derive"] }
//! rosc = "0.11.4"
//! ```
#![feature(vec_into_chunks)] // Requires nightly rust (rust-script --toolchain nightly)

use clap::Parser;
use rosc::{OscArray, OscBundle, OscMessage, OscPacket, OscTime, OscType, decoder, encoder};

use std::{
    collections::VecDeque,
    net::{SocketAddr, UdpSocket},
    time::{Duration, Instant},
};

struct Handler {
    normalization_data: VecDeque<(f64, Duration)>,
    normalization_duration: Duration,
    agc_length: Duration,
    agc_target_minimum: f64,
    above_masking: f64,
    below_masking: f64,

    frequency_scale: Vec<(f32, f32, f32)>,
}

impl Handler {
    fn new(config: &Args) -> Self {
        Self {
            normalization_data: VecDeque::with_capacity(8192),
            normalization_duration: Duration::ZERO,
            agc_length: Duration::from_secs_f32(config.agc_length),
            agc_target_minimum: config.agc_target_minimum as f64,
            above_masking: config.above_masking as f64,
            below_masking: config.below_masking as f64,

            frequency_scale: Vec::with_capacity(139),
        }
    }
    fn get_normalization_targets(&mut self, masking_mean: f32, duration: Duration) -> (f32, f32) {
        self.normalization_data
            .push_back((masking_mean as f64, duration));
        self.normalization_duration += duration;

        while self.normalization_duration > self.agc_length {
            let front = self.normalization_data.pop_front().unwrap();
            self.normalization_duration -= front.1;
        }

        let mean = (self
            .normalization_data
            .iter()
            .map(|(d, l)| d * l.as_secs_f64())
            .sum::<f64>()
            / self.normalization_duration.as_secs_f64())
        .max(self.agc_target_minimum);

        (
            (mean - self.below_masking) as f32,
            (mean + self.above_masking) as f32,
        )
    }
    fn update_frequency_scale(&mut self, sorted_analysis: &[(f32, f32, f32, f32, f32)]) {
        let mut scratchpad = [(0.0, 0.0); 139];

        for (frequency, _, _, volume, _) in sorted_analysis {
            if *frequency > 20000.0 {
                break;
            } else if *frequency < 20.0 || !volume.is_finite() {
                continue;
            }

            let index = scale_erb(*frequency as f64).round() as usize;
            scratchpad[index].0 += *frequency as f64 * *volume as f64;
            scratchpad[index].1 += *volume as f64;
        }

        let mut mean_erb = [0.0; 139];

        for (index, (sum, count)) in scratchpad.into_iter().enumerate() {
            mean_erb[index] = if count > 0.0 {
                scale_erb(sum / count).clamp(index as f64 - 0.5, index as f64 + 0.5)
            } else {
                index as f64
            };
        }

        self.frequency_scale.clear();

        let mut lower = 20.0;

        for i in 0..139 {
            let x = mean_erb[i];

            if x == 0.0 {
                self.frequency_scale.push((0.0, 0.0, 0.0));
                continue;
            }

            let low = inv_scale_erb(x - 0.5).min(lower);
            let hi = if i < 138 {
                let current_erb = x + 0.5;
                let next_erb = mean_erb[i + 1] - 0.5;

                inv_scale_erb(current_erb.max((current_erb + next_erb) / 2.0))
            } else {
                inv_scale_erb(x + 0.5)
            };

            lower = hi;

            self.frequency_scale
                .push((low as f32, inv_scale_erb(x) as f32, hi as f32));
        }
    }
    fn handle_packet(&mut self, packet: OscPacket) -> anyhow::Result<Vec<OscPacket>> {
        let mut data = VisualizerData::try_from(packet).map_err(|e| anyhow::anyhow!(e))?;

        let (lower, upper) = self.get_normalization_targets(data.masking_mean, data.duration);

        data.analysis.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        self.update_frequency_scale(&data.analysis);

        let mut scale_index = 0;
        let mut scratchpad = [(0.0, 0.0); 139];

        for (frequency, _bandwidth, _pan, volume, _stm) in data.analysis {
            while self.frequency_scale[scale_index].0 < frequency && scale_index < 138 {
                scale_index += 1;
            }

            scratchpad[scale_index].0 += volume as f64;
            scratchpad[scale_index].1 += 1.0
        }

        let mut scale_amplitudes = [0.0; 139];

        for (index, (sum, count)) in scratchpad.into_iter().enumerate() {
            if count > 0.0 {
                scale_amplitudes[index] =
                    map_value_f32((sum / count) as f32, lower, upper, 0.0, 1.0).clamp(0.0, 1.0)
            }
        }

        let frequency_messages = self
            .frequency_scale
            .iter()
            .enumerate()
            .filter(|(i, _)| *i > 1)
            .map(|(i, (_, f, _))| {
                OscPacket::Message(OscMessage {
                    addr: ["/tones/", &(i - 1).to_string(), "/frequency"].concat(),
                    args: vec![OscType::Float(*f)],
                })
            });

        let amplitude_messages = scale_amplitudes
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i > 1)
            .map(|(i, v)| {
                OscPacket::Message(OscMessage {
                    addr: ["/tones/", &(i - 1).to_string(), "/amplitude"].concat(),
                    args: vec![OscType::Float(v)],
                })
            });

        /*Ok(vec![
            OscPacket::Bundle(OscBundle {
                timetag: data.timetag,
                content: frequency_messages.chain(amplitude_messages).collect(),
            }),
        ])*/

        Ok(frequency_messages
            .zip(amplitude_messages)
            .map(|(frequency_message, amplitude_message)| {
                OscPacket::Bundle(OscBundle {
                    timetag: data.timetag,
                    content: vec![frequency_message, amplitude_message],
                })
            })
            .collect())
    }
}

struct VisualizerData {
    timetag: OscTime,
    duration: Duration,

    masking_mean: f32,
    mean: f32,
    max: f32,
    analysis: Vec<(f32, f32, f32, f32, f32)>,
}

/// A server for converting KatVisualizer's OSC output into a more usable format.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The UDP/IP address that the server should listen on for OSC packets from KatVisualizer.
    #[arg(short, long)]
    listen_address: SocketAddr,

    /// The UDP/IP address that converted OSC packets should be sent to.
    #[arg(short, long)]
    destination_address: SocketAddr,

    #[arg(long, default_value_t = 0.128)]
    agc_length: f32,

    #[arg(long, default_value_t = 15.0)]
    agc_target_minimum: f32,

    #[arg(long, default_value_t = 30.0)]
    above_masking: f32,

    #[arg(long, default_value_t = 0.0)]
    below_masking: f32,
}

const MAX_PACKET_SIZE: usize = 65535;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let socket = UdpSocket::bind(args.listen_address)?;
    let mut read_buf = [0; MAX_PACKET_SIZE];
    let mut write_buf = Vec::with_capacity(MAX_PACKET_SIZE);

    let mut handler = Handler::new(&args);

    println!(
        "Listening address: {}\nDestination address: {}",
        args.listen_address, args.destination_address
    );

    loop {
        let length = socket.recv(&mut read_buf)?;
        let received = &read_buf[0..length];

        //let start = Instant::now();

        match decoder::decode_udp(received) {
            Ok((_, packet)) => match handler.handle_packet(packet) {
                Ok(responses) => {
                    for response in responses {
                        write_buf.clear();
                        encoder::encode_into(&response, &mut write_buf)?;
                        socket.send_to(&write_buf, args.destination_address)?;

                        //println!("Sent {} bytes", write_buf.len());
                    }

                    /*println!(
                        "Processing time: {:.1} ms",
                        start.elapsed().as_secs_f32() / 1000.0
                    );*/
                }
                Err(e) => {
                    eprintln!("Failed to handle OSC packet:\n{:#?}", e);
                }
            },
            Err(e) => {
                eprintln!("Failed to decode packet:\n{:#?}", e);
            }
        }
    }
}

const FORMAT_VERSION: &str = "v0.8.4";

impl TryFrom<OscPacket> for VisualizerData {
    type Error = &'static str;

    fn try_from(value: OscPacket) -> Result<Self, Self::Error> {
        match value {
            OscPacket::Bundle(mut bundle) => {
                if let (
                    Some(OscPacket::Message(mut analysis)),
                    Some(OscPacket::Message(mut metadata)),
                ) = (bundle.content.pop(), bundle.content.pop())
                {
                    if metadata.addr != ["/katvisualizer/", FORMAT_VERSION, "/stats"].concat()
                        || analysis.addr != ["/katvisualizer/", FORMAT_VERSION, "/tones"].concat()
                    {
                        return Err("One or more message addresses are incorrect");
                    }

                    if metadata.args.len() != 4 || analysis.args.len() != 1 {
                        return Err("One or more message arguments are malformed");
                    }

                    let duration_secs = metadata
                        .args
                        .pop()
                        .ok_or("One or more message arguments are malformed")?
                        .float()
                        .ok_or("One or more message arguments are malformed")?;
                    let max = metadata
                        .args
                        .pop()
                        .ok_or("One or more message arguments are malformed")?
                        .float()
                        .ok_or("One or more message arguments are malformed")?;
                    let mean = metadata
                        .args
                        .pop()
                        .ok_or("One or more message arguments are malformed")?
                        .float()
                        .ok_or("One or more message arguments are malformed")?;
                    let masking_mean = metadata
                        .args
                        .pop()
                        .ok_or("One or more message arguments are malformed")?
                        .float()
                        .ok_or("One or more message arguments are malformed")?;
                    let duration = Duration::from_secs_f32(duration_secs);

                    let analysis_array = analysis
                        .args
                        .pop()
                        .ok_or("One or more message arguments are malformed")?
                        .array()
                        .ok_or("One or more message arguments are malformed")?
                        .content;

                    let mut analysis = Vec::with_capacity(analysis_array.len());

                    for tones in analysis_array.into_iter() {
                        let mut tone_array = tones
                            .array()
                            .ok_or("One or more message arguments are malformed")?
                            .content;

                        if tone_array.len() != 5 {
                            return Err("One or more message arguments are malformed");
                        }

                        let signal_mask_ratio = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;
                        let volume = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;
                        let pan = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;
                        let bandwidth = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;
                        let frequency = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;

                        analysis.push((frequency, bandwidth, pan, volume, signal_mask_ratio))
                    }

                    Ok(Self {
                        timetag: bundle.timetag,
                        duration,
                        masking_mean,
                        mean,
                        max,
                        analysis,
                    })
                } else {
                    Err("Message bundle is malformed")
                }
            }
            OscPacket::Message(_) => Err("Packet does not contain message bundle"),
        }
    }
}

fn map_value_f32(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
    (x - min) / (max - min) * (target_max - target_min) + target_min
}

fn scale_erb(x: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * x).log2()
}

fn inv_scale_erb(x: f64) -> f64 {
    (1.0 / 0.00437) * ((2.0_f64.powf(x / 21.4)) - 1.0)
}
