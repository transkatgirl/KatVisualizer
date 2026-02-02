#!/usr/bin/env rust-script
//! This wrapper is one attempt at converting KatVisualizer's OSC output into a format directly usable within lighting software.
//!
//! This script is meant as a *starting point* for using KatVisualizer for lighting control, and should be customized by the user to fit their needs.
//!
//! ```cargo
//! [dependencies]
//! anyhow = "1.0.100"
//! clap = { version = "4.5.51", features = ["derive"] }
//! rosc = "0.11.4"
//! ```
use clap::Parser;
use rosc::{OscBundle, OscMessage, OscPacket, OscTime, OscType, decoder, encoder};

use std::{
    collections::VecDeque,
    net::{SocketAddr, UdpSocket},
    time::{Duration, Instant},
};

struct Handler {
    normalization_data: VecDeque<(f64, Duration, f64)>,
    normalization_duration: Duration,
    agc_length: Duration,
    agc_target_minimum: f64,
    above_masking: f64,
    below_masking: f64,
    above_mean_stm: f32,

    frequency_scale: Vec<(f32, f32, f32)>,
    active_bins: Vec<bool>,
}

impl Handler {
    fn new(config: &Args) -> Self {
        let scale_bins =
            config.frequency_scale_bins.max(3) - if config.aggregate_end { 1 } else { 0 };

        let scale_minimum = scale_erb(20.0);
        let scale_maximum = scale_erb(if config.aggregate_end {
            6000.0
        } else {
            20000.0
        });
        let target_max = (scale_bins - 1) as f32;

        let mut frequency_scale: Vec<(f32, f32, f32)> = (0..scale_bins)
            .map(|i| {
                let i = i as f32;

                (
                    inv_scale_erb(map_value_f32(
                        i - 0.5,
                        0.0,
                        target_max,
                        scale_minimum,
                        scale_maximum,
                    )),
                    inv_scale_erb(map_value_f32(
                        i,
                        0.0,
                        target_max,
                        scale_minimum,
                        scale_maximum,
                    )),
                    inv_scale_erb(map_value_f32(
                        i + 0.5,
                        0.0,
                        target_max,
                        scale_minimum,
                        scale_maximum,
                    )),
                )
            })
            .collect();

        if config.aggregate_end {
            frequency_scale.push((6000.0, 8000.0, 20000.0));
        }

        Self {
            normalization_data: VecDeque::with_capacity(8192),
            normalization_duration: Duration::ZERO,
            agc_length: Duration::from_secs_f32(config.agc_length),
            agc_target_minimum: config.agc_target_minimum as f64,
            above_masking: config.above_masking as f64,
            below_masking: config.below_masking as f64,
            above_mean_stm: if config.frequency_scale_bins >= 32 {
                if config.above_mean_stm > 0.0 {
                    config.above_mean_stm
                } else {
                    f32::NEG_INFINITY
                }
            } else {
                f32::NEG_INFINITY
            },
            frequency_scale,
            active_bins: vec![false; config.frequency_scale_bins.max(3) as usize],
        }
    }
    fn get_normalization_targets(
        &mut self,
        masking_mean: f32,
        mean: f32,
        duration: Duration,
    ) -> (f32, f32, f32) {
        self.normalization_data.push_back((
            masking_mean as f64,
            duration,
            (mean - masking_mean) as f64,
        ));
        self.normalization_duration += duration;

        while self.normalization_duration > self.agc_length {
            let front = self.normalization_data.pop_front().unwrap();
            self.normalization_duration -= front.1;
        }

        let target = (self
            .normalization_data
            .iter()
            .map(|(d, l, _)| d * l.as_secs_f64())
            .sum::<f64>()
            / self.normalization_duration.as_secs_f64())
        .max(self.agc_target_minimum);

        let target_stm = self
            .normalization_data
            .iter()
            .map(|(_, l, d)| d * l.as_secs_f64())
            .sum::<f64>()
            / self.normalization_duration.as_secs_f64();

        (
            (target - self.below_masking) as f32,
            (target + self.above_masking) as f32,
            target_stm as f32,
        )
    }
    fn handle_packet(&mut self, packet: OscPacket) -> anyhow::Result<Vec<OscPacket>> {
        let mut data = VisualizerData::try_from(packet).map_err(|e| anyhow::anyhow!(e))?;

        let (lower, upper, mean_stm) =
            self.get_normalization_targets(data.masking_mean, data.mean, data.duration);

        data.analysis.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let mut scale_index = 0;
        let mut bin_sums = vec![(0.0, 0.0, 0.0); self.frequency_scale.len()];

        for (frequency, _bandwidth, _pan, volume, stm) in data.analysis {
            while self.frequency_scale[scale_index].0 < frequency
                && scale_index < self.frequency_scale.len() - 1
            {
                scale_index += 1;
            }

            if volume.is_finite() {
                bin_sums[scale_index].0 += volume;
                bin_sums[scale_index].1 += stm;
                bin_sums[scale_index].2 += 1.0;
            }
        }

        let mut scale_amplitudes = vec![f32::NEG_INFINITY; self.frequency_scale.len()];
        let mut scale_stms = vec![f32::NEG_INFINITY; self.frequency_scale.len()];

        for (i, (volume_sum, stm_sum, count)) in bin_sums.into_iter().enumerate() {
            scale_amplitudes[i] = volume_sum / count;
            scale_stms[i] = stm_sum / count;
        }

        let stm_threshold = mean_stm + self.above_mean_stm;

        for (i, stm) in scale_stms.into_iter().enumerate() {
            if self.active_bins[i] {
                if stm < (stm_threshold - 3.0) {
                    self.active_bins[i] = false;
                }
            } else {
                if stm > stm_threshold {
                    self.active_bins[i] = true;
                }
            }
        }

        scale_amplitudes.iter_mut().enumerate().for_each(|(i, a)| {
            *a = if self.active_bins[i] || self.frequency_scale[i].0 >= 6000.0 {
                if self.frequency_scale[i].0 <= 300.0 {
                    map_value_f32(*a as f32, lower - 4.0, upper - 4.0, 0.0, 1.0).clamp(0.0, 1.0)
                } else {
                    map_value_f32(*a as f32, lower, upper, 0.0, 1.0).clamp(0.0, 1.0)
                }
            } else {
                0.0
            };

            if *a > 0.8 {
                print!("█");
            } else if *a > 0.6 {
                print!("▓");
            } else if *a > 0.4 {
                print!("▒");
            } else if *a > 0.2 {
                print!("░");
            } else {
                print!(" ");
            }
        });

        print!("\n");

        let frequency_messages = self
            .frequency_scale
            .iter()
            .enumerate()
            .map(|(i, (_, f, _))| {
                OscPacket::Message(OscMessage {
                    addr: ["/tones/", &(i + 1).to_string(), "/frequency"].concat(),
                    args: vec![OscType::Float(*f)],
                })
            });

        let amplitude_messages = scale_amplitudes.into_iter().enumerate().map(|(i, v)| {
            OscPacket::Message(OscMessage {
                addr: ["/tones/", &(i + 1).to_string(), "/amplitude"].concat(),
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

    /// The length of the automatic gain control used to normalize bin values between 0 and 1.
    #[arg(long, default_value_t = 0.128)]
    agc_length: f32,

    /// The minimum value that the AGC can select as its target (the target is the masking threshold mean).
    #[arg(long, default_value_t = 15.0)]
    agc_target_minimum: f32,

    /// The value (relative to the masking threshold mean) that maps to a bin value of 1.
    #[arg(long, default_value_t = 30.0)]
    above_masking: f32,

    /// The value (relative to the masking threshold mean) that maps to a bin value of 0.
    #[arg(long, default_value_t = -10.0)]
    below_masking: f32,

    /// The minimum signal-to-masking-threshold ratio (relative to the mean for the agc-length) for a bin to count as valid. Disabled if set to 0.
    ///
    /// Ignored if frequency_scale_bins < 32, only applied for bins <6kHz
    #[arg(long, default_value_t = 1.5)]
    above_mean_stm: f32,

    /// The number of frequency bins to output
    #[arg(long, default_value_t = 128)]
    frequency_scale_bins: u16,

    /// If this is enabled, all frequencies >6kHz will be aggregated into a single bin
    #[arg(long, default_value_t = false)]
    aggregate_end: bool,
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

const FORMAT_VERSION: &str = "v0.12.13";

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
                        || analysis.addr
                            != ["/katvisualizer/", FORMAT_VERSION, "/frequencies"].concat()
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

fn scale_erb(x: f32) -> f32 {
    21.4 * (1.0 + 0.00437 * x).log10()
}

fn inv_scale_erb(x: f32) -> f32 {
    (1.0 / 0.00437) * ((10.0_f32.powf(x / 21.4)) - 1.0)
}
