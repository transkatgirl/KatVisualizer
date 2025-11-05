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
    time::Duration,
};

struct Handler {
    normalization_data: VecDeque<(f64, Duration)>,
    normalization_duration: Duration,
    agc_length: Duration,
    above_masking: f64,
    below_masking: f64,
}

impl Handler {
    fn new(config: &Args) -> Self {
        Self {
            normalization_data: VecDeque::with_capacity(8192),
            normalization_duration: Duration::ZERO,
            agc_length: Duration::from_secs_f32(config.agc_length),
            above_masking: config.above_masking as f64,
            below_masking: config.below_masking as f64,
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

        let mean = self
            .normalization_data
            .iter()
            .map(|(d, l)| d * l.as_secs_f64())
            .sum::<f64>()
            / self.normalization_duration.as_secs_f64();

        (
            (mean - self.below_masking) as f32,
            (mean + self.above_masking) as f32,
        )
    }
    fn handle_packet(&mut self, packet: OscPacket) -> anyhow::Result<Vec<OscPacket>> {
        let data = VisualizerData::try_from(packet).map_err(|e| anyhow::anyhow!(e))?;

        let (lower, upper) = self.get_normalization_targets(data.masking_mean, data.duration);

        // Magnitude should range from 0 to 1
        // Buttons should be either float(1) or float(0)

        Ok(vec![OscPacket::Bundle(OscBundle {
            timetag: data.timetag,
            content: vec![OscPacket::Message(OscMessage {
                addr: "/tones".to_string(),
                args: vec![OscType::Array(OscArray {
                    content: data
                        .analysis
                        .into_iter()
                        .map(|(f, _b, _p, v, _stm)| {
                            OscType::Array(OscArray {
                                content: vec![
                                    OscType::Float(f),
                                    OscType::Float(
                                        map_value_f32(v, lower, upper, 0.0, 1.0).clamp(0.0, 1.0),
                                    ),
                                ],
                            })
                        })
                        .collect(),
                })],
            })],
        })])
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

    #[arg(long, default_value_t = 1.0)]
    agc_length: f32,

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

        match decoder::decode_udp(received) {
            Ok((_, packet)) => match handler.handle_packet(packet) {
                Ok(responses) => {
                    for response in responses {
                        write_buf.clear();
                        encoder::encode_into(&response, &mut write_buf)?;
                        socket.send_to(&write_buf, args.destination_address)?;
                    }
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

pub fn map_value_f32(x: f32, min: f32, max: f32, target_min: f32, target_max: f32) -> f32 {
    (x - min) / (max - min) * (target_max - target_min) + target_min
}
