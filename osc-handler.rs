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

use std::net::{SocketAddr, UdpSocket};

struct Handler {}

impl Handler {
    fn new() -> Self {
        Self {}
    }
    fn handle_packet(&mut self, packet: OscPacket) -> anyhow::Result<Vec<OscPacket>> {
        let data = VisualizerData::try_from(packet).map_err(|e| anyhow::anyhow!(e))?;

        Ok(vec![OscPacket::Bundle(OscBundle {
            timetag: data.timetag,
            content: vec![
                OscPacket::Message(OscMessage {
                    addr: "/katvisualizer/stats".to_string(),
                    args: vec![
                        OscType::Float(data.masking_mean),
                        OscType::Float(data.mean),
                        OscType::Float(data.max),
                    ],
                }),
                OscPacket::Message(OscMessage {
                    addr: "/katvisualizer/tones".to_string(),
                    args: vec![OscType::Array(OscArray {
                        content: data
                            .analysis
                            .into_iter()
                            .map(|(f, p, v, stm)| {
                                OscType::Array(OscArray {
                                    content: vec![
                                        OscType::Float(f),
                                        OscType::Float(p),
                                        OscType::Float(v),
                                        OscType::Float(stm),
                                    ],
                                })
                            })
                            .collect(),
                    })],
                }),
            ],
        })])
    }
}

const FORMAT_VERSION: &str = "v0.8.4";

struct VisualizerData {
    timetag: OscTime,
    masking_mean: f32,
    mean: f32,
    max: f32,
    analysis: Vec<(f32, f32, f32, f32)>,
}

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

                    if metadata.args.len() != 3 || analysis.args.len() != 1 {
                        return Err("One or more message arguments are malformed");
                    }

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

                        if tone_array.len() != 4 {
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
                        let frequency = tone_array
                            .pop()
                            .unwrap()
                            .float()
                            .ok_or("One or more message arguments are malformed")?;

                        analysis.push((frequency, pan, volume, signal_mask_ratio))
                    }

                    Ok(Self {
                        timetag: bundle.timetag,
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
}

const MAX_PACKET_SIZE: usize = 32768;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let socket = UdpSocket::bind(args.listen_address)?;
    let mut read_buf = [0; MAX_PACKET_SIZE];
    let mut write_buf = Vec::with_capacity(MAX_PACKET_SIZE);

    let mut handler = Handler::new();

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
