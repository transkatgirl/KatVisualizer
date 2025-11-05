use std::{
    net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, UdpSocket},
    time::SystemTime,
};

use egui_extras::syntax_highlighting::SyntectSettings;
use rhai::{AST, Engine, Scope};
use rosc::{
    OscBundle, OscMessage, OscPacket, OscTime,
    encoder::{self},
};
use syntect::{
    highlighting::ThemeSet,
    parsing::{SyntaxDefinition, SyntaxSetBuilder},
};

use crate::MAX_PEAK_OUTPUTS;

const RHAI_SYNTAX_DEF: &str = include_str!("rhai.sublime-syntax");

pub fn get_highlighting_settings() -> SyntectSettings {
    let mut builder = SyntaxSetBuilder::new();
    builder.add(SyntaxDefinition::load_from_str(RHAI_SYNTAX_DEF, true, None).unwrap());

    SyntectSettings {
        ps: builder.build(),
        ts: ThemeSet::load_defaults(),
    }
}

fn new_socket(address: SocketAddr) -> Result<UdpSocket, std::io::Error> {
    UdpSocket::bind(match address {
        SocketAddr::V4(addr) => SocketAddr::V4(SocketAddrV4::new(
            if addr.ip().is_loopback() {
                *addr.ip()
            } else {
                Ipv4Addr::UNSPECIFIED
            },
            0,
        )),
        SocketAddr::V6(addr) => SocketAddr::V6(SocketAddrV6::new(
            if addr.ip().is_loopback() {
                Ipv6Addr::LOCALHOST
            } else {
                Ipv6Addr::UNSPECIFIED
            },
            0,
            0,
            0,
        )),
    })
}

/*fn update_socket(socket: &mut UdpSocket, address: SocketAddr) -> Result<(), std::io::Error> {
    if let Ok(socket_address) = socket.local_addr() {
        if address.ip().is_loopback() != socket_address.ip().is_loopback() {
            *socket = new_socket(address)?;
        } else {
            match socket_address {
                SocketAddr::V4(addr) => {
                    if !address.is_ipv4() || (addr.ip().is_loopback() && address.ip() != *addr.ip())
                    {
                        *socket = new_socket(address)?;
                    }
                }
                SocketAddr::V6(_) => {
                    if !address.is_ipv6() {
                        *socket = new_socket(address)?;
                    }
                }
            }
        }
    } else {
        *socket = new_socket(address)?;
    }

    Ok(())
}*/

pub struct OSCEngine {
    rhai_engine: Engine,
    rhai_scope: Scope<'static>,
    rhai_script: AST,
    osc_socket: Option<UdpSocket>,
    osc_buffer: Vec<u8>,
}

const OSC_MAX_PACKET_SIZE: usize = 12288;

impl OSCEngine {
    pub fn new() -> Self {
        Self {
            rhai_engine: Engine::new(),
            rhai_scope: Scope::with_capacity(MAX_PEAK_OUTPUTS * 4),
            rhai_script: AST::empty(),
            osc_socket: None,
            osc_buffer: Vec::with_capacity(OSC_MAX_PACKET_SIZE),
        }
    }

    pub fn update(&mut self, script: &str, osc_endpoint: SocketAddr) -> Result<(), String> {
        self.rhai_script = self
            .rhai_engine
            .compile(script)
            .map_err(|e| format!("Unable to parse script:\n{}", e))?;
        self.rhai_scope = Scope::with_capacity(MAX_PEAK_OUTPUTS * 4);

        self.osc_socket = Some(
            new_socket(osc_endpoint).map_err(|e| format!("Failed to create UDP socket:\n{}", e))?,
        );

        Ok(())
    }
    pub fn run(&mut self, timestamp: SystemTime) -> Result<(), String> {
        if let Some(socket) = &mut self.osc_socket {
            let bundles: Vec<Vec<OscMessage>> = self
                .rhai_engine
                .eval_ast_with_scope(&mut self.rhai_scope, &self.rhai_script)
                .map_err(|e| format!("Script evaluation error:\n{}", e))?;

            let timetag = OscTime::try_from(timestamp).unwrap();

            for bundle in bundles {
                self.osc_buffer.clear();
                encoder::encode_into(
                    &OscPacket::Bundle(OscBundle {
                        timetag,
                        content: bundle.into_iter().map(OscPacket::Message).collect(),
                    }),
                    &mut self.osc_buffer,
                )
                .map_err(|e| format!("Unable to encode OSC packet:\n{}", e))?;

                if self.osc_buffer.len() <= OSC_MAX_PACKET_SIZE {
                    socket
                        .send(&self.osc_buffer)
                        .map_err(|e| format!("Failed to send UDP packet:\n{}", e))?;
                } else {
                    return Err(format!(
                        "OSC packet of length {} exceeded buffer size of {}",
                        self.osc_buffer.len(),
                        OSC_MAX_PACKET_SIZE
                    ));
                }
            }
        }

        Ok(())
    }
}
