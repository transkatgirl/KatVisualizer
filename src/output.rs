use std::time::Duration;

use crate::{
    AnalysisMetrics,
    analyzer::{BetterAnalysis, Spectrogram},
};

/// Destination for a completed analyzer result. Both chain modes use this
/// interface, regardless of whether the slice is rendered directly or sent to
/// another thread.
pub(crate) trait AnalysisSink {
    fn submit(
        &mut self,
        duration: Duration,
        update: impl FnOnce(&mut BetterAnalysis) -> AnalysisMetrics,
    );
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) struct DirectAnalysisSink<'a> {
    pub(crate) spectrogram: &'a mut Spectrogram,
    pub(crate) metrics: &'a mut AnalysisMetrics,
}

#[cfg(any(target_arch = "wasm32", test))]
impl AnalysisSink for DirectAnalysisSink<'_> {
    fn submit(
        &mut self,
        _duration: Duration,
        update: impl FnOnce(&mut BetterAnalysis) -> AnalysisMetrics,
    ) {
        let mut metrics = None;
        self.spectrogram
            .update_with(|analysis| metrics = Some(update(analysis)));
        *self.metrics = metrics.expect("spectrogram update always runs");
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    use rtrb::{Consumer, Producer, PushError, RingBuffer};

    use super::*;

    pub(crate) const TRANSPORT_CAPACITY: usize = 2048;
    const MAX_QUEUED_NS: u64 = 500_000_000;

    pub(crate) struct AnalysisPacket {
        analysis: BetterAnalysis,
        metrics: AnalysisMetrics,
        generation: u64,
        start_ns: u64,
        end_ns: u64,
    }

    struct Timeline {
        generation: AtomicU64,
        analyzed_ns: AtomicU64,
        queued_ns: AtomicU64,
    }

    pub(crate) struct NativeAnalysisSink {
        packets: Producer<AnalysisPacket>,
        recycle: Consumer<BetterAnalysis>,
        spare: Option<BetterAnalysis>,
        timeline: Arc<Timeline>,
        next_ns: u64,
    }

    pub(crate) struct NativeAnalysisReceiver {
        packets: Consumer<AnalysisPacket>,
        recycle: Producer<BetterAnalysis>,
        timeline: Arc<Timeline>,
        generation: u64,
        rendered_ns: u64,
    }

    pub(crate) fn native_transport(
        slice_capacity: usize,
    ) -> (NativeAnalysisSink, NativeAnalysisReceiver) {
        let (packets, packet_consumer) = RingBuffer::new(TRANSPORT_CAPACITY);
        let (mut recycle_producer, recycle) = RingBuffer::new(TRANSPORT_CAPACITY);

        for _ in 0..TRANSPORT_CAPACITY {
            recycle_producer
                .push(BetterAnalysis::new(slice_capacity))
                .expect("new recycle ring has enough capacity");
        }

        let timeline = Arc::new(Timeline {
            generation: AtomicU64::new(0),
            analyzed_ns: AtomicU64::new(0),
            queued_ns: AtomicU64::new(0),
        });

        (
            NativeAnalysisSink {
                packets,
                recycle,
                spare: None,
                timeline: timeline.clone(),
                next_ns: 0,
            },
            NativeAnalysisReceiver {
                packets: packet_consumer,
                recycle: recycle_producer,
                timeline,
                generation: 0,
                rendered_ns: 0,
            },
        )
    }

    impl NativeAnalysisSink {
        /// Starts a new logical stream without touching either ring on the audio
        /// thread. The renderer performs the bounded flush.
        pub(crate) fn reset_stream(&self) {
            self.timeline.generation.fetch_add(1, Ordering::AcqRel);
        }

        #[cfg(test)]
        pub(crate) fn recycle_supply(&self) -> usize {
            self.recycle.slots() + usize::from(self.spare.is_some())
        }
    }

    impl AnalysisSink for NativeAnalysisSink {
        fn submit(
            &mut self,
            duration: Duration,
            update: impl FnOnce(&mut BetterAnalysis) -> AnalysisMetrics,
        ) {
            let duration_ns = duration.as_nanos().min(u64::MAX as u128) as u64;
            let start_ns = self.next_ns;
            let end_ns = start_ns.saturating_add(duration_ns);
            self.next_ns = end_ns;

            let generation = self.timeline.generation.load(Ordering::Acquire);
            let queued_ns = self.timeline.queued_ns.load(Ordering::Acquire);

            if queued_ns.saturating_add(duration_ns) <= MAX_QUEUED_NS
                && let Some(mut analysis) = self.spare.take().or_else(|| self.recycle.pop().ok())
            {
                let metrics = update(&mut analysis);

                self.timeline
                    .queued_ns
                    .fetch_add(duration_ns, Ordering::AcqRel);

                let packet = AnalysisPacket {
                    analysis,
                    metrics,
                    generation,
                    start_ns,
                    end_ns,
                };

                if let Err(PushError::Full(packet)) = self.packets.push(packet) {
                    self.timeline
                        .queued_ns
                        .fetch_sub(duration_ns, Ordering::AcqRel);
                    self.spare = Some(packet.analysis);
                }
            }

            // When a packet was pushed, this is published afterwards. An acquire
            // load by the renderer therefore observes every packet in its snapshot.
            self.timeline.analyzed_ns.store(end_ns, Ordering::Release);
        }
    }

    impl NativeAnalysisReceiver {
        fn recycle(&mut self, analysis: BetterAnalysis) {
            if let Err(PushError::Full(_analysis)) = self.recycle.push(analysis) {
                debug_assert!(false, "analysis recycle ring overflowed");
            }
        }

        fn flush(&mut self) -> u64 {
            let mut newest_end_ns = 0;
            let packets_to_flush = self.packets.slots();

            for _ in 0..packets_to_flush {
                let Ok(packet) = self.packets.pop() else {
                    break;
                };

                self.timeline.queued_ns.fetch_sub(
                    packet.end_ns.saturating_sub(packet.start_ns),
                    Ordering::AcqRel,
                );
                newest_end_ns = newest_end_ns.max(packet.end_ns);
                self.recycle(packet.analysis);
            }
            newest_end_ns
        }

        /// Implements editor fresh-start semantics and is also used for stream
        /// generation changes.
        pub(crate) fn fresh_start(&mut self, spectrogram: &mut Spectrogram) {
            let generation = self.timeline.generation.load(Ordering::Acquire);
            let watermark_before = self.timeline.analyzed_ns.load(Ordering::Acquire);
            let flushed_end = self.flush();
            let watermark_after = self.timeline.analyzed_ns.load(Ordering::Acquire);
            self.generation = generation;
            self.rendered_ns = watermark_before.max(flushed_end).max(watermark_after);
            spectrogram.clear();
        }

        pub(crate) fn invalidate_history(&mut self, spectrogram: &mut Spectrogram) {
            self.timeline.generation.fetch_add(1, Ordering::AcqRel);
            self.fresh_start(spectrogram);
        }

        /// Drains all output visible in the current analyzed-time snapshot and
        /// returns the queued analyzed duration measured before draining.
        pub(crate) fn drain_into(
            &mut self,
            spectrogram: &mut Spectrogram,
            metrics: &mut AnalysisMetrics,
        ) -> Duration {
            let current_generation = self.timeline.generation.load(Ordering::Acquire);
            if current_generation != self.generation {
                self.fresh_start(spectrogram);
            }

            let queued_before = self.timeline.queued_ns.load(Ordering::Acquire);
            let watermark = self.timeline.analyzed_ns.load(Ordering::Acquire);
            // Take the packet snapshot after the watermark. Publishing the
            // watermark happens after pushing its packet, so this budget includes
            // everything represented by the watermark while bounding this frame's
            // work even if the producer keeps adding packets.
            let packets_to_drain = self.packets.slots();

            for _ in 0..packets_to_drain {
                let Ok(packet) = self.packets.pop() else {
                    break;
                };

                self.timeline.queued_ns.fetch_sub(
                    packet.end_ns.saturating_sub(packet.start_ns),
                    Ordering::AcqRel,
                );

                if packet.generation != self.generation || packet.end_ns <= self.rendered_ns {
                    self.recycle(packet.analysis);
                    continue;
                }

                if packet.start_ns > self.rendered_ns {
                    spectrogram.insert_blank_span(
                        Duration::from_nanos(packet.start_ns - self.rendered_ns),
                        packet.analysis.duration,
                        packet.analysis.data.len(),
                    );
                }

                self.rendered_ns = packet.end_ns;
                *metrics = packet.metrics;
                let evicted = spectrogram.rotate_in(packet.analysis);
                self.recycle(evicted);
            }

            if watermark > self.rendered_ns {
                let newest = spectrogram.newest();
                let slice_duration = newest.duration;
                let bins = newest.data.len();
                spectrogram.insert_blank_span(
                    Duration::from_nanos(watermark - self.rendered_ns),
                    slice_duration,
                    bins,
                );
                self.rendered_ns = watermark;
            }

            Duration::from_nanos(queued_before)
        }

        #[cfg(test)]
        pub(crate) fn rendered_ns(&self) -> u64 {
            self.rendered_ns
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{NativeAnalysisReceiver, NativeAnalysisSink, native_transport};

#[cfg(test)]
mod tests {
    use super::*;

    fn submit(sink: &mut NativeAnalysisSink, id: f32, duration: Duration) {
        sink.submit(duration, |analysis| {
            let data_pointer = analysis.data.as_ptr();
            let masking_pointer = analysis.masking.as_ptr();
            analysis.data.resize(8, (id, id));
            analysis.masking.resize(8, (id, id));
            assert_eq!(analysis.data.as_ptr(), data_pointer);
            assert_eq!(analysis.masking.as_ptr(), masking_pointer);
            analysis.data.fill((id, id));
            analysis.masking.fill((id, id));
            analysis.masking_mean = id;
            analysis.duration = duration;
            AnalysisMetrics {
                processing: duration,
            }
        });
    }

    #[test]
    fn direct_sink_updates_render_owned_spectrogram() {
        let mut spectrogram = Spectrogram::new(3, 8);
        let mut metrics = AnalysisMetrics::default();
        let mut sink = DirectAnalysisSink {
            spectrogram: &mut spectrogram,
            metrics: &mut metrics,
        };

        sink.submit(Duration::from_millis(10), |analysis| {
            analysis.duration = Duration::from_millis(10);
            analysis.data.fill((0.5, 6.0));
            AnalysisMetrics {
                processing: Duration::from_millis(2),
            }
        });

        assert_eq!(spectrogram.newest().data[0], (0.5, 6.0));
        assert_eq!(metrics.processing, Duration::from_millis(2));
    }

    #[test]
    fn native_fifo_delivery_and_bidirectional_recycling() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(8, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        submit(&mut sink, 1.0, Duration::from_millis(10));
        submit(&mut sink, 2.0, Duration::from_millis(10));
        submit(&mut sink, 3.0, Duration::from_millis(10));

        let queued = receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(queued, Duration::from_millis(30));
        assert_eq!(
            spectrogram
                .newest_to_oldest()
                .take(3)
                .map(|row| row.data[0].0)
                .collect::<Vec<_>>(),
            vec![3.0, 2.0, 1.0]
        );
        assert_eq!(metrics.processing, Duration::from_millis(10));
        assert_eq!(receiver.rendered_ns(), 30_000_000);
        assert_eq!(sink.recycle_supply(), native::TRANSPORT_CAPACITY);
    }

    #[test]
    fn backlog_limit_places_trailing_and_interior_blanks() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(16, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        for id in 1..=7 {
            submit(&mut sink, id as f32, Duration::from_millis(100));
        }

        receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(spectrogram.at_age(0).unwrap().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(1).unwrap().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(2).unwrap().data[0].0, 5.0);

        // A later real packet must appear after the already-rendered missing
        // interval, leaving those blank rows correctly positioned in history.
        submit(&mut sink, 8.0, Duration::from_millis(100));
        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(spectrogram.at_age(0).unwrap().data[0].0, 8.0);
        assert_eq!(spectrogram.at_age(1).unwrap().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(2).unwrap().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(3).unwrap().data[0].0, 5.0);
    }

    #[test]
    fn hard_capacity_exhaustion_advances_time_without_allocating() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(32, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        for id in 0..=native::TRANSPORT_CAPACITY {
            submit(&mut sink, id as f32, Duration::from_nanos(1));
        }

        receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(receiver.rendered_ns(), 2049);
        assert_eq!(spectrogram.newest().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(1).unwrap().data[0].0, 2047.0);
        assert_eq!(sink.recycle_supply(), native::TRANSPORT_CAPACITY);
    }

    #[test]
    fn fresh_start_and_generation_reset_discard_queued_packets() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(8, 8);
        let mut metrics = AnalysisMetrics::default();

        submit(&mut sink, 1.0, Duration::from_millis(10));
        receiver.fresh_start(&mut spectrogram);
        assert_eq!(spectrogram.newest().data[0].1, f32::NEG_INFINITY);

        submit(&mut sink, 2.0, Duration::from_millis(10));
        sink.reset_stream();
        submit(&mut sink, 3.0, Duration::from_millis(10));
        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(spectrogram.newest().data[0].1, f32::NEG_INFINITY);

        submit(&mut sink, 4.0, Duration::from_millis(10));
        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(spectrogram.newest().data[0].0, 4.0);
    }

    #[test]
    fn native_transport_remains_ordered_during_concurrent_handoff() {
        const PACKETS: u64 = 1024;

        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(PACKETS as usize, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        let sink = std::thread::scope(|scope| {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            let producer = scope.spawn(move || {
                for id in 1..=PACKETS {
                    submit(&mut sink, id as f32, Duration::from_nanos(1));
                }
                sink
            });

            while receiver.rendered_ns() < PACKETS {
                receiver.drain_into(&mut spectrogram, &mut metrics);
                assert!(
                    std::time::Instant::now() < deadline,
                    "analysis handoff timed out"
                );
                std::thread::yield_now();
            }

            producer.join().expect("analysis producer did not panic")
        });

        assert_eq!(spectrogram.newest().data[0].0, PACKETS as f32);
        assert_eq!(receiver.rendered_ns(), PACKETS);
        assert_eq!(sink.recycle_supply(), native::TRANSPORT_CAPACITY);
    }
}
