use std::time::Duration;

use crate::{
    AnalysisMetrics,
    analyzer::{BetterAnalysis, Spectrogram},
};

/// Destination for a completed analyzer result. Both chain modes use this
/// interface, regardless of whether the slice is rendered directly or sent to
/// another thread. A chain brackets all slices produced by one process call in
/// a batch so native timeline state can be published once at the block boundary.
pub(crate) trait AnalysisSink {
    fn begin_batch(&mut self) {}

    fn submit(
        &mut self,
        duration: Duration,
        update: impl FnOnce(&mut BetterAnalysis) -> AnalysisMetrics,
    );

    fn finish_batch(&mut self) {}
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

    use crossbeam_utils::CachePadded;
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
        generation_start_ns: AtomicU64,
        analyzed_ns: AtomicU64,
        enqueued_ns: CachePadded<AtomicU64>,
        consumed_ns: CachePadded<AtomicU64>,
    }

    pub(crate) struct NativeAnalysisSink {
        packets: Producer<AnalysisPacket>,
        recycle: Consumer<BetterAnalysis>,
        spare: Option<BetterAnalysis>,
        timeline: Arc<Timeline>,
        generation: u64,
        generation_start_ns: u64,
        next_ns: u64,
        enqueued_ns: u64,
        batch_consumed_ns: u64,
        batch_start_enqueued_ns: u64,
        batch_analyzed_ns: Option<u64>,
        batch_active: bool,
        #[cfg(test)]
        enqueued_publications: u64,
        #[cfg(test)]
        analyzed_publications: u64,
    }

    pub(crate) struct NativeAnalysisReceiver {
        packets: Consumer<AnalysisPacket>,
        recycle: Producer<BetterAnalysis>,
        timeline: Arc<Timeline>,
        generation: u64,
        rendered_ns: u64,
        consumed_ns: u64,
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
            generation_start_ns: AtomicU64::new(0),
            analyzed_ns: AtomicU64::new(0),
            enqueued_ns: CachePadded::new(AtomicU64::new(0)),
            consumed_ns: CachePadded::new(AtomicU64::new(0)),
        });

        (
            NativeAnalysisSink {
                packets,
                recycle,
                spare: None,
                timeline: timeline.clone(),
                generation: 0,
                generation_start_ns: 0,
                next_ns: 0,
                enqueued_ns: 0,
                batch_consumed_ns: 0,
                batch_start_enqueued_ns: 0,
                batch_analyzed_ns: None,
                batch_active: false,
                #[cfg(test)]
                enqueued_publications: 0,
                #[cfg(test)]
                analyzed_publications: 0,
            },
            NativeAnalysisReceiver {
                packets: packet_consumer,
                recycle: recycle_producer,
                timeline,
                generation: 0,
                rendered_ns: 0,
                consumed_ns: 0,
            },
        )
    }

    impl NativeAnalysisSink {
        /// Starts a new logical stream without touching either ring on the audio
        /// thread. The renderer performs the bounded flush.
        pub(crate) fn reset_stream(&mut self) {
            self.generation = self.generation.wrapping_add(1);
            self.generation_start_ns = self.next_ns;
        }

        #[cfg(test)]
        pub(crate) fn recycle_supply(&self) -> usize {
            self.recycle.slots() + usize::from(self.spare.is_some())
        }

        #[cfg(test)]
        pub(crate) fn generation(&self) -> u64 {
            self.generation
        }

        #[cfg(test)]
        pub(crate) fn publication_counts(&self) -> (u64, u64) {
            (self.enqueued_publications, self.analyzed_publications)
        }
    }

    impl AnalysisSink for NativeAnalysisSink {
        fn begin_batch(&mut self) {
            debug_assert!(!self.batch_active, "analysis sink batch was already active");
            self.batch_consumed_ns = self.timeline.consumed_ns.load(Ordering::Relaxed);
            self.batch_start_enqueued_ns = self.enqueued_ns;
            self.batch_analyzed_ns = None;
            self.batch_active = true;
        }

        fn submit(
            &mut self,
            duration: Duration,
            update: impl FnOnce(&mut BetterAnalysis) -> AnalysisMetrics,
        ) {
            debug_assert!(self.batch_active, "analysis submitted outside a batch");
            let duration_ns = duration.as_nanos().min(u64::MAX as u128) as u64;
            let start_ns = self.next_ns;
            let end_ns = start_ns.saturating_add(duration_ns);
            self.next_ns = end_ns;

            let queued_ns = self.enqueued_ns.saturating_sub(self.batch_consumed_ns);

            if queued_ns.saturating_add(duration_ns) <= MAX_QUEUED_NS
                && let Some(mut analysis) = self.spare.take().or_else(|| self.recycle.pop().ok())
            {
                let metrics = update(&mut analysis);

                let packet = AnalysisPacket {
                    analysis,
                    metrics,
                    generation: self.generation,
                    start_ns,
                    end_ns,
                };

                if let Err(PushError::Full(packet)) = self.packets.push(packet) {
                    self.spare = Some(packet.analysis);
                } else {
                    self.enqueued_ns = self.enqueued_ns.saturating_add(duration_ns);
                }
            }

            self.batch_analyzed_ns = Some(end_ns);
        }

        fn finish_batch(&mut self) {
            debug_assert!(self.batch_active, "analysis sink batch was not active");

            if self.enqueued_ns != self.batch_start_enqueued_ns {
                self.timeline
                    .enqueued_ns
                    .store(self.enqueued_ns, Ordering::Relaxed);
                #[cfg(test)]
                {
                    self.enqueued_publications += 1;
                }
            }

            if self.timeline.generation.load(Ordering::Relaxed) != self.generation {
                self.timeline
                    .generation_start_ns
                    .store(self.generation_start_ns, Ordering::Relaxed);
                // Publish the new stream only after all of its packets have been
                // pushed. The receiver preserves packets from this generation.
                self.timeline
                    .generation
                    .store(self.generation, Ordering::Release);
            }

            // All packet pushes and the enqueued total are published before this
            // release. The renderer uses the acquired analyzed time as its packet
            // visibility watermark.
            if let Some(analyzed_ns) = self.batch_analyzed_ns {
                self.timeline
                    .analyzed_ns
                    .store(analyzed_ns, Ordering::Release);
                #[cfg(test)]
                {
                    self.analyzed_publications += 1;
                }
            }

            self.batch_active = false;
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
            let consumed_before = self.consumed_ns;

            for _ in 0..packets_to_flush {
                let Ok(packet) = self.packets.pop() else {
                    break;
                };

                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));
                newest_end_ns = newest_end_ns.max(packet.end_ns);
                self.recycle(packet.analysis);
            }
            if self.consumed_ns != consumed_before {
                self.timeline
                    .consumed_ns
                    .store(self.consumed_ns, Ordering::Relaxed);
            }
            newest_end_ns
        }

        fn start_generation(&mut self, generation: u64, spectrogram: &mut Spectrogram) {
            let consumed_before = self.consumed_ns;

            // Generations are FIFO. Remove only obsolete packets and leave the
            // first packet from this or a later published generation in the
            // ring. The wrapping comparison also handles generation overflow.
            loop {
                let Ok(packet) = self.packets.peek() else {
                    break;
                };
                let generation_distance = packet.generation.wrapping_sub(generation);
                if generation_distance < (1_u64 << 63) {
                    break;
                }

                let packet = self
                    .packets
                    .pop()
                    .expect("peeked analysis packet is available");
                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));
                self.recycle(packet.analysis);
            }

            if self.consumed_ns != consumed_before {
                self.timeline
                    .consumed_ns
                    .store(self.consumed_ns, Ordering::Relaxed);
            }

            self.generation = generation;
            self.rendered_ns = self.timeline.generation_start_ns.load(Ordering::Relaxed);
            spectrogram.clear();
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

        /// Drains all output visible in the current analyzed-time snapshot.
        pub(crate) fn drain_into(
            &mut self,
            spectrogram: &mut Spectrogram,
            metrics: &mut AnalysisMetrics,
        ) {
            let watermark = loop {
                let current_generation = self.timeline.generation.load(Ordering::Acquire);
                if current_generation != self.generation {
                    self.start_generation(current_generation, spectrogram);
                }

                let watermark = self.timeline.analyzed_ns.load(Ordering::Acquire);
                // The producer publishes a generation just before its analyzed
                // watermark. Retry if those two loads straddled that boundary.
                if self.timeline.generation.load(Ordering::Acquire) == self.generation {
                    break watermark;
                }
            };
            // Bound this frame's work, and never consume a packet from a batch
            // whose analyzed-time watermark has not yet been published.
            let packets_to_drain = self.packets.slots();
            let consumed_before = self.consumed_ns;

            for _ in 0..packets_to_drain {
                let Ok(packet) = self.packets.peek() else {
                    break;
                };
                if packet.end_ns > watermark {
                    break;
                }

                let Ok(packet) = self.packets.pop() else {
                    break;
                };

                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));

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

            if self.consumed_ns != consumed_before {
                self.timeline
                    .consumed_ns
                    .store(self.consumed_ns, Ordering::Relaxed);
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

    fn submit_unpublished(sink: &mut NativeAnalysisSink, id: f32, duration: Duration) {
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

    fn submit(sink: &mut NativeAnalysisSink, id: f32, duration: Duration) {
        sink.begin_batch();
        submit_unpublished(sink, id, duration);
        sink.finish_batch();
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

        receiver.drain_into(&mut spectrogram, &mut metrics);
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
    fn native_batch_publishes_once_and_hides_in_progress_packets() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(8, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        sink.begin_batch();
        submit_unpublished(&mut sink, 1.0, Duration::from_millis(10));
        submit_unpublished(&mut sink, 2.0, Duration::from_millis(10));
        submit_unpublished(&mut sink, 3.0, Duration::from_millis(10));

        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(receiver.rendered_ns(), 0);
        assert_eq!(sink.publication_counts(), (0, 0));

        sink.finish_batch();
        assert_eq!(sink.publication_counts(), (1, 1));
        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(
            spectrogram
                .newest_to_oldest()
                .take(3)
                .map(|row| row.data[0].0)
                .collect::<Vec<_>>(),
            vec![3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn backlog_limit_places_trailing_and_interior_blanks() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(16, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);

        sink.begin_batch();
        for id in 1..=7 {
            submit_unpublished(&mut sink, id as f32, Duration::from_millis(100));
        }
        sink.finish_batch();

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

        sink.begin_batch();
        for id in 0..=native::TRANSPORT_CAPACITY {
            submit_unpublished(&mut sink, id as f32, Duration::from_nanos(1));
        }
        sink.finish_batch();

        receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(receiver.rendered_ns(), 2049);
        assert_eq!(spectrogram.newest().data[0].1, f32::NEG_INFINITY);
        assert_eq!(spectrogram.at_age(1).unwrap().data[0].0, 2047.0);
        assert_eq!(sink.recycle_supply(), native::TRANSPORT_CAPACITY);
    }

    #[test]
    fn fresh_start_discards_old_packets_and_generation_reset_preserves_new_packets() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(8, 8);
        let mut metrics = AnalysisMetrics::default();

        submit(&mut sink, 1.0, Duration::from_millis(10));
        receiver.fresh_start(&mut spectrogram);
        assert_eq!(spectrogram.newest().data[0].1, f32::NEG_INFINITY);

        submit(&mut sink, 2.0, Duration::from_millis(10));
        sink.reset_stream();

        sink.begin_batch();
        submit_unpublished(&mut sink, 3.0, Duration::from_millis(10));
        // The reset is not visible until the batch is complete, so the receiver
        // cannot mistake this new-generation packet for obsolete history.
        receiver.drain_into(&mut spectrogram, &mut metrics);
        sink.finish_batch();

        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(spectrogram.newest().data[0].0, 3.0);

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
