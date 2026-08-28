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
    use std::{
        cell::UnsafeCell,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use crossbeam_utils::CachePadded;

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

    struct ConsumerProgress {
        sequence: AtomicU64,
        consumed_ns: AtomicU64,
    }

    struct AnalysisSlotRing {
        slots: Box<[UnsafeCell<AnalysisPacket>]>,
        published_sequence: CachePadded<AtomicU64>,
        consumer: CachePadded<ConsumerProgress>,
    }

    // SAFETY: the ring has one producer and one consumer. The producer owns
    // free and unpublished slots; the consumer owns published, unconsumed
    // slots. Release/acquire publication of the two wrapping sequences transfers
    // exclusive access in each direction, and neither side accesses a slot after
    // transferring it until the opposite sequence returns ownership.
    unsafe impl Send for AnalysisSlotRing {}
    // SAFETY: all shared non-atomic slot access is covered by the ownership
    // protocol documented above.
    unsafe impl Sync for AnalysisSlotRing {}

    impl AnalysisSlotRing {
        fn new(slice_capacity: usize) -> Self {
            let slots = (0..TRANSPORT_CAPACITY)
                .map(|_| {
                    UnsafeCell::new(AnalysisPacket {
                        analysis: BetterAnalysis::new(slice_capacity),
                        metrics: AnalysisMetrics::default(),
                        generation: 0,
                        start_ns: 0,
                        end_ns: 0,
                    })
                })
                .collect();
            Self {
                slots,
                published_sequence: CachePadded::new(AtomicU64::new(0)),
                consumer: CachePadded::new(ConsumerProgress {
                    sequence: AtomicU64::new(0),
                    consumed_ns: AtomicU64::new(0),
                }),
            }
        }

        #[inline]
        fn slot(&self, sequence: u64) -> *mut AnalysisPacket {
            self.slots[sequence as usize % TRANSPORT_CAPACITY].get()
        }
    }

    struct Timeline {
        generation: AtomicU64,
        generation_start_ns: AtomicU64,
        analyzed_ns: AtomicU64,
    }

    pub(crate) struct NativeAnalysisSink {
        ring: Arc<AnalysisSlotRing>,
        timeline: Arc<Timeline>,
        write_sequence: u64,
        generation: u64,
        // Sink-local mirror: the renderer never writes `Timeline::generation`.
        published_generation: u64,
        generation_start_ns: u64,
        next_ns: u64,
        enqueued_ns: u64,
        batch_consumed_sequence: u64,
        batch_consumed_ns: u64,
        batch_start_write_sequence: u64,
        batch_analyzed_ns: Option<u64>,
        batch_active: bool,
        #[cfg(test)]
        slot_publications: u64,
        #[cfg(test)]
        analyzed_publications: u64,
    }

    pub(crate) struct NativeAnalysisReceiver {
        ring: Arc<AnalysisSlotRing>,
        timeline: Arc<Timeline>,
        read_sequence: u64,
        generation: u64,
        rendered_ns: u64,
        consumed_ns: u64,
    }

    pub(crate) fn native_transport(
        slice_capacity: usize,
    ) -> (NativeAnalysisSink, NativeAnalysisReceiver) {
        let ring = Arc::new(AnalysisSlotRing::new(slice_capacity));

        let timeline = Arc::new(Timeline {
            generation: AtomicU64::new(0),
            generation_start_ns: AtomicU64::new(0),
            analyzed_ns: AtomicU64::new(0),
        });

        (
            NativeAnalysisSink {
                ring: Arc::clone(&ring),
                timeline: timeline.clone(),
                write_sequence: 0,
                generation: 0,
                published_generation: 0,
                generation_start_ns: 0,
                next_ns: 0,
                enqueued_ns: 0,
                batch_consumed_sequence: 0,
                batch_consumed_ns: 0,
                batch_start_write_sequence: 0,
                batch_analyzed_ns: None,
                batch_active: false,
                #[cfg(test)]
                slot_publications: 0,
                #[cfg(test)]
                analyzed_publications: 0,
            },
            NativeAnalysisReceiver {
                ring,
                timeline,
                read_sequence: 0,
                generation: 0,
                rendered_ns: 0,
                consumed_ns: 0,
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn set_empty_sequence_for_test(
        sink: &mut NativeAnalysisSink,
        receiver: &mut NativeAnalysisReceiver,
        sequence: u64,
    ) {
        assert_eq!(sink.write_sequence, receiver.read_sequence);
        sink.write_sequence = sequence;
        receiver.read_sequence = sequence;
        sink.ring
            .published_sequence
            .store(sequence, Ordering::Relaxed);
        sink.ring
            .consumer
            .sequence
            .store(sequence, Ordering::Relaxed);
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
            // Every slot permanently contains one reusable analysis allocation.
            TRANSPORT_CAPACITY
        }

        #[cfg(test)]
        pub(crate) fn generation(&self) -> u64 {
            self.generation
        }

        #[cfg(test)]
        pub(crate) fn publication_counts(&self) -> (u64, u64) {
            (self.slot_publications, self.analyzed_publications)
        }

        #[cfg(test)]
        pub(crate) fn sequence(&self) -> u64 {
            self.write_sequence
        }

        #[cfg(test)]
        pub(crate) fn slot_data_pointer(&self, sequence: u64) -> *const (f32, f32) {
            // SAFETY: test callers inspect slots only while producer and
            // consumer activity is paused.
            unsafe { (&*self.ring.slot(sequence)).analysis.data.as_ptr() }
        }
    }

    impl AnalysisSink for NativeAnalysisSink {
        fn begin_batch(&mut self) {
            debug_assert!(!self.batch_active, "analysis sink batch was already active");
            // Acquiring the consumed sequence returns all newly freed slots and
            // makes the consumer's preceding consumed-time update visible.
            self.batch_consumed_sequence = self.ring.consumer.sequence.load(Ordering::Acquire);
            self.batch_consumed_ns = self.ring.consumer.consumed_ns.load(Ordering::Relaxed);
            self.batch_start_write_sequence = self.write_sequence;
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
            let occupied = self
                .write_sequence
                .wrapping_sub(self.batch_consumed_sequence);

            if queued_ns.saturating_add(duration_ns) <= MAX_QUEUED_NS
                && occupied < TRANSPORT_CAPACITY as u64
            {
                // SAFETY: `occupied < capacity` means this slot was returned by
                // the acquired consumer sequence. It remains producer-owned and
                // unpublished until `finish_batch()` release-publishes it.
                let packet = unsafe { &mut *self.ring.slot(self.write_sequence) };
                packet.metrics = update(&mut packet.analysis);
                packet.generation = self.generation;
                packet.start_ns = start_ns;
                packet.end_ns = end_ns;

                self.write_sequence = self.write_sequence.wrapping_add(1);
                self.enqueued_ns = self.enqueued_ns.saturating_add(duration_ns);
            }

            self.batch_analyzed_ns = Some(end_ns);
        }

        fn finish_batch(&mut self) {
            debug_assert!(self.batch_active, "analysis sink batch was not active");

            if self.write_sequence != self.batch_start_write_sequence {
                self.ring
                    .published_sequence
                    .store(self.write_sequence, Ordering::Release);
                #[cfg(test)]
                {
                    self.slot_publications += 1;
                }
            }

            if self.published_generation != self.generation {
                self.timeline
                    .generation_start_ns
                    .store(self.generation_start_ns, Ordering::Relaxed);
                // Publish the new stream only after all of its packets have been
                // pushed. The receiver preserves packets from this generation.
                self.timeline
                    .generation
                    .store(self.generation, Ordering::Release);
                self.published_generation = self.generation;
            }

            // Slot and generation publication happen before this release. The
            // renderer acquires this analyzed-time visibility watermark before
            // snapshotting the published sequence.
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
        fn publish_consumed(&self, consumed_before: u64) {
            if self.read_sequence != consumed_before {
                self.ring
                    .consumer
                    .consumed_ns
                    .store(self.consumed_ns, Ordering::Relaxed);
                self.ring
                    .consumer
                    .sequence
                    .store(self.read_sequence, Ordering::Release);
            }
        }

        fn flush(&mut self, published_sequence: u64) -> u64 {
            let mut newest_end_ns = 0;
            let consumed_before = self.read_sequence;

            while self.read_sequence != published_sequence {
                // SAFETY: the acquired published sequence transfers this slot
                // to the consumer until it publishes the advanced read sequence.
                let packet = unsafe { &mut *self.ring.slot(self.read_sequence) };
                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));
                newest_end_ns = newest_end_ns.max(packet.end_ns);
                self.read_sequence = self.read_sequence.wrapping_add(1);
            }
            self.publish_consumed(consumed_before);
            newest_end_ns
        }

        fn start_generation(
            &mut self,
            generation: u64,
            published_sequence: u64,
            spectrogram: &mut Spectrogram,
        ) {
            let consumed_before = self.read_sequence;

            // Generations are FIFO. Remove only obsolete packets and leave the
            // first packet from this or a later published generation available.
            while self.read_sequence != published_sequence {
                // SAFETY: this slot is within the acquired published snapshot.
                let packet = unsafe { &mut *self.ring.slot(self.read_sequence) };
                let generation_distance = packet.generation.wrapping_sub(generation);
                if generation_distance < (1_u64 << 63) {
                    break;
                }

                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));
                self.read_sequence = self.read_sequence.wrapping_add(1);
            }
            self.publish_consumed(consumed_before);

            self.generation = generation;
            self.rendered_ns = self.timeline.generation_start_ns.load(Ordering::Relaxed);
            spectrogram.clear();
        }

        /// Implements editor fresh-start semantics and is also used for stream
        /// generation changes.
        pub(crate) fn fresh_start(&mut self, spectrogram: &mut Spectrogram) {
            let generation = self.timeline.generation.load(Ordering::Acquire);
            let watermark_before = self.timeline.analyzed_ns.load(Ordering::Acquire);
            let published_sequence = self.ring.published_sequence.load(Ordering::Acquire);
            let flushed_end = self.flush(published_sequence);
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
            let (watermark, published_sequence) = loop {
                let current_generation = self.timeline.generation.load(Ordering::Acquire);
                if current_generation != self.generation {
                    let published_sequence = self.ring.published_sequence.load(Ordering::Acquire);
                    self.start_generation(current_generation, published_sequence, spectrogram);
                }

                let watermark = self.timeline.analyzed_ns.load(Ordering::Acquire);
                let published_sequence = self.ring.published_sequence.load(Ordering::Acquire);
                // The producer publishes a generation just before its analyzed
                // watermark. Retry if those two loads straddled that boundary.
                if self.timeline.generation.load(Ordering::Acquire) == self.generation {
                    break (watermark, published_sequence);
                }
            };
            // Bound this frame's work, and never consume a packet from a batch
            // whose analyzed-time watermark has not yet been published.
            let packets_to_drain = published_sequence
                .wrapping_sub(self.read_sequence)
                .min(TRANSPORT_CAPACITY as u64);
            let consumed_before = self.read_sequence;

            for _ in 0..packets_to_drain {
                // SAFETY: this slot is within the acquired published snapshot
                // and remains consumer-owned until the final release store.
                let packet = unsafe { &mut *self.ring.slot(self.read_sequence) };
                if packet.end_ns > watermark {
                    break;
                }

                self.consumed_ns = self
                    .consumed_ns
                    .saturating_add(packet.end_ns.saturating_sub(packet.start_ns));
                self.read_sequence = self.read_sequence.wrapping_add(1);

                if packet.generation != self.generation || packet.end_ns <= self.rendered_ns {
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
                spectrogram.rotate_in_place(&mut packet.analysis);
            }

            self.publish_consumed(consumed_before);

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

        #[cfg(test)]
        pub(crate) fn sequence(&self) -> u64 {
            self.read_sequence
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

    #[test]
    fn native_slot_sequences_wrap_without_reordering() {
        let (mut sink, mut receiver) = native_transport(8);
        let initial_sequence = u64::MAX - 1;
        native::set_empty_sequence_for_test(&mut sink, &mut receiver, initial_sequence);

        let mut spectrogram = Spectrogram::new(8, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);
        sink.begin_batch();
        submit_unpublished(&mut sink, 1.0, Duration::from_nanos(1));
        submit_unpublished(&mut sink, 2.0, Duration::from_nanos(1));
        submit_unpublished(&mut sink, 3.0, Duration::from_nanos(1));
        sink.finish_batch();
        receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(sink.sequence(), 1);
        assert_eq!(receiver.sequence(), 1);
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
    fn delivered_slot_swaps_with_oldest_spectrogram_buffer() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(3, 8);
        let mut metrics = AnalysisMetrics::default();
        receiver.fresh_start(&mut spectrogram);
        assert_eq!(spectrogram.len(), 3);

        let slot_pointer = sink.slot_data_pointer(0);
        let oldest_pointer = spectrogram.at_age(2).unwrap().data.as_ptr();
        submit(&mut sink, 1.0, Duration::from_millis(10));
        receiver.drain_into(&mut spectrogram, &mut metrics);

        assert_eq!(spectrogram.newest().data.as_ptr(), slot_pointer);
        assert_eq!(sink.slot_data_pointer(0), oldest_pointer);
    }

    #[test]
    fn fresh_start_does_not_consume_an_unpublished_batch() {
        let (mut sink, mut receiver) = native_transport(8);
        let mut spectrogram = Spectrogram::new(3, 8);
        let mut metrics = AnalysisMetrics::default();

        sink.begin_batch();
        submit_unpublished(&mut sink, 1.0, Duration::from_millis(10));
        receiver.fresh_start(&mut spectrogram);
        assert_eq!(receiver.rendered_ns(), 0);
        assert_eq!(receiver.sequence(), 0);

        sink.finish_batch();
        receiver.drain_into(&mut spectrogram, &mut metrics);
        assert_eq!(spectrogram.newest().data[0].0, 1.0);
        assert_eq!(receiver.rendered_ns(), 10_000_000);
    }
}
