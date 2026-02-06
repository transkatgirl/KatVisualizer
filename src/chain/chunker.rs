// Copied from nih_plug's StftHelper

use std::cmp;

pub trait StftInput {
    fn num_samples(&self) -> usize;
    fn num_channels(&self) -> usize;
    unsafe fn get_sample_unchecked(&self, channel: usize, sample_idx: usize) -> f32;
}

pub trait StftInputMut: StftInput {
    unsafe fn get_sample_unchecked_mut(&mut self, channel: usize, sample_idx: usize) -> &mut f32;
}

pub struct StftHelper<const NUM_SIDECHAIN_INPUTS: usize = 0> {
    main_input_ring_buffers: Vec<Vec<f32>>,
    main_output_ring_buffers: Vec<Vec<f32>>,
    sidechain_ring_buffers: [Vec<Vec<f32>>; NUM_SIDECHAIN_INPUTS],
    scratch_buffer: Vec<f32>,
    padding_buffers: Vec<Vec<f32>>,
    current_pos: usize,
    padding: usize,
}

struct NoSidechain;

impl StftInput for [&[f32]] {
    #[inline]
    fn num_samples(&self) -> usize {
        if self.is_empty() { 0 } else { self[0].len() }
    }

    #[inline]
    fn num_channels(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn get_sample_unchecked(&self, channel: usize, sample_idx: usize) -> f32 {
        unsafe { *self.get_unchecked(channel).get_unchecked(sample_idx) }
    }
}

impl StftInput for [&mut [f32]] {
    #[inline]
    fn num_samples(&self) -> usize {
        if self.is_empty() { 0 } else { self[0].len() }
    }

    #[inline]
    fn num_channels(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn get_sample_unchecked(&self, channel: usize, sample_idx: usize) -> f32 {
        unsafe { *self.get_unchecked(channel).get_unchecked(sample_idx) }
    }
}

impl StftInputMut for [&mut [f32]] {
    #[inline]
    unsafe fn get_sample_unchecked_mut(&mut self, channel: usize, sample_idx: usize) -> &mut f32 {
        unsafe {
            self.get_unchecked_mut(channel)
                .get_unchecked_mut(sample_idx)
        }
    }
}

impl StftInput for NoSidechain {
    fn num_samples(&self) -> usize {
        0
    }

    fn num_channels(&self) -> usize {
        0
    }

    unsafe fn get_sample_unchecked(&self, _channel: usize, _sample_idx: usize) -> f32 {
        0.0
    }
}

impl<const NUM_SIDECHAIN_INPUTS: usize> StftHelper<NUM_SIDECHAIN_INPUTS> {
    pub fn new(num_channels: usize, max_block_size: usize, max_padding: usize) -> Self {
        assert_ne!(num_channels, 0);
        assert_ne!(max_block_size, 0);

        Self {
            main_input_ring_buffers: vec![vec![0.0; max_block_size]; num_channels],
            main_output_ring_buffers: vec![vec![0.0; max_block_size]; num_channels],
            sidechain_ring_buffers: [(); NUM_SIDECHAIN_INPUTS]
                .map(|_| vec![vec![0.0; max_block_size]; num_channels]),
            scratch_buffer: vec![0.0; max_block_size + max_padding],
            padding_buffers: vec![vec![0.0; max_padding]; num_channels],

            current_pos: 0,
            padding: max_padding,
        }
    }
    pub fn set_block_size(&mut self, block_size: usize) {
        assert!(block_size <= self.main_input_ring_buffers[0].capacity());

        self.update_buffers(block_size);
    }
    /*pub fn set_padding(&mut self, padding: usize) {
        assert!(padding <= self.padding_buffers[0].capacity());

        self.padding = padding;
        self.update_buffers(self.main_input_ring_buffers[0].len());
    }
    pub fn num_channels(&self) -> usize {
        self.main_input_ring_buffers.len()
    }
    pub fn max_block_size(&self) -> usize {
        self.main_input_ring_buffers.capacity()
    }
    pub fn max_padding(&self) -> usize {
        self.padding_buffers[0].capacity()
    }*/
    pub fn latency_samples(&self) -> u32 {
        self.main_input_ring_buffers[0].len() as u32
    }
    pub fn process_overlap_add<M, F>(
        &mut self,
        main_buffer: &mut M,
        overlap_times: usize,
        mut process_cb: F,
    ) where
        M: StftInputMut + ?Sized,
        F: FnMut(usize, &mut [f32]),
    {
        self.process_overlap_add_sidechain(
            main_buffer,
            [&NoSidechain; NUM_SIDECHAIN_INPUTS],
            overlap_times,
            |channel_idx, sidechain_idx, real_fft_scratch_buffer| {
                if sidechain_idx.is_none() {
                    process_cb(channel_idx, real_fft_scratch_buffer);
                }
            },
        );
    }
    pub fn process_overlap_add_sidechain<M, S, F>(
        &mut self,
        main_buffer: &mut M,
        sidechain_buffers: [&S; NUM_SIDECHAIN_INPUTS],
        overlap_times: usize,
        mut process_cb: F,
    ) where
        M: StftInputMut + ?Sized,
        S: StftInput,
        F: FnMut(usize, Option<usize>, &mut [f32]),
    {
        assert_eq!(
            main_buffer.num_channels(),
            self.main_input_ring_buffers.len()
        );
        assert!(overlap_times > 0);

        let main_buffer_len = main_buffer.num_samples();
        let num_channels = main_buffer.num_channels();
        let block_size = self.main_input_ring_buffers[0].len();
        let window_interval = (block_size / overlap_times) as i32;
        let mut already_processed_samples = 0;
        while already_processed_samples < main_buffer_len {
            let remaining_samples = main_buffer_len - already_processed_samples;
            let samples_until_next_window = ((window_interval - self.current_pos as i32 - 1)
                .rem_euclid(window_interval)
                + 1) as usize;
            let samples_to_process = samples_until_next_window.min(remaining_samples);

            for sample_offset in 0..samples_to_process {
                for channel_idx in 0..num_channels {
                    let sample = unsafe {
                        main_buffer.get_sample_unchecked_mut(
                            channel_idx,
                            already_processed_samples + sample_offset,
                        )
                    };
                    let input_ring_buffer_sample = unsafe {
                        self.main_input_ring_buffers
                            .get_unchecked_mut(channel_idx)
                            .get_unchecked_mut(self.current_pos + sample_offset)
                    };
                    let output_ring_buffer_sample = unsafe {
                        self.main_output_ring_buffers
                            .get_unchecked_mut(channel_idx)
                            .get_unchecked_mut(self.current_pos + sample_offset)
                    };
                    *input_ring_buffer_sample = *sample;
                    *sample = *output_ring_buffer_sample;

                    *output_ring_buffer_sample = 0.0;
                }
            }

            for (sidechain_buffer, sidechain_ring_buffers) in sidechain_buffers
                .iter()
                .zip(self.sidechain_ring_buffers.iter_mut())
            {
                for sample_offset in 0..samples_to_process {
                    for channel_idx in 0..num_channels {
                        let sample = unsafe {
                            sidechain_buffer.get_sample_unchecked(
                                channel_idx,
                                already_processed_samples + sample_offset,
                            )
                        };
                        let ring_buffer_sample = unsafe {
                            sidechain_ring_buffers
                                .get_unchecked_mut(channel_idx)
                                .get_unchecked_mut(self.current_pos + sample_offset)
                        };
                        *ring_buffer_sample = sample;
                    }
                }
            }

            already_processed_samples += samples_to_process;
            self.current_pos = (self.current_pos + samples_to_process) % block_size;

            if samples_to_process == samples_until_next_window {
                for (sidechain_idx, sidechain_ring_buffers) in
                    self.sidechain_ring_buffers.iter().enumerate()
                {
                    for (channel_idx, sidechain_ring_buffer) in
                        sidechain_ring_buffers.iter().enumerate()
                    {
                        copy_ring_to_scratch_buffer(
                            &mut self.scratch_buffer,
                            self.current_pos,
                            sidechain_ring_buffer,
                        );
                        if self.padding > 0 {
                            self.scratch_buffer[block_size..].fill(0.0);
                        }

                        process_cb(channel_idx, Some(sidechain_idx), &mut self.scratch_buffer);
                    }
                }

                for (channel_idx, ((input_ring_buffer, output_ring_buffer), padding_buffer)) in self
                    .main_input_ring_buffers
                    .iter()
                    .zip(self.main_output_ring_buffers.iter_mut())
                    .zip(self.padding_buffers.iter_mut())
                    .enumerate()
                {
                    copy_ring_to_scratch_buffer(
                        &mut self.scratch_buffer,
                        self.current_pos,
                        input_ring_buffer,
                    );
                    if self.padding > 0 {
                        self.scratch_buffer[block_size..].fill(0.0);
                    }

                    process_cb(channel_idx, None, &mut self.scratch_buffer);

                    if self.padding > 0 {
                        let padding_to_copy = cmp::min(self.padding, block_size);
                        for (scratch_sample, padding_sample) in self.scratch_buffer
                            [..padding_to_copy]
                            .iter_mut()
                            .zip(&mut padding_buffer[..padding_to_copy])
                        {
                            *scratch_sample += *padding_sample;
                        }

                        padding_buffer.copy_within(padding_to_copy.., 0);

                        padding_buffer[self.padding - padding_to_copy..].fill(0.0);
                    }

                    add_scratch_to_ring_buffer(
                        &self.scratch_buffer,
                        self.current_pos,
                        output_ring_buffer,
                    );

                    if self.padding > 0 {
                        for (padding_sample, scratch_sample) in padding_buffer
                            .iter_mut()
                            .zip(&mut self.scratch_buffer[block_size..])
                        {
                            *padding_sample += *scratch_sample;
                        }
                    }
                }
            }
        }
    }
    pub fn process_analyze_only<B, F>(
        &mut self,
        buffer: &B,
        overlap_times: usize,
        mut analyze_cb: F,
    ) where
        B: StftInput + ?Sized,
        F: FnMut(usize, &mut [f32]),
    {
        assert_eq!(buffer.num_channels(), self.main_input_ring_buffers.len());
        assert!(overlap_times > 0);

        let main_buffer_len = buffer.num_samples();
        let num_channels = buffer.num_channels();
        let block_size = self.main_input_ring_buffers[0].len();
        let window_interval = (block_size / overlap_times) as i32;
        let mut already_processed_samples = 0;
        while already_processed_samples < main_buffer_len {
            let remaining_samples = main_buffer_len - already_processed_samples;
            let samples_until_next_window = ((window_interval - self.current_pos as i32 - 1)
                .rem_euclid(window_interval)
                + 1) as usize;
            let samples_to_process = samples_until_next_window.min(remaining_samples);

            for sample_offset in 0..samples_to_process {
                for channel_idx in 0..num_channels {
                    let sample = unsafe {
                        buffer.get_sample_unchecked(
                            channel_idx,
                            already_processed_samples + sample_offset,
                        )
                    };
                    let input_ring_buffer_sample = unsafe {
                        self.main_input_ring_buffers
                            .get_unchecked_mut(channel_idx)
                            .get_unchecked_mut(self.current_pos + sample_offset)
                    };
                    *input_ring_buffer_sample = sample;
                }
            }

            already_processed_samples += samples_to_process;
            self.current_pos = (self.current_pos + samples_to_process) % block_size;

            if samples_to_process == samples_until_next_window {
                for (channel_idx, input_ring_buffer) in
                    self.main_input_ring_buffers.iter().enumerate()
                {
                    copy_ring_to_scratch_buffer(
                        &mut self.scratch_buffer,
                        self.current_pos,
                        input_ring_buffer,
                    );
                    if self.padding > 0 {
                        self.scratch_buffer[block_size..].fill(0.0);
                    }

                    analyze_cb(channel_idx, &mut self.scratch_buffer);
                }
            }
        }
    }
    fn update_buffers(&mut self, block_size: usize) {
        for main_ring_buffer in &mut self.main_input_ring_buffers {
            main_ring_buffer.resize(block_size, 0.0);
            main_ring_buffer.fill(0.0);
        }
        for main_ring_buffer in &mut self.main_output_ring_buffers {
            main_ring_buffer.resize(block_size, 0.0);
            main_ring_buffer.fill(0.0);
        }
        for sidechain_ring_buffers in &mut self.sidechain_ring_buffers {
            for sidechain_ring_buffer in sidechain_ring_buffers {
                sidechain_ring_buffer.resize(block_size, 0.0);
                sidechain_ring_buffer.fill(0.0);
            }
        }
        self.scratch_buffer.resize(block_size + self.padding, 0.0);
        self.scratch_buffer.fill(0.0);

        for padding_buffer in &mut self.padding_buffers {
            padding_buffer.resize(self.padding, 0.0);
            padding_buffer.fill(0.0);
        }

        self.current_pos = 0;
    }
}

#[inline]
fn copy_ring_to_scratch_buffer(
    scratch_buffer: &mut [f32],
    current_pos: usize,
    ring_buffer: &[f32],
) {
    let block_size = ring_buffer.len();
    let num_copy_before_wrap = block_size - current_pos;
    scratch_buffer[0..num_copy_before_wrap].copy_from_slice(&ring_buffer[current_pos..block_size]);
    scratch_buffer[num_copy_before_wrap..block_size].copy_from_slice(&ring_buffer[0..current_pos]);
}

#[inline]
fn add_scratch_to_ring_buffer(scratch_buffer: &[f32], current_pos: usize, ring_buffer: &mut [f32]) {
    let block_size = ring_buffer.len();
    let num_copy_before_wrap = block_size - current_pos;
    for (scratch_sample, ring_sample) in scratch_buffer[0..num_copy_before_wrap]
        .iter()
        .zip(&mut ring_buffer[current_pos..block_size])
    {
        *ring_sample += *scratch_sample;
    }
    for (scratch_sample, ring_sample) in scratch_buffer[num_copy_before_wrap..block_size]
        .iter()
        .zip(&mut ring_buffer[0..current_pos])
    {
        *ring_sample += *scratch_sample;
    }
}
