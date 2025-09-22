1 | Remove Transitional Dependency
| Files: Ref/whisperX/ (entire directory), scripts/asr_whisperx.py
| Implementation: src/omoai/asr/whisperx_runner.py uses faster-whisper directly; can remove Ref/whisperX after verifying all functionality is ported
| Status: Enhancement - Remove dependency on Ref/whisperX reference implementation

2 | Eliminate PYTHONPATH=Ref/whisperX requirement
| Files: scripts/asr_whisperx.py (remove PYTHONPATH export)
| Implementation: src/omoai/asr/whisperx_runner.py already uses faster-whisper directly without PYTHONPATH
| Status: Enhancement - Remove environment variable dependency

3 | Migrate remaining assets (mel filters) to proper location
| Files: Ref/whisperX/whisperx/assets/mel_filters.npz
| Implementation: Move to assets/ directory or integrate into models/whisperx/EraX/
| Status: Migration - Asset relocation for standalone operation

4 | Update documentation to reflect standalone operation
| Files: docs/whisperx-integration.md
| Implementation: Update to reflect direct faster-whisper usage without Ref/whisperX dependency
| Status: Documentation - Update implementation details

5 |
6 | Advanced Output Formats
| Files: config.yaml (output.formats), scripts/post.py, Ref/whisperX/whisperx/utils.py, Ref/whisperX/whisperx/__main__.py
| Implementation: Reference WhisperX supports [txt, srt, vtt, tsv, json] and optional [aud]; it also supports output_format=all. Our pipeline uses [json, text, srt, vtt, md] where "text" and "md" are pipeline-specific (not native WhisperX). Map text→txt when aligning behaviors.
| Status: Clarification - Align format names/mapping with reference

7 |
8 | Add SRT/VTT subtitle generation
| Files: scripts/post.py, Ref/whisperX/whisperx/utils.py (WriteSRT/WriteVTT), Ref/whisperX/whisperx/SubtitlesProcessor.py
| Implementation: Reference includes SRT/VTT writers with word-highlighting and line constraints; align our post.py writing/formatting with reference behavior and options.
| Status: Enhancement - Align subtitle export with reference

9 | Implement word-level highlighting options
| Files: src/omoai/integrations/alignment.py, config.yaml (alignment config)
| Implementation: alignment.return_char_alignments config option exists; need to implement highlighting in output
| Status: Enhancement - Add word-level highlighting features

10 | Support multiple output formats simultaneously
| Files: config.yaml (output.formats), scripts/post.py
| Implementation: Already configured to generate multiple formats; need to verify simultaneous generation works
| Status: Enhancement - Verify and improve multi-format support

1 |
12 | Performance Optimization
| Files: src/omoai/asr/whisperx_runner.py, src/omoai/integrations/vad.py
| Implementation: VAD chunking and overlap handling already implemented
| Status: Optimization - Improve existing performance features

13 |
14 | Add max_duration chunking for long audio files
| Files: src/omoai/asr/whisperx_runner.py (lines 210, 266-275), config.yaml (asr.total_batch_duration_s)
| Implementation: Already has VAD-based chunking; can add max_duration parameter for forced chunking
| Status: Enhancement - Add configurable audio chunking

15 | Wildcard emissions in alignment
| Files: Ref/whisperX/whisperx/alignment.py (get_wildcard_emission), src/omoai/integrations/alignment.py
| Implementation: Already implemented (used in trellis/backtrack). Verify parity and keep covered by tests.
| Status: Verification - Confirm behavior parity

16 | Optimize VAD merging with overlap handling
| Files: src/omoai/integrations/vad.py (merge_chunks, apply_overlap functions)
| Implementation: Functions exist but can be optimized for better overlap handling
| Status: Optimization - Improve VAD merging algorithms

17 |
18 | Sentence-Level Segmentation
| Files: Ref/whisperX/whisperx/alignment.py, src/omoai/integrations/alignment.py
| Implementation: Reference uses NLTK Punkt sentence tokenizer; our code currently uses a simple splitter. Integrate Punkt to match reference behavior.
| Status: Enhancement - Add NLTK-based segmentation

19 |
20 | Integrate NLTK sentence tokenization
| Files: src/omoai/integrations/alignment.py, pyproject.toml (nltk dependency)
| Implementation: Use PunktSentenceTokenizer (+ abbreviations) to compute sentence spans as in reference.
| Status: Enhancement - Add NLTK integration

21 | Add sentence boundary detection in alignment
| Files: src/omoai/integrations/alignment.py
| Implementation: After NLTK integration, propagate per-sentence start/end and aggregate segments like the reference.
| Status: Enhancement - Add sentence boundary detection

2 | Support sentence-level timestamps
| Files: src/omoai/integrations/alignment.py, scripts/post.py
| Implementation: Reference produces sentence-level subsegments (start/end, words, optional chars); mirror this structure and expose timestamps.
| Status: Enhancement - Add sentence-level timing

23 |
24 | Advanced Configuration
| Files: src/omoai/config/schemas.py, config.yaml
| Implementation: Configuration schemas exist but can be extended with advanced options
| Status: Enhancement - Add advanced configuration options

25 |
26 | Add temperature sampling options
| Files: src/omoai/config/schemas.py (SamplingConfig), config.yaml
| Implementation: SamplingConfig already exists with temperature; need to connect to ASR/LLM processes
| Status: Enhancement - Connect temperature settings to processing

27 | Implement compression ratio thresholds
| Files: src/omoai/asr/whisperx_runner.py, src/omoai/config/schemas.py
| Implementation: Add compression ratio checking to detect hallucinations
| Status: Enhancement - Add quality control measures

28 | Support hallucination silence detection
| Files: src/omoai/asr/whisperx_runner.py, src/omoai/integrations/vad.py
| Implementation: Add silence detection to filter hallucinations
| Status: Enhancement - Add hallucination filtering

29 |
30 | Subtitle Processing
| Files: Ref/whisperX/whisperx/SubtitlesProcessor.py, scripts/post.py
| Implementation: Port SubtitlesProcessor features to post.py
| Status: Enhancement - Implement subtitle processing

31 |
32 | Implement SubtitlesProcessor.py features
| Files: Ref/whisperX/whisperx/SubtitlesProcessor.py, scripts/post.py
| Implementation: Port advanced subtitle splitting and formatting features
| Status: Enhancement - Port reference implementation

33 | Add max line width/count constraints
| Files: Ref/whisperX/whisperx/SubtitlesProcessor.py (max_line_length), scripts/post.py
| Implementation: Add line width constraints to subtitle generation
| Status: Enhancement - Add subtitle formatting constraints

34 | Support language-specific formatting rules
| Files: Ref/whisperX/whisperx/SubtitlesProcessor.py (complex_script_languages), scripts/post.py
| Implementation: Add language-specific formatting for subtitle generation
| Status: Enhancement - Add localization support
