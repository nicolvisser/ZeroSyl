# !/bin/bash

assert_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: Required directory '$1' is missing." >&2
        exit 1
    fi
}


# ---------------------------------------------------------------------------------------------------------------------------------------------

# For boundary and syllable discovery evaluations you need the test-clean and test-other sets of LibriSpeech from https://www.openslr.org/12
# You can download the alignments from https://storage.googleapis.com/zerospeech-checkpoints/alignments-librispeech-dev-test.tar.xz

assert_dir "data/waveforms/LibriSpeech/test-clean"
assert_dir "data/waveforms/LibriSpeech/test-other"
assert_dir "data/alignments/LibriSpeech/test-clean"
assert_dir "data/alignments/LibriSpeech/test-other"

echo "Encoding LibriSpeech/test*/ with ZeroSylCollapsed.."
zerosyl encode data/waveforms/LibriSpeech/ output/segments/ZeroSylCollapsed/LibriSpeech/ --input-pattern test*/**/*.flac

echo "Encoding LibriSpeech/test*/ with ZeroSylDiscrete (where silences are not collapsed)..."
zerosyl encode data/waveforms/LibriSpeech/ output/segments/ZeroSylDiscrete/LibriSpeech/ --input-pattern test*/**/*.flac --output discrete

echo Evaluating boundaries of ZeroSyl...
zerosyl evaluate boundaries output/segments/ZeroSylCollapsed/LibriSpeech/ data/alignments/LibriSpeech/ --segments-pattern test*/**/*.pt --textgrid-pattern test*/**/*.TextGrid --constant-shift -0.005

echo Evaluating syllable discovery of ZeroSylCollapsed...
zerosyl evaluate clustering output/segments/ZeroSylCollapsed/LibriSpeech/ data/alignments/LibriSpeech/ --segments-pattern test*/**/*.pt --textgrid-pattern test*/**/*.TextGrid

echo Evaluating syllable discovery of ZeroSylDiscrete...
zerosyl evaluate clustering output/segments/ZeroSylDiscrete/LibriSpeech/ data/alignments/LibriSpeech/ --segments-pattern test*/**/*.pt --textgrid-pattern test*/**/*.TextGrid


# ---------------------------------------------------------------------------------------------------------------------------------------------

# For sWUGGY and sBLIMP evaluations you need the lexical/dev and syntactic/dev sets of sLM21-dataset from https://github.com/zerospeech/benchmarks
# Using the zrc toolkit, download the sLM21-dataset:
# zrc datasets:pull sLM21
# You should also create the submission directories:
# zrc submission:init sLM21 output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed
# zrc submission:init sLM21 output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed
# zrc submission:init sLM21 output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed
# after running this script you will also have to use the zrc toolkit to get the scores for the sWUGGY and sBLIMP evaluations

assert_dir ~/zr-data/datasets/sLM21-dataset/lexical/dev
assert_dir ~/zr-data/datasets/sLM21-dataset/syntactic/dev
assert_dir output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed
assert_dir output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed
assert_dir output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed

echo "Encoding sLM21-dataset/lexical/dev with ZeroSylCollapsed..."
zerosyl encode ~/zr-data/datasets/sLM21-dataset/lexical/dev output/segments/ZeroSylCollapsed/sLM21-dataset/lexical/dev

echo "Encoding sLM21-dataset/syntactic/dev with ZeroSylCollapsed..."
zerosyl encode ~/zr-data/datasets/sLM21-dataset/syntactic/dev output/segments/ZeroSylCollapsed/sLM21-dataset/syntactic/dev

echo "Computing loglikelihoods for sLM21-dataset/lexcial/dev with the language model trained on 600h..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/lexical/dev output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed/lexical/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-600h-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Computing loglikelihoods for sLM21-dataset/syntactic/dev with the language model trained on 600h..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/syntactic/dev output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed/syntactic/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-600h-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Computing loglikelihoods for sLM21-dataset/lexcial/dev with the language model trained on 6kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/lexical/dev output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed/lexical/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-6kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Computing loglikelihoods for sLM21-dataset/syntactic/dev with the language model trained on 6kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/syntactic/dev output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed/syntactic/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-6kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Computing loglikelihoods for sLM21-dataset/lexcial/dev with the language model trained on 60kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/lexical/dev output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed/lexical/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-60kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Computing loglikelihoods for sLM21-dataset/syntactic/dev with the language model trained on 60kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/sLM21-dataset/syntactic/dev output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed/syntactic/dev.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-60kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

# finally evaluate with the zrc toolkit
# zrc benchmarks:run sLM21 output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed -s dev -t lexical syntactic
# zrc benchmarks:run sLM21 output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed -s dev -t lexical syntactic
# zrc benchmarks:run sLM21 output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed -s dev -t lexical syntactic



# ---------------------------------------------------------------------------------------------------------------------------------------------

# For the tSC evaluation you can download the tStoryCloze dataset from https://github.com/slp-rl/SpokenStoryCloze

assert_dir data/waveforms/tSC

echo "Encoding sLM21-dataset/lexical/dev with ZeroSylCollapsed..."
zerosyl encode data/waveforms/tSC output/segments/ZeroSylCollapsed/tSC

echo "Compute likelihoods for tSC with the language mode trained on 600h..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/tSC output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed/tSC.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-600h-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Compute likelihoods for tSC with the language mode trained on 6kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/tSC output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed/tSC.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-6kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo "Compute likelihoods for tSC with the language mode trained on 60kh..."
zerosyl evaluate loglikelihoods output/segments/ZeroSylCollapsed/tSC output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed/tSC.txt --checkpoint-path https://storage.googleapis.com/zerospeech-checkpoints/OPT-125M-LibriLight-60kh-ZeroSylCollapsed-v040-k-9116.pt --normalize

echo ZeroSyl LM trained on 600h:
zerosyl evaluate tsc output/submissions/OPT-125M-LibriLight-600h-ZeroSylCollapsed/tSC.txt

echo ZeroSyl LM trained on 6kh:
zerosyl evaluate tsc output/submissions/OPT-125M-LibriLight-6kh-ZeroSylCollapsed/tSC.txt

echo ZeroSyl LM trained on 60kh:
zerosyl evaluate tsc output/submissions/OPT-125M-LibriLight-60kh-ZeroSylCollapsed/tSC.txt

