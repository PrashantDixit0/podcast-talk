from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

# run the pipeline on an audio file
diarization = pipeline("demo.mp3")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)